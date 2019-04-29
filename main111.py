import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import os
import shutil
import time
import yaml
import glob
import datetime
from collections import OrderedDict

import numpy
from PIL import Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import models

from segment_loader import SegmentDataLoader

from classes import RedirectableClasses 

from tensorboardX import SummaryWriter
import logging

from losses import BackgroundLoss, PairLoss, EmbeddingNormLoss

from memcached_dataset import McDataset
from tusimple import TuSimpleSegmentation
from utils import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, IterLRScheduler,DistributedGivenIterationSampler, simple_group_split, accuracy2d, DistributedSampler, color_image_fill
from utils import metrics, visualize_colored, trim_color, visualize_debug
from dist_utils import dist_init, reduce_gradients, DistModule
from meanshift_skylearn import *
from meanshift_shift_points_scaling import * 

import linklink as link

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--class-config', default='class_conf.d/simple_unify.yaml')
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--resume-opt', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--sync', action='store_true')
parser.add_argument('--port', default='23456', type=str)
parser.add_argument('--adam', action='store_true')

parser.add_argument('--instance-dims', type=int, default=20, help='dimensions used for distinguishing instances')
parser.add_argument('--sigma', type=float, default=0.2, help='threshold of pixel distance to distinguish two different objects')
parser.add_argument('--samples', type=int, default=1024, help='sample pixels chosen for each image')
parser.add_argument('--distance-limit', type=int, default=999, help='a distance limiting choice of pixels')
parser.add_argument('--rolling', action='store_true')
parser.add_argument('--spatial', action='store_true')
parser.add_argument('--fusion', action='store_true')
parser.add_argument('--crop', action='store_true')
parser.add_argument('--backbone', default='resnet50')
parser.add_argument('--rotation', type=int, default=180)

class ColorAugmentation(object):
    def __init__(self, eig_vec=None, eig_val=None):
        if eig_vec == None:
            eig_vec = torch.Tensor([
                [ 0.4009,  0.7192, -0.5675],
                [-0.8140, -0.0045, -0.5808],
                [ 0.4203, -0.6948, -0.5836],
            ])
        if eig_val == None:
            eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])
        self.eig_val = eig_val  # 1*3
        self.eig_vec = eig_vec  # 3*3

    def __call__(self, tensor):
        assert tensor.size(0) == 3
        alpha = torch.normal(means=torch.zeros_like(self.eig_val))*0.1
        quatity = torch.mm(self.eig_val*alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

class InstancerLoss(nn.Module):
    def __init__(self, class_dims, instance_dims, weights=None):
        super(InstancerLoss, self).__init__()
        weights_tensor = weight=torch.FloatTensor(weights).cuda() if weights is not None else None
        self.bg = nn.CrossEntropyLoss(weight=weights_tensor)
        self.pair = PairLoss(distance_limit=args.distance_limit, samples=args.samples, sigma=args.sigma, reduce=True)
        #self.pair = nn.DataParallel(self.pair).cuda()
        self.class_dims = class_dims
        self.instance_dims = instance_dims
    
    def forward(self, input, class_gt, instance_gt, detail=False, pairweight=1):
        assert input.shape[1] == self.class_dims + self.instance_dims
        bg_loss = self.bg(input[:, :self.class_dims, :, :], class_gt)
        pair_loss = self.pair(input[:, self.class_dims:, :, :], instance_gt) * pairweight
        #pair_loss = pair_loss.sum()
        #pair_loss = torch.zeros(1, requires_grad=True).cuda()
        if detail:
            return bg_loss+pair_loss, pair_loss, bg_loss
        else:
            return bg_loss+pair_loss

def main():
    global args, best_acc
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in config['common'].items():
        setattr(args, k, v)

    rank, world_size = dist_init()
    print('link link initialized\n')

    if hasattr(args, 'weights'):
        weights = args.weights
    else:
        weights = None

    # BN group
    bn_sync_stats = args.bn_sync_stats
    bn_group_size = args.bn_group_size
    bn_var_mode = args.bn_var_mode
    if bn_group_size == 1:
        bn_group = None
    else:
        assert world_size % bn_group_size == 0
        bn_group = simple_group_split(world_size, rank, world_size // bn_group_size)
    bn_var_mode = (link.syncbnVarMode_t.L1 
        if bn_var_mode == 'L1' 
        else link.syncbnVarMode_t.L2)

    
    args.class_config = RedirectableClasses(args.class_config)
    
    class_redirection, num_classes = args.class_config.redirect_classes()
    args.num_classes = num_classes

    # create model
    print("=> creating model '{}'".format(args.arch))
    input_size = tuple(args.input_size)
    model = models.__dict__[args.arch](num_classes=num_classes+args.instance_dims,
                                       spatial=args.spatial, backbone=args.backbone,
                bn_group_size=bn_group_size, 
                bn_group=bn_group, 
                bn_var_mode=bn_var_mode,
                bn_sync_stats=bn_sync_stats,
                )

    model.cuda()
    model = DistModule(model, args.sync)

    # define loss function (criterion) and optimizer
    criterion = InstancerLoss(class_dims=args.num_classes, instance_dims= args.instance_dims, weights=weights)

    optimizer = torch.optim.SGD(model.parameters(), args.base_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), args.base_lr)

    # optionally resume from a checkpoint
    last_iter = -1
    best_acc = 0
    if args.load_path:
        if args.resume_opt:
            best_acc, last_iter = load_state(args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    #train_dataset = TuSimpleSegmentation(
    #    base_dir=args.train_root,
    #    )
    #val_dataset = TuSimpleSegmentation(
    #    base_dir=args.val_root,
    #    split='val'
    #    )

    train_dataset = McDataset(args.train_root, args.train_source, no_mc=args.no_mc, redirection=class_redirection)
    val_dataset = McDataset(args.val_root, args.val_source, no_mc=args.no_mc, redirection=class_redirection)

    train_sampler = DistributedGivenIterationSampler(train_dataset, args.max_iter, args.batch_size, 
                                                     last_iter=last_iter)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = SegmentDataLoader(
        train_dataset, base_size=tuple(args.input_size), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler, scales=(1,1), cutout=True, rotation=args.rotation, crop=args.crop)

    val_loader = SegmentDataLoader(
        val_dataset, base_size=tuple(args.input_size), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    if args.warmup_steps > 0:
        gap = args.warmup_lr - args.base_lr
        warmup_mults = [(args.base_lr + (i+1)*gap/args.warmup_steps) / (args.base_lr + i*gap/args.warmup_steps) for i in range(args.warmup_steps)]
        warmup_steps = list(range(args.warmup_steps))
        args.lr_mults = warmup_mults + args.lr_mults
        args.lr_steps = warmup_steps + args.lr_steps

    lr_scheduler = IterLRScheduler(optimizer, args.lr_steps, args.lr_mults, last_iter=last_iter)

    if rank == 0:
        tb_logger = SummaryWriter(os.path.join(args.save_path, args.save_path+str(datetime.datetime.now())))
        logger = create_logger('global_logger', args.save_path+'/log.txt')
        logger.info('{}'.format(args))
    else:
        tb_logger = None

    if args.evaluate and args.fusion:
        validate(val_loader, model, criterion, fusion=True, heatup=True)
        validate(val_loader, model, criterion, fusion=True, heatup=False)
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, last_iter+1, tb_logger)

def train(train_loader, val_loader, model, criterion, optimizer, lr_scheduler, start_iter, tb_logger):

    global best_acc

    batch_time = AverageMeter(10)
    data_time = AverageMeter(10)
    losses = AverageMeter(10)
    acc_avg = AverageMeter(10)

    # switch to train mode
    model.train()

    world_size = link.get_world_size()
    rank = link.get_rank()

    logger = logging.getLogger('global_logger')

    end = time.time()
    for i, (input, class_target, instance_target, tid) in enumerate(train_loader):
        #if rank==0: print('entering loop again')
        curr_step = start_iter + i
        if not args.adam: lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]
        # measure data loading time
        data_time.update(time.time() - end)

        class_target = class_target.cuda(non_blocking=True)
        instance_target = instance_target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        input_var = torch.autograd.Variable(input)
        class_target_var = torch.autograd.Variable(class_target)
        instance_target_var = torch.autograd.Variable(instance_target)

        #if rank ==0: print('just before forwarding')

        # compute output
        output = model(input_var)

        #if rank==0: print('forwarded')

        # measure accuracy and record loss

        weight = 1
        if args.rolling:
            if curr_step<75000:
                weight = curr_step/75000

        loss = criterion(output, class_target_var, instance_target_var, pairweight=weight) / world_size
        #if rank==0: print('loss computed')
        acc = accuracy2d(output.data[:, :args.num_classes, :, :], class_target_var)

        reduced_loss = loss.data.clone()
        reduced_acc = acc.clone() / world_size

        link.allreduce(reduced_loss)
        link.allreduce(reduced_acc)

        losses.update(reduced_loss.item())
        acc_avg.update(reduced_acc.item())

        #if rank==0: print('before reducing gradient')

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        reduce_gradients(model)
        optimizer.step()

        #if rank==0: print('reduced gradients')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if curr_step % args.print_freq == 0 and rank == 0:
            tb_logger.add_scalar('loss_train', losses.avg, curr_step)
            tb_logger.add_scalar('iou_train', acc_avg.avg, curr_step)
            tb_logger.add_scalar('lr', current_lr, curr_step)
            logger.info('Iter: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc_avg.val:.4f} ({acc_avg.avg:.4f})\t'
                  'LR {lr:.4f}'.format(
                   curr_step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, acc_avg=acc_avg, lr=current_lr))

        if curr_step % args.val_freq == 0 and curr_step > 0:
            
            if rank==0 : save_checkpoint({
                    'step': curr_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, False, args.save_path+'/Tckpt')
         
            #visualize(output.data.cpu(), visualize_dir=args.visualize_dir, prefix='train.{}.'.format(rank), input=input.cpu(), gt=target.cpu())
            val_loss, iou, f1 = validate(curr_step,val_loader, model, criterion, tb_logger, step=curr_step)

            if not tb_logger is None:
                tb_logger.add_scalar('loss_val', val_loss, curr_step)
                tb_logger.add_scalar('iou_val', iou, curr_step)
                tb_logger.add_scalar('f1_val', f1, curr_step)


            if rank == 0:
                # remember best prec@1 and save checkpoint
                is_best = f1 > best_acc
                best_acc = max(f1, best_acc)
                save_checkpoint({
                    'step': curr_step,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.save_path+'/ckpt')

from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid, save_image

def validate(epc,val_loader, model, criterion, tb_logger=None, fusion=False, heatup=False, step=0):

    if fusion:
        print('In fusion eval mode')

    if heatup:
        print('heating the model')

    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    ap = AverageMeter(0)
    ar = AverageMeter(0)
    f1 = AverageMeter(0)

    aps = [AverageMeter(0) for _ in range(args.num_classes-1)]
    ars = [AverageMeter(0) for _ in range(args.num_classes-1)]

    # switch to evaluate mode
    model.eval()

    rank = link.get_rank()
    world_size = link.get_world_size()

    logger = logging.getLogger('global_logger')

    end = time.time()
    for i, (input, class_target, instance_target, tid) in enumerate(val_loader):
        class_target = class_target.cuda(non_blocking=True)
        instance_target = instance_target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        input_var = input
        class_target_var = class_target
        instance_target_var = instance_target

        tid = tid.cuda(non_blocking=True)

        #input_var = torch.autograd.Variable(input)
        #target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var, fusion=fusion, store=heatup, grouping=tid)

        if not heatup:
            # measure accuracy and record loss
            loss = criterion(output, class_target_var, instance_target_var) / world_size
            acc = accuracy2d(output.data[:, :args.num_classes, :, :], class_target)

            reduced_loss = loss.data.clone()
            reduced_acc = acc.clone() / world_size

            class_pred = output.data[:, :args.num_classes, :, :].max(1)[1]
            instance_features = output.data[:, args.num_classes:, :, :]

            if epc <= 40000:
                colored = color_image_fill(class_pred.cpu().numpy(), instance_features.cpu().numpy())
            else:
                colored = Meanshifter(0.2,1e-3,0.4).meanshift(class_pred.cpu().numpy(), instance_features.cpu().numpy())
            colored = trim_color(colored, class_pred.cpu().numpy())
            APs, ARs = metrics(class_pred.cpu().numpy(), colored, class_target.cpu().numpy(), instance_target.cpu().numpy(), num_classes=args.num_classes)

            APs = torch.Tensor(APs)
            ARs = torch.Tensor(ARs)

            reduced_metrics = torch.stack((APs, ARs)) / world_size

            link.allreduce(reduced_loss)
            link.allreduce(reduced_acc)
            link.allreduce(reduced_metrics)

            losses.update(reduced_loss.item())
            top1.update(reduced_acc.item())
            ap.update((reduced_metrics.sum(1)/reduced_metrics.shape[1])[0])
            ar.update((reduced_metrics.sum(1)/reduced_metrics.shape[1])[1])
            f1.update((ap.val*ar.val*2)/(ap.val+ar.val+1e-12))

            for cls in range(args.num_classes-1):
                aps[cls].update(reduced_metrics[0,cls])
                ars[cls].update(reduced_metrics[1,cls])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and rank == 0:
                logger.info('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'IOU {top1.val:.4f} ({top1.avg:.4f})\t'
                        'AP {ap.val:.4f} ({ap.avg:.4f})\t'
                        'AR {ar.val:.4f} ({ar.avg:.4f})\t'
                        'F1 {f1.val:.4f} ({f1.avg:.4f})\t'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1,
                        ap=ap,
                        ar=ar,
                        f1=f1,))
            #visualize(output.data.cpu(), visualize_dir=args.visualize_dir, prefix='{}.{}.'.format(i, rank), input=input.cpu(), gt=target.cpu())
            visualize_colored(colored, class_pred.cpu().numpy(), instance_target.cpu().numpy(), class_target.cpu().numpy(), input.cpu(), visualize_dir=args.visualize_dir, prefix='{}.{}.'.format(i, rank))
            visualize_debug(instance_features.cpu().numpy(), instance_target.cpu().numpy(), input.cpu(), visualize_dir=args.visualize_dir, prefix='{}.{}.'.format(i, rank))
            #numpy.save('instance_output', instance_features.cpu().numpy())

    if not heatup and rank == 0:
        logger.info(' * Loss {loss.avg:.4f}\tIOU {top1.avg:.4f}\tAP {ap.avg:.4f}\tAR {ar.avg:.4f}\tF1 {f1:.4f}'.format(top1=top1,
        loss=losses,
        ap=ap,
        ar=ar,
        f1=(ap.avg*ar.avg*2)/(ap.avg+ar.avg+1e-12),
        ))
        for cls in range(args.num_classes-1):
            logger.info(' -- CLASS {cls}: AP {ap:.4f} AR {ar:.4f} F1 {f1:.4f}'.format(
                cls= cls+1,
                ap = aps[cls].avg,
                ar = ars[cls].avg,
                f1 = (aps[cls].avg*ars[cls].avg*2)/(aps[cls].avg+ars[cls].avg+1e-12),
            ))
        if not tb_logger is None:
            totensor = ToTensor()
            path = args.visualize_dir
            imgs = []
            for idx, f in enumerate(glob.glob(os.path.join(path, 'union*.jpg'))):
                img = Image.open(f)
                imgs.append(totensor(img))
            grid = make_grid(imgs, nrow=1)
            tb_logger.add_image('res', grid, step)


    model.train()
    
    return losses.avg, top1.avg, (ap.avg*ar.avg*2)/(ap.avg+ar.avg+1e-12)

if __name__ == '__main__':
    main()
    link.finalize()
