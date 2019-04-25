import numpy as np

def generateDataMultiNormal(n,k,limit):   #k classes,n for each class, abs(feature value) < limit
    data = np.zeros((n * k,2))
    cnt = 0
    stdev = 50
    for i in range(k):
        center = np.random.randint(0,limit,size = (1,2))
        data[cnt] = center
        cnt += 1
        #print(center)
        for j in range(n - 1):
            x = np.random.normal(center[0,0],stdev,1)
            y = np.random.normal(center[0,1],stdev,1)
            data[cnt,0] = x
            data[cnt,1] = y
            cnt += 1
    np.random.shuffle(data)
    return data

def generateDataMultiNormal1(n,k,limit):   #k classes,n for each class, abs(feature value) < limit
    data = np.zeros((2,n * k))
    cnt = 0
    stdev = 50
    for i in range(k):
        center = np.random.randint(0,limit,size = (1,2))
        data[:,cnt] = center
        cnt += 1
        #print(center)
        for j in range(n - 1):
            x = np.random.normal(center[0,0],stdev,1)
            y = np.random.normal(center[0,1],stdev,1)
            data[0,cnt] = x
            data[1,cnt] = y
            cnt += 1
    #np.random.shuffle(data[0])
    #np.random.shuffle(data[1])
    data = data.reshape(2,n,k)
    return data

def generateBatch(batchSize,n,k,limit):
    data = np.zeros((batchSize,2,n,k))
    for i in range(batchSize):
        data[i] = generateDataMultiNormal1(n,k,limit)
    return data
