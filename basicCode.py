import pickle

def getPaths(name,resolution):
    basePath='/ #### For CSF
    if name=='images':
        path1=basePath+'Images2/'+resolution+'/'
        path2=basePath+'Data/'
        return path1,path2
    elif name=='output':
        return basePath+'OutputSinglePredictor/'

def getPartitionLabels(path1,partitionFile):
    pkl_file1 = open(path1+partitionFile, 'rb')
    partition=pickle.load(pkl_file1)
    pkl_file1.close()
    return partition
    
def getLabels(path2,labelsName):
    pkl_file1 = open(path2+labelsName+'.pkl', 'rb') 
    labels=pickle.load(pkl_file1)
    pkl_file1.close()
    return labels  
    
    
    
