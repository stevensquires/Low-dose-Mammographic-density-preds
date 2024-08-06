import loadModels
import dataGeneratorDataAug
import basicCode
import torch
import numpy as np
import time
def getModel(modelChoice,optim_choice,learning_rate,fineTuning): ## fineTuning is True or False (feature extraction)
    model=loadModels.returnModel(modelChoice,fineTuning)
    criterion=torch.nn.MSELoss()
#    criterion=torch.nn.L1Loss()
    if fineTuning==False:
        paramsToUpdate=[{'params':model.classifier.parameters()},{'params': model.aux_classifier.parameters()}]
    elif fineTuning==True:
        for param in model.parameters():
            param.requires_grad = True
    if optim_choice=='Adam':
        if fineTuning==False:
            optimizer = torch.optim.Adam(paramsToUpdate, lr=learning_rate)
        elif fineTuning==True:
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_choice=='SGD':
        if fineTuning==False:
            optimizer = torch.optim.SGD(paramsToUpdate, lr=learning_rate, momentum=0.9)
        elif fineTuning==True:
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    return model,optimizer,criterion

def getGenerators(batchSize,labelName,resolution):
    pathImages,pathPart=basicCode.getPaths('images',resolution)
    partition=basicCode.getPartitionLabels(pathPart,'partition.pkl')
    labels=basicCode.getLabels(pathPart,labelName)
    trainGen,valGen=dataGeneratorDataAug.makeGenerators(partition,labels,batchSize,pathImages)
    return trainGen,valGen
 
def runTraining(maxEpochs,model,optimizer,criterion,trainingGenerator,validationGenerator,gpuOrCpu,outputPath,outputName):
    nTotal=maxEpochs*(trainingGenerator.__len__())
    nTotEval=maxEpochs*(validationGenerator.__len__())
    errStore=np.zeros((nTotal,2),float)
    errStoreEval=np.zeros((nTotEval,2),float)
    if gpuOrCpu=='gpu':
        device=torch.device("cuda:0")
    elif gpuOrCpu=='cpu':
        device = torch.device("cpu")    
    model = model.to(device) 
    i,j=0,0
    time0=time.time()
    minVal=10000000
    for epoch in range(maxEpochs):
        model.train()
        with torch.set_grad_enabled(True):
            for local_batch, local_targets,ID in trainingGenerator:
                local_batch, local_targets = local_batch.float().to(device), local_targets.float().to(device)
                optimizer.zero_grad()
                local_outputs=model(local_batch).squeeze()
                local_outputs2=local_outputs                   
                temp=local_targets.squeeze()
#                print(local_outputs[0],temp[0])  
                loss = criterion(local_outputs2,temp)
                lossNp=loss.detach().cpu().numpy()
#                print(i,' of ',nTotal,lossNp)
                errStore[i,0]=i
                errStore[i,1]=lossNp
                loss.backward()
                optimizer.step()
                i+=1
        model.eval()
        with torch.set_grad_enabled(False):
            tempVal=0
            totalErr=0
            for local_batch, local_targets,ID in validationGenerator:
                local_batch, local_targets = local_batch.float().to(device), local_targets.float().to(device)
                local_outputs=model(local_batch).squeeze()
                local_outputs2=local_outputs
                temp=local_targets.squeeze()
                   
                loss = criterion(local_outputs2,temp)
                lossNp=loss.detach().cpu().numpy()
                print(j,' of eval',nTotal,lossNp)
                errStoreEval[j,0]=j
                errStoreEval[j,1]=lossNp
                totalErr=totalErr+lossNp
                j+=1
                tempVal+=1
        if totalErr<minVal:
            minVal=totalErr
            torch.save(model.state_dict(), outputPath+outputName+'.params')
        totalTime=(time.time()-time0)/60
        #if (epoch+1)==int(maxEpochs/2):
        #    optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']/2
        print(epoch,'of',maxEpochs,'Loss=',totalErr,'Total time=',totalTime)
    return model,errStore,errStoreEval

def saveAll(model,trErrs,valErrs,outputPath,outputName):
#    torch.save(model.state_dict(), outputPath+outputName+'.params')
    np.save(outputPath+outputName+'.npy',trErrs)
    np.save(outputPath+outputName+'Eval.npy',valErrs)
    

def main(modelChoice,batchSize,gpuOrCpu,optim_choice,learning_rate,maxEpochs,
         fineTuning,outName,labelName,resolution):
    model,optimizer,criterion=getModel(modelChoice,optim_choice,learning_rate,fineTuning)
    trainingGenerator,validationGenerator=getGenerators(batchSize,labelName,resolution)
    model,errStore,errStoreEval=runTraining(maxEpochs,model,optimizer,criterion,trainingGenerator,validationGenerator,gpuOrCpu,basicCode.getPaths('output',resolution),outName)
    saveAll(model,errStore,errStoreEval,basicCode.getPaths('output',resolution),outName)

    
    
    
    
    
    
