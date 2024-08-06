import runTraining1

batchSize,optim_choice=15,'Adam' 
gpuOrCpu='gpu'
learningRates=[1e-4,1e-4,1e-4,1e-4,1e-4]
labelName='Averagedscores'
versions=['v4a4','v4a1','v4a2','v4a4','v4a5']
for modelChoice in ['ResNet']: #### for these have removed data-augmentation
    for i,resolution in enumerate(['Resolution4','Resolution5','Resolution5','Resolution5','Resolution5']):
        learning_rate=learningRates[i]
        version=versions[i]
        maxEpochs=20
        fineTuning=True
        outName=modelChoice+'OnePredictor'+resolution+version
        runTraining1.main(modelChoice,batchSize,gpuOrCpu,optim_choice,
                          learning_rate,maxEpochs,fineTuning,outName,labelName,resolution)   





















