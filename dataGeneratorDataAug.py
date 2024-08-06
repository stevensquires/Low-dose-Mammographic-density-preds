import numpy as np
from torch.utils import data
from torchvision import transforms
## this class returns a 3 channel image, normalised to be fed into the pytorch 
## models.

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, IDs,labels,inPath,augmentOrNot): ### 
        'Initialization'
        self.IDs = IDs
        self.in_path = inPath
        self.Labels= labels
        self.augmentOrNot=augmentOrNot
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.IDs)

  
  
  def imageProcessing(self,ID):
      # Load data and get label
        inPath=self.in_path
        img=np.load(inPath+ID)
        augmentOrNot=self.augmentOrNot
        ### data augmentation 1: left/right flips
        if augmentOrNot:
            if np.random.uniform()>0.5: 
                img=np.fliplr(img)
            
        ### data augmentation 2: 90 degree rotations

            #randInt=np.random.randint(0, high=4)
            #img=np.rot90(img,randInt)
        ### data augmentation 3: addition of random noise - only to the image, not the target
            randNoise=np.random.normal(loc=0.0, scale=0.01, size=(img.shape[0],img.shape[1]))
            img=img+randNoise
        img=img.reshape(img.shape[0],img.shape[1],1)
        img=np.repeat(img,3,axis=2)
        trf = transforms.Compose([transforms.ToTensor(), 
                         transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])])
        inImage = trf(img)#.unsqueeze(0) 
        inImage.float()
        return inImage
       
  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample    
#        tarPath=self.tar_path
        ID = self.IDs[index]
        label=self.Labels[ID]
        inImage= self.imageProcessing(ID)
        return inImage,label,ID

### call to make generators: partition is the training and testing split

        
def makeGenerators(partition,labels,batchSize,path1): ### 
    training_set=Dataset(partition['Training'],labels,path1,True)
    paramsTrain = {'batch_size':batchSize,'shuffle': False}
    training_generator = data.DataLoader(training_set, **paramsTrain)
    validation_set=Dataset(partition['Validation'],labels,path1,False)
    paramsValidation = {'batch_size':batchSize,'shuffle': False}
    validation_generator = data.DataLoader(validation_set, **paramsValidation)
    return training_generator,validation_generator












