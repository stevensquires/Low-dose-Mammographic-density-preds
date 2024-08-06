from torchvision import models
import torch.nn as nn

def returnResNetModel(fineTuning):
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1))#,nn.Softmax(dim=1))   
#    model.fc = Identity()
#    model.eval()
    return model  

def returnModel(modelChoice,fineTuning):
    if modelChoice=='ResNet':
        model=returnResNetModel(fineTuning)
        return model
    







