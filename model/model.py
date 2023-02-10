import torch.nn as nn 
import torchvision 

backbone = torchvision.models.video.r3d_18(pretrained=True)

class Actionclassfier(nn.Module):
    
    def __init__(self, backbone = backbone , num_classes =13):
        super(Actionclassfier,self).__init__()
        self.backbone = backbone
        self.head = nn.Linear(400, num_classes)
    def forward(self,x ):
        batch_size =x.size(0)
        x= self.backbone(x)
        x= x.view(batch_size,-1)
        x= self.head(x)
        
        return x

