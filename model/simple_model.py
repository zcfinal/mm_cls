from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch
from .fusion import get_fusion

def conv_bn_relu_pool(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        self.prep = conv_bn_relu_pool(3, 64)
        self.layer1_head = conv_bn_relu_pool(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(conv_bn_relu_pool(128, 128), conv_bn_relu_pool(128, 128))
        self.layer2 = conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_head = conv_bn_relu_pool(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(conv_bn_relu_pool(512, 512), conv_bn_relu_pool(512, 512))
        self.MaxPool2d = nn.Sequential(
            nn.MaxPool2d(4))
    
    def forward(self,x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.MaxPool2d(x)
        x = x.view(x.size(0), -1)

        return x
    
class GRU_Text(nn.Module):
    def __init__(self):
        super().__init__()
        

class SimpleModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.text_model = 
        self.visul_model = ResNet9()

        for name,parameters in self.model.named_parameters():
            parameters.requires_grad = False

        self.fusion = get_fusion(self.args.fusion)(args)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if self.args.dataset=='mmimdb':
            self.criterien = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterien = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, data):
        label=None
        if 'label' in data and data['label'] is not None:
            label=data['label']
            del data['label']
        outputs = self.model(**data)
        image_embeds = outputs['image_embeds']
        text_embeds = outputs['text_embeds']

        logits = self.fusion(image_embeds, text_embeds)
        loss=None
        if label is not None:
            loss = self.criterien(logits,label)
            loss = torch.mean(loss)
        rslt = {}
        rslt['logits']=logits
        rslt['loss']=loss
        return rslt
