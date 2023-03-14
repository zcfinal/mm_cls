from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import torch
from .fusion import get_fusion

class ClipModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        for name,parameters in self.model.named_parameters():
            parameters.requires_grad = False

        self.fusion = get_fusion(self.args.fusion)(args,512)
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
