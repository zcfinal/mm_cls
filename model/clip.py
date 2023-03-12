from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn

class ClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def forward(self,data):
        return self.model(**data)