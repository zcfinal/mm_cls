import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.cls = nn.Sequential(
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,self.args.num_classes),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self,image_emb,text_emb):
        emb = torch.cat([image_emb,text_emb],dim=1)
        logits = self.cls(emb)
        return logits

