import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self,args,dim):
        super().__init__()
        self.args = args
        self.dim = dim
        self.cls = nn.Sequential(
            nn.Linear(dim*2,dim),
            nn.Tanh(),
            nn.Linear(dim,dim//2),
            nn.Tanh(),
            nn.Linear(dim//2,self.args.num_classes),
        )
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self,image_emb,text_emb):
        emb = torch.cat([image_emb,text_emb],dim=1)
        logits = self.cls(emb)
        return logits

