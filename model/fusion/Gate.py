import torch.nn as nn
import torch

class GateCls(nn.Module):
    def __init__(self,args,dim):
        super().__init__()
        self.args = args
        self.dim = dim
        self.trans1 = nn.Linear(dim,dim)
        self.trans2 = nn.Linear(dim,dim)
        self.trans_all = nn.Linear(dim*2,dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(dim,self.args.num_classes)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_normal_(module.weight)
    
    def forward(self,image_emb,text_emb):
        whole = torch.cat([image_emb,text_emb],1)
        z = self.sigmoid(self.trans_all(whole))
        image_emb = self.tanh(self.trans1(image_emb))
        text_emb = self.tanh(self.trans2(text_emb))
        emb = z*image_emb + (1-z)*text_emb
        logits = self.fc(emb)
        return logits