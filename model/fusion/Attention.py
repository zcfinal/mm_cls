import torch.nn as nn
import torch

class AttentionCls(nn.Module):
    def __init__(self,args,dim):
        super().__init__()
        self.args = args
        self.dim = dim
        self.pooling = AttentionPooling(dim)
        self.fc = nn.Linear(dim,self.args.num_classes)
    
    def forward(self,image_emb,text_emb):
        emb = torch.stack([image_emb,text_emb],dim=1)
        emb = self.pooling(emb)
        logits = self.fc(emb)
        return logits


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        self.hidden_size=hidden_size
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.att_fc2 = nn.Linear(self.hidden_size//2, 1)
        self.apply(self.init_weights)
        
    def init_weights(self, module):
        if isinstance(module, (nn.Linear)):
            nn.init.xavier_normal_(module.weight)
            
                
    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  
        return x