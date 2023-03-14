import torch.nn as nn
import torch
from .fusion import get_fusion
from .fusion.Attention import AttentionPooling
import numpy as np
from nltk.tokenize import word_tokenize
from transformers import CLIPImageProcessor

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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_head = conv_bn_relu_pool(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(conv_bn_relu_pool(128, 128), conv_bn_relu_pool(128, 128))
        self.layer2 = conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_head = conv_bn_relu_pool(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(conv_bn_relu_pool(512, 512), conv_bn_relu_pool(512, 512))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.trans = nn.Linear(512,128)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.trans(x)
        x = self.dropout(x)
        return x
    
class RNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding_dim = 200
        self.rnn = nn.GRU(self.embedding_dim,
                        hidden_size=200,
                        num_layers=1,
                        bidirectional=True,
                        batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.attention = AttentionPooling(400)
        self.fc = nn.Linear(400, 128)
    
    def build_embedding(self,emb):
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(emb),freeze=True)

    def forward(self, text):
        emb = self.embedding(text)
        emb = self.dropout(emb)
        emb = self.rnn(emb)[0]
        emb = self.dropout(emb)
        emb = self.attention(emb)
        emb = self.fc(emb)
        return emb

class Glove_Tokenizer:
    def __init__(self) -> None:
        self.embedding_path = '/data/mm_data/glove/glove.6B.200d.txt'
        self.max_length=77

    def build_vocab(self,datas):
        vocab = []
        for data in datas:
            for li in data:
                vocab.extend(word_tokenize(li['text'].lower()))
        vocab = list(set(vocab))
        vocab = {word:i for i,word in enumerate(vocab,1)}
        print(f'total token {len(vocab)}')
        return vocab

    def build_word_embedding(self,datas):
        vocab = self.build_vocab(datas)
        embedding_matrix = np.zeros((len(vocab)+1,200),dtype='float32')
        vis_word = []
        cand = []
        #padding 0
        with open(self.embedding_path,'r') as f:
            for i,line in enumerate(f):
                values = line.split()
                word = values[0]
                if word in vocab:
                    vis_word.append(word)
                    idx = vocab[word]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embedding_matrix[idx]=coefs
                    cand.append(coefs)

        cand=np.array(cand,dtype='float32')
        mu=np.mean(cand, axis=0)
        Sigma=np.cov(cand.T)
        norm=np.random.multivariate_normal(mu, Sigma, 1)
        for word,i in vocab.items():
            if word not in vis_word:
                embedding_matrix[i]=np.reshape(norm, 200)
        embedding_matrix[0]=np.zeros(200,dtype='float32')
        self.embedding_matrix=embedding_matrix
        self.vocab=vocab


    def __call__(self,sentences):
        token = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [self.vocab[word] for word in words[:self.max_length]]
            words = [0]*(self.max_length-len(words))+words
            token.append(words)
        token = torch.LongTensor(token)
        return token
        

class Preprocess:
    def __init__(self) -> None:
        self.tokenzier = Glove_Tokenizer()
        self.need_build_vocab = True
        self.image_process = CLIPImageProcessor()

    def build_vocab(self,datas):
        self.tokenzier.build_word_embedding(datas)


    def __call__(self, *args, **kwds):
        text = self.tokenzier(kwds['text'])
        image = self.image_process(kwds['images'])
        image = torch.FloatTensor(np.array(image['pixel_values']))
        rslt={'image':image,'text':text}
        return rslt
        
        

class SimpleModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.processor =Preprocess()
        self.text_model = RNN()
        self.visul_model = ResNet9()

        self.fusion = get_fusion(self.args.fusion)(args,128) 
        if self.args.dataset=='mmimdb':
            self.criterien = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.criterien = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, data):
        label=None
        if 'label' in data and data['label'] is not None:
            label=data['label']
            del data['label']
        
        image_embeds = self.visul_model(data['image'])
        text_embeds = self.text_model(data['text'])

        logits = self.fusion(image_embeds, text_embeds)
        loss=None
        if label is not None:
            loss = self.criterien(logits,label)
            loss = torch.mean(loss)
        rslt = {}
        rslt['logits']=logits
        rslt['loss']=loss
        return rslt
