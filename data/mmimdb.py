from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
import json
from collections import OrderedDict, Counter

class ITDataset(Dataset):
    def __init__(self,datas,data_dir,label2id):
        self.datas = datas
        self.data_dir = data_dir
        self.label2id = label2id
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self,idx):
        ids=self.datas[idx]['id']
        image = Image.open(self.data_dir+f'/dataset/{ids}.jpeg')
        text=self.datas[idx]['text']
        label_idx = [self.label2id[lab] for lab in self.datas[idx]['label'] if lab in self.label2id]
        label=torch.zeros(len(self.label2id)).float()
        if len(label_idx)>0:
            label[label_idx]=1
        
        return image,text,label

class MMImdbDataloaderSet:
    def __init__(self,args,model):
        self.args = args
        self.data_dir = '/data/zclfe/mm_cls/data/mmimdb'
        self.model=model
        self.processor = model.processor
        self.label2id={}
        self.label_list=[]
        traindata = self.init_dataset('train')
        devdata = self.init_dataset('dev')
        testdata = self.init_dataset('test')
        datas=[traindata,devdata,testdata]
        self.build_vocab(datas)
        self.count_label()

        traindataset = ITDataset(traindata,self.data_dir,self.label2id)
        devdataset = ITDataset(devdata,self.data_dir,self.label2id)
        testdataset = ITDataset(testdata,self.data_dir,self.label2id)

        self.train_dataloader = DataLoader(traindataset,batch_size=self.args.train_batch_size,collate_fn=self.collate_fn)
        self.dev_dataloader = DataLoader(devdataset,batch_size=self.args.eval_batch_size,collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(testdataset,batch_size=self.args.eval_batch_size,collate_fn=self.collate_fn)

    def count_label(self):
        counts = OrderedDict(
        Counter(self.label_list).most_common())
        target_names = list(counts.keys())[:23]
        for i,name in enumerate(target_names):
            self.label2id[name]=i

    def collate_fn(self,batch):
        image,text,label = zip(*batch)
        data = self.processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        data['label']=torch.stack(label,0)
        return data

    def build_vocab(self,datas):
        if hasattr(self.processor,'need_build_vocab'):
            self.processor.build_vocab(datas)
            self.model.text_model.build_embedding(self.processor.tokenzier.embedding_matrix)

    def init_dataset(self,mode):
        select_data = None
        if self.args.dataset_version is not None and mode=='train':
            with open(self.data_dir+'/'+self.args.dataset_version,'r')as f:
                select_data = json.load(f)
        data_path = self.data_dir+'/split.json'
        with open(data_path, "r+", encoding="utf8") as f:
            data = json.load(f)
        data = data[mode]

        datas=[]
        for ids in data:
            with open(self.data_dir+f'/dataset/{ids}.json','r')as f:
                metadata = json.load(f)
                text = metadata['plot'][-1]
                label = metadata['genres']
                self.label_list.extend(label)
                if mode=='train' and select_data is not None:
                    if ids not in select_data:
                        continue
                datas.append({'id':ids,'text':text,'label':label})
        
        print(f'{mode} data length: {len(datas)}')
        return datas
    