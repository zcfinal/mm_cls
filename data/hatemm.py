from torch.utils.data import Dataset,DataLoader
from PIL import Image
import torch
import jsonlines


class ITDataset(Dataset):
    def __init__(self,datas,data_dir):
        self.datas = datas
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self,idx):
        label=self.datas[idx]['label']
        image = Image.open(self.data_dir+self.datas[idx]['img'])
        text=self.datas[idx]['text']
        return image,text,label

class HateMMDataloaderSet:
    def __init__(self,args,model):
        self.args = args
        self.data_dir = '/data/zclfe/mm_cls/data/data/'
        self.model=model
        self.processor = model.processor
        traindata = self.init_dataset('train')
        devdata = self.init_dataset('dev')
        testdata = self.init_dataset('test')
        datas=[traindata,devdata,testdata]
        self.build_vocab(datas)

        traindataset = ITDataset(traindata,self.data_dir)
        devdataset = ITDataset(devdata,self.data_dir)
        testdataset = ITDataset(testdata,self.data_dir)

        self.train_dataloader = DataLoader(traindataset,batch_size=self.args.train_batch_size,collate_fn=self.collate_fn)
        self.dev_dataloader = DataLoader(devdataset,batch_size=self.args.eval_batch_size,collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(testdataset,batch_size=self.args.eval_batch_size,collate_fn=self.collate_fn)

    def collate_fn(self,batch):
        image,text,label = zip(*batch)
        data = self.processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
        data['label']=torch.LongTensor(label)
        return data

    def build_vocab(self,datas):
        if hasattr(self.processor,'need_build_vocab'):
            self.processor.build_vocab(datas)
            self.model.text_model.build_embedding(self.processor.tokenzier.embedding_matrix)

    def init_dataset(self,mode):
        data_path = self.data_dir+mode+'.jsonl'
        datas = []
        with open(data_path, "r+", encoding="utf8") as f:
            for item in jsonlines.Reader(f):
                datas.append(item)

        print(f'{mode} data length: {len(datas)}')
        return datas
