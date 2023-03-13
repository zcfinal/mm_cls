from PIL import Image
import torch
import requests
from transformers import CLIPProcessor, CLIPModel
import jsonlines
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader

#----------------------load model----------------------#
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = model.cuda()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#----------------------preprocess data----------------------#
validate_file = "/data/zclfe/mm_cls/data/data/train.jsonl"
data_dir = "/data/zclfe/mm_cls/data/data/"
datas = []
with open(validate_file, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        datas.append(item)

print(f'data length: {len(datas)}')

class Dataset(Dataset):
    def __init__(self,datas):
        self.datas = datas
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self,idx):
        ids = self.datas[idx]['id']
        image = Image.open(data_dir+self.datas[idx]['img'])
        text=self.datas[idx]['text']
        return ids,image,text

def collate_fn(batch):
    ids,image,text = zip(*batch)
    data = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True, max_length=77)
    return ids,data

dataset = Dataset(datas)
dataloader = DataLoader(dataset,batch_size=64,collate_fn=collate_fn)

#--------------------------compute similarity---------------------#

def to_cuda(data):
    for key in data:
        data[key] = data[key].cuda()
    return data

file_out = '/data/zclfe/mm_cls/log/hatemm_sim_train.txt'
f=open(file_out,'w')
for ids, data in tqdm(dataloader):
    data = to_cuda(data)
    outputs = model(**data)

    image_embeds = outputs['image_embeds']
    text_embeds = outputs['text_embeds']
    sim = torch.matmul(text_embeds, image_embeds.t())
    for i,id_name in enumerate(ids):
        f.write(f'{id_name},{sim[i][i].item()}\n')
        
f.close()
