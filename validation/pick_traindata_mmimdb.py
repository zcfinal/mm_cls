import json
import random
#----------------count sim--------------#
sim={}
sim_file = '/data/zclfe/mm_cls/log/mmimdb_sim_train.txt'
with open(sim_file,'r')as f:
    for line in f:
        ids,simi = line.split(',')
        sim[ids]=float(simi)

#----------------count label samples------------#
label2sample={}
filters=['News','Talk-Show','Adult','Reality-TV']
data_dir='/data/zclfe/mm_cls/data/mmimdb/dataset'
for ids,simi in sim.items():
    with open(data_dir+f'/{ids}.json','r')as f:
            metadata = json.load(f)
            label = metadata['genres']
            for lab in label:
                if lab not in filters:
                    if lab not in label2sample:
                        label2sample[lab]=[]
                    label2sample[lab].append((ids,simi))

for label,sample in label2sample.items():
    label2sample[label] = sorted(sample,key=lambda x:x[1],reverse=True)

ratio=0.3
label_train_top = []
label_train_bottom = []
for label,sample in label2sample.items():
    sample_size = int(ratio*len(sample))
    label_train_top.extend([t[0] for t in sample[:sample_size]])
    label_train_bottom.extend([t[0] for t in sample[-sample_size:]])

label_train_top=list(set(label_train_top))
label_train_bottom=list(set(label_train_bottom))
random.shuffle(label_train_top)
random.shuffle(label_train_bottom)

output_top='/data/zclfe/mm_cls/data/mmimdb/top30.json'
output_bottom='/data/zclfe/mm_cls/data/mmimdb/bottom30.json'

print(len(label_train_top))
print(len(label_train_bottom))
with open(output_top,'w')as f:
    json.dump(label_train_top,f)

with open(output_bottom,'w')as f:
    json.dump(label_train_bottom,f)




