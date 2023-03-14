import numpy as np
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

#---------data read---------------#
file_sim = '/data/zclfe/mm_cls/log/hatemm_sim_train.txt'
data = []
with open(file_sim,'r')as fin:
    for line in fin:
        ids,sim = line.split(',')
        data.append(float(sim))

#------------draw-------------$
sns.distplot(data,hist=False,kde=True,color='b',label='Hateful Meme')

file_sim = '/data/zclfe/mm_cls/log/mmimdb_sim_train.txt'
data = []
with open(file_sim,'r')as fin:
    for line in fin:
        ids,sim = line.split(',')
        data.append(float(sim))
sns.distplot(data,hist=False,kde=True,color='r',label='MM-Imdb')

plt.legend()
plt.xlabel('similarity')
fig_path = '/data/zclfe/mm_cls/log/hatemm_sim_dis.png'
plt.savefig(fig_path)


