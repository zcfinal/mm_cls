import torch
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class Metric:
    def __init__(self,args):
        self.args=args
        self.all_metric = {'acc':self.acc,'auc':self.auc,'f1':self.f1}
        #main metric is the last one
        if self.args.dataset=='HateMM':
            self.metric=['acc','auc']
        if self.args.dataset=='mmimdb':
            self.metric=['f1']
        
    
    def acc(self,preds,labels):
        logits = torch.max(preds,1)[1]
        labels=labels.squeeze(-1)
        score=(logits==labels).float()
        acc = score.mean().item()
        return acc


    def auc(self,preds,labels):
        preds=preds.cpu().numpy()
        labels=labels.cpu().numpy()
        auc=roc_auc_score(labels,preds)
        return auc

    def f1(self,preds,labels):
        preds = torch.sigmoid(preds)
        preds = (preds>0.5).int()
        preds=preds.cpu().numpy()
        labels=labels.cpu().numpy()
        avgs = ('micro', 'macro', 'weighted', 'samples')
        rslt = {}
        for avg in avgs:
            rslt[avg] = precision_recall_fscore_support(labels,preds,average=avg)[:3]
        return rslt

    
    def __call__(self, preds, labels):
        result = {}
        main_rslt = None
        for name in self.metric:
            result[name]=self.all_metric[name](preds,labels)
        main_rslt = result[name]
        if isinstance(main_rslt,dict):
            main_rslt = main_rslt['micro'][2]
        return main_rslt,result