from .hatemm import HateMMDataloaderSet
from .mmimdb import MMImdbDataloaderSet

def get_dataloader(dataset_type):
    if dataset_type=='HateMM':
        return HateMMDataloaderSet
    if dataset_type=='mmimdb':
        return MMImdbDataloaderSet
    return eval(dataset_type)


