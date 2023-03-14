from .MLP import MLP
from .Attention import AttentionCls

def get_fusion(fusion_type):
    return eval(fusion_type)