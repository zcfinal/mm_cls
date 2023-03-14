from .MLP import MLP
from .Attention import AttentionCls
from .Gate import GateCls

def get_fusion(fusion_type):
    return eval(fusion_type)