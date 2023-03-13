from .MLP import MLP

def get_fusion(fusion_type):
    return eval(fusion_type)