from .Clip import ClipModel
from .simple_model import SimpleModel

def get_model(model_type):
    return eval(model_type)
