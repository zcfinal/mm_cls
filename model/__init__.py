from clip import ClipModel

def get_model(model_type):
    return eval(model_type)