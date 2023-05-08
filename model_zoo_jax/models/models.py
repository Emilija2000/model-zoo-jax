import haiku as hk
from haiku.initializers import *

from meta_transformer.utils import process_datapoint

from models.cnn import *
from models.resnet import *

def get_initializer(name):
    if name is None:
        return None
    if name=="N":
        return RandomNormal()
    elif name=="TN":
        return TruncatedNormal(stddev=0.5)
    elif name=="U":
        return RandomUniform()
    else:
        raise ValueError("unknown initialization")
    
def get_model(config):
    "second output is signal if model.apply is already batched"
    fcn = lambda x,is_training: forward_with_augment(x, is_training, config)
    if config.model_name in ["smallCNN","largeCNN","lenet5","alexnet"]:
        return hk.transform(fcn), False
    elif config.model_name in ["resnet18"]:
        return hk.transform_with_state(fcn), True
    else:
        raise ValueError("Unknown model name")
 
def forward_with_augment(x, is_training, config):
    x = process_datapoint(hk.next_rng_key(),x,augment=config.augment if is_training else False)
    y = get_forward(config)(x, is_training)
    return y
     

def get_forward(config):
    if config.model_name=="smallCNN":
        return lambda x, is_training: forward_cnn_small(x, is_training, 
                                                                      num_cls= config.num_classes,
                                                                      dropout = config.dropout,
                                                                      activation=config.activation,
                                                                      data_mean=config.data_mean,
                                                                      data_std=config.data_std,
                                                                      init = get_initializer(config.init))
    elif config.model_name=="largeCNN":
        return lambda x, is_training: forward_cnn_large(x, is_training, 
                                                                      num_cls= config.num_classes,
                                                                      dropout = config.dropout,
                                                                      activation=config.activation,
                                                                      data_mean=config.data_mean,
                                                                      data_std=config.data_std,
                                                                      init = get_initializer(config.init))
    elif config.model_name=="lenet5":
        return lambda x, is_training: forward_lenet5(x, is_training, 
                                                                      num_cls= config.num_classes,
                                                                      dropout = config.dropout,
                                                                      activation=config.activation,
                                                                      data_mean=config.data_mean,
                                                                      data_std=config.data_std,
                                                                      init = get_initializer(config.init))
    elif config.model_name=="alexnet":
        return lambda x, is_training: forward_alexnet(x, is_training, 
                                                                      num_cls= config.num_classes,
                                                                      dropout = config.dropout,
                                                                      activation=config.activation,
                                                                      data_mean=config.data_mean,
                                                                      data_std=config.data_std,
                                                                      init = get_initializer(config.init))
    elif config.model_name=="resnet18":
        return lambda x, is_training: forward_resnet18(x, is_training, 
                                                                      num_cls= config.num_classes,
                                                                      data_mean=config.data_mean,
                                                                      data_std=config.data_std,
                                                                      init = get_initializer(config.init))
    else:
        raise ValueError('Available models are: smallCNN, largeCNN, alexnet, lenet5 and resnet18')




