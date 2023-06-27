import haiku as hk
from haiku.initializers import *

import jax
import jax.numpy as jnp

from model_zoo_jax.models.cnn import *
from model_zoo_jax.models.resnet import *

import dm_pix as pix

def random_rot90(rng, image):
    """randomly rotate the image 50% of the time"""
    rot = jax.random.bernoulli(rng, 0.5)
    return jax.lax.cond(
        rot,
        lambda img: pix.rot90(img, 1),
        lambda img: img,
        image
    )

def augment_datapoint(rng, img):
    """Apply a random augmentation to a single image. Pixel values are assumed to be in [0, 1]"""
    rng = jax.random.split(rng, 7)
#    img = pix.random_brightness(rng[0], img, 0.3)
#    img = pix.random_contrast(rng[1], img, lower=0.2, upper=3)
#    img = pix.random_saturation(rng[2], img, lower=0, upper=3)
    img = pix.random_flip_left_right(rng[2], img)
    img = pix.random_flip_up_down(rng[3], img)
    img = random_rot90(rng[4], img)
    return img

def process_datapoint(rng: jnp.ndarray, 
                      img: jnp.array,
                      augment: bool = True) -> jnp.array:
    img = img / 255.0
    img = jax.lax.cond(  # Random augment?
            augment, 
            lambda img: augment_datapoint(rng, img),
            lambda img: img,
            img
        )
    return img

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
    if len(x.shape)==2:
        x=jnp.expand_dims(x,2)
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




