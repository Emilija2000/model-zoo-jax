from dataclasses import dataclass
from typing import List, Optional,Union, Sequence, Tuple

import haiku as hk

import jax
import jax.numpy as jnp
from jax import nn,jit

import functools

@dataclass
class ConvConfig:
    channels: int
    kernel:int
    stride: Optional[int] = 1
    padding: Union[str, Sequence[Tuple[int, int]]] = 'VALID'
    w_init: Optional[hk.initializers.Initializer] = None
    b_init: Optional[hk.initializers.Initializer] = None
    follow_by_pooling: Optional[bool] = False
    pooling_window: Optional[int] = 2
    pooling_stride: Optional[int] = 2
    
@dataclass
class LinConfig:
    size: int
    w_init: Optional[hk.initializers.Initializer] = None
    b_init: Optional[hk.initializers.Initializer] = None   
 
@dataclass   
class AdaptiveAvgPool2D(hk.Module):
    output_size:int 
    format: str = "NHWC" #that or NCHW
    
    def get_hw(self, x):
        input_shape = x.shape
        if self.format=="NCHW":
            if len(input_shape)==4:
                batch_size, channels, height, width = input_shape
            else:
                channels, height, width = input_shape
        else:
            if len(input_shape)==4:
                batch_size, height, width, channels = input_shape
            else:
                height, width, channels = input_shape
        return height,width
    
    #def pool_2(p, window_dimensions,window_strides,padding):
    #    return jax.lax.reduce_window(p, onp.zeros((), p.dtype), jax.lax.add,
    #                                window_dimensions=window_shape,
    #                                window_strides=window_shape,
    #                                padding=padding)
    
    def get_window_size_and_padding(self,x):
        input_shape = x.shape
        input_height, input_width = self.get_hw(x)
        output_height, output_width = self.output_size,self.output_size
        padding_height = max(0, (output_height - input_height % output_height))
        padding_width = max(0, (output_width - input_width % output_width))
        padded_height = input_height + padding_height
        padded_width = input_width + padding_width
        padding_height = padding_height//2
        padding_width = padding_width//2
        window_height = padded_height // output_height 
        window_width = padded_width // output_width
        
        if self.format=="NCHW":
            if len(input_shape)==4:
                return (1,1,window_height,window_width),((0, 0),(0, 0),(padding_height, padding_height), (padding_width, padding_width))
            else:
                return (1,window_height,window_width), ((0, 0),(padding_height, padding_height), (padding_width, padding_width))
        else:    
            if len(input_shape)==4:
                return (1,window_height,window_width,1),((0, 0),(padding_height, padding_height), (padding_width, padding_width), (0, 0))
                
            else:
                return (window_height,window_width,1),((padding_height, padding_height), (padding_width, padding_width), (0, 0))
    
    def __call__(self, x):
        # Get input shape and output shape
        window_shape, padding = self.get_window_size_and_padding(x)
        
        if window_shape[0]<1 or window_shape[1]<1 or window_shape[2]<1:
            return x
        
        # average pooling
        y = jax.lax.reduce_window(x, jnp.zeros((), x.dtype), 
                                  jax.lax.add, 
                                  window_dimensions=window_shape,
                                  window_strides=window_shape,
                                  padding=padding)
        y = y / jnp.prod(jnp.array(window_shape))
        
        return y
    
@dataclass
class CNN(hk.Module):
    """
    A CNN architecture with arbitrary number of CNN layers and linear layers.
    
    Attributes:
        output_size: Number of possible output classes
        nlin: Nonlinearity. One of: 'leakyrelu','relu','tanh','sigmoid','silu','gelu'
        dropout_rate: Dropout rate - dropout is used between every two layers of the same type (not when conv is followed by lin)
        conv_config: list of configurations of convolutional layers
        lin_config: list of configurations of linear layers
    """
    output_size: int
    nlin: str
    dropout_rate: float
    conv_config: List[ConvConfig]
    lin_config: List[LinConfig]
    dropout_for_conv: Optional[bool] = True
    fixed_emb_size: Optional[int] = None
    
    def __call__(self, x:jnp.array, is_training:bool) -> jnp.array:
        dropout_rate = self.dropout_rate if is_training else 0.
        
        # convolutional part
        first = True
        for config in self.conv_config:
            
            if self.dropout_for_conv and not(first) and dropout_rate > 0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
            first = False
            
            x = hk.Conv2D(output_channels = config.channels, 
                          kernel_shape = config.kernel, 
                          stride = config.stride, 
                          padding = config.padding,
                          w_init = config.w_init,
                          b_init = config.b_init,
                          data_format="NHWC")(x)
            
            x =self.__get_nonlin()(x)
            if config.follow_by_pooling:
                x = hk.MaxPool(window_shape=(config.pooling_window,config.pooling_window,1),
                               strides=(config.pooling_stride,config.pooling_stride,1),
                               padding="VALID")(x)
            
        # flatten
        if self.fixed_emb_size is not None:
            #x = AdaptiveAvgPool2D(self.fixed_emb_size)(x)
            x = hk.MaxPool(window_shape=(2,2,1),strides=1,padding="VALID")(x) #temporary
        
        x = jnp.ravel(x)
            
        # linear part
        for config in self.lin_config:
            x = hk.Linear(config.size, w_init=config.w_init, b_init=config.b_init)(x)
            x = self.__get_nonlin()(x)
            if dropout_rate>0:
                x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
    
        x = hk.Linear(output_size=self.output_size)(x)
        #x = jax.nn.log_softmax(x)
        return x
    
    def __get_nonlin(self):
        if self.nlin == 'leakyrelu':
            return nn.leaky_relu
        if self.nlin == 'relu':
            return nn.relu
        if self.nlin == 'tanh':
            return jnp.tanh
        if self.nlin == 'sigmoid':
            return nn.sigmoid
        if self.nlin == 'silu':
            return nn.silu
        if self.nlin == 'gelu':
            return nn.gelu
        
@functools.partial(jit, static_argnums=(1,2))
def norm(x, mean, std):
    if isinstance(mean, tuple):
        # Vectorized version for multi-channel images
        mean = jnp.array(mean)
        std = jnp.array(std)
        return (x - mean[None, None, :]) / std[None, None, :]
    else:
        # Scalar version for single-channel images
        return (x - mean) / std
    
@functools.partial(jit, static_argnums=(1,2))
def augment(x, mean, std):
    #x = x/255.0
    x= norm(x, mean,std)
    return x
        
def forward_cnn_small(x:jnp.array, is_training: bool=True, num_cls=10, dropout=0., activation="leakyrelu", data_mean=0.5, data_std=0.5, init:hk.initializers.Initializer=None):
    "Small CNN from Model Zoo paper: https://arxiv.org/pdf/2209.14764v1.pdf"
    
    x = augment(x, data_mean, data_std)
    return CNN(output_size=num_cls,
             nlin=activation,
             dropout_rate=dropout,
             conv_config=[
                ConvConfig(channels=8, kernel=5,follow_by_pooling=True, w_init=init, b_init=init),
                ConvConfig(channels=6, kernel=5,follow_by_pooling=True, w_init=init, b_init=init),
                ConvConfig(channels=4, kernel=2,follow_by_pooling=True, pooling_stride=1, w_init=init, b_init=init),
               ],
             lin_config=[LinConfig(20, w_init=init, b_init=init)]
            )(x, is_training=is_training)
    
def forward_cnn_large(x:jnp.array, is_training: bool=True, num_cls=10, dropout=0., activation="leakyrelu", data_mean=0.5, data_std=0.5, init:hk.initializers.Initializer=None):
    "Big CNN from Model Zoo paper: https://arxiv.org/pdf/2209.14764v1.pdf"
    x = augment(x, data_mean, data_std)
    return CNN(output_size=num_cls,
             nlin=activation,
             dropout_rate=dropout,
             conv_config=[
                ConvConfig(channels=16, kernel=3,follow_by_pooling=True, w_init=init, b_init=init),
                ConvConfig(channels=32, kernel=3,follow_by_pooling=True, w_init=init, b_init=init),
                ConvConfig(channels=15, kernel=3,follow_by_pooling=True, pooling_stride=1, w_init=init, b_init=init),
               ],
             lin_config=[LinConfig(20, w_init=init, b_init=init)]
            )(x, is_training=is_training)
    
def forward_lenet5(x:jnp.array, is_training: bool=True, num_cls=10, dropout=0., activation="leakyrelu", data_mean=0.5, data_std=0.5, init:hk.initializers.Initializer=None):
    x = augment(x, data_mean, data_std)
    return CNN(output_size=num_cls,
             nlin=activation,
             dropout_rate=dropout,
             conv_config=[
                ConvConfig(channels=6, kernel=5,follow_by_pooling=True, w_init=init, b_init=init),
                ConvConfig(channels=16, kernel=5,follow_by_pooling=True, w_init=init, b_init=init),
            ],
             lin_config=[LinConfig(120, w_init=init, b_init=init),
                         LinConfig(84, w_init=init, b_init=init)
                         ]
            )(x, is_training=is_training)

def forward_alexnet(x:jnp.array, is_training: bool=True, num_cls=10, dropout=0.5, activation="relu", data_mean=0.5, data_std=0.5, init:hk.initializers.Initializer=None):
    x = augment(x, data_mean, data_std)
    x = jnp.resize(x, (256,256,x.shape[2]))
    return CNN(output_size=num_cls,
             nlin=activation,
             dropout_rate=dropout,
             dropout_for_conv=False,
             conv_config=[
                ConvConfig(channels=64, kernel=11, stride=4, padding=(2,2),follow_by_pooling=True, pooling_window=3,pooling_stride=2, w_init=init, b_init=init),
                ConvConfig(channels=192, kernel=5, stride=1, padding=(2,2),follow_by_pooling=True, pooling_window=3,pooling_stride=2, w_init=init, b_init=init),
                ConvConfig(channels=384, kernel=3, stride=1, padding=(1,1),follow_by_pooling=False, w_init=init, b_init=init),
                ConvConfig(channels=256, kernel=3, stride=1, padding=(1,1),follow_by_pooling=False, w_init=init, b_init=init),
                ConvConfig(channels=256, kernel=3, stride=1, padding=(1,1),follow_by_pooling=True, pooling_window=3,pooling_stride=2, w_init=init, b_init=init),
            ],
             fixed_emb_size=6,
             lin_config=[LinConfig(4096, w_init=init, b_init=init),
                         LinConfig(4096, w_init=init, b_init=init)
                         ]
            )(x, is_training=is_training)

if __name__ == "__main__":
    import jax
    import numpy as np
    from torchvision.datasets import CIFAR10

    
    def custom_transform(x):
        return np.array(x, dtype=jnp.float32)

    def custom_target_transform(x):
        return np.array(x,dtype = jnp.int32)
    
    #model = hk.transform(forward_cnn_large)
    model = hk.transform(forward_lenet5)
    #model = hk.transform(forward_alexnet)
    
    
    train_dataset = CIFAR10(root='datasets/cifar10/train_cifar10', train=True, download=True, transform=custom_transform, target_transform=custom_target_transform)
    dummy_x = train_dataset[0][0] 
    
    rng_key = jax.random.PRNGKey(42)
    key,subkey = jax.random.split(rng_key)

    params = model.init(rng=subkey, x=dummy_x, is_training=True)

    #test 
    print("param count:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    print("param tree:", jax.tree_map(lambda x: x.shape, params))
    print("model out:",model.apply(params,rng_key,train_dataset[0][0]))