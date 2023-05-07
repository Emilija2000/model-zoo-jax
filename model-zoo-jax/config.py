import chex
from typing import Optional,Union,Tuple
import jax.random 
import jax.numpy as jnp

@chex.dataclass(frozen=True)
class Parameters:
    seed: jax.random.PRNGKey
    dataset: Optional[str] = "CIFAR100" #cifar10, cifar100 or mnist
    augment: Optional[bool] = False
    num_classes: Optional[int] = 99
    class_dropped: Optional[int] = 0
    model_name: Optional[str] = "resnet18" #smallCNN, largeCNN, lenet5, alexnet, resnet18
    activation: Optional[str] = "relu" #relu, leakyrelu,tanh,sigmoid,silu,gelu
    init: Optional[str] = "TN" #U - uniform, N - normal, TN -truncated normal, none will use default haiku values
    data_mean: Optional[Union[Tuple[jnp.float32],jnp.float32]] = (0.507, 0.4865, 0.4409)
    data_std: Optional[jnp.float32] = (0.2673, 0.2564, 0.2762)
    batch_size: Optional[int] = 128
    num_epochs: Optional[int] = 200
    optimizer: Optional[str] = "sgd_scheduler"
    dropout: Optional[jnp.float32] = 0.0
    weight_decay: Optional[jnp.float32] = 0.0
    lr: Optional[jnp.float32] = 0.1
    
def sample_parameters(rng_key, dataset_name, 
                      model_name=None, 
                      activation=None, 
                      init="random", 
                      batch_size=None,
                      dropout=None,
                      weight_decay=None,
                      lr=None,
                      opt=None,
                      num_epochs=None, 
                      augment=False,
                      class_dropped=None):
    new_key, seed, key_class_dropped, key_act, key_init, key_batch,key_dropout, key_weight_decay, key_lr,key_opt,key_model = jax.random.split(rng_key, num=11)
    
    # dataset specific one-class-omission
    if dataset_name == "MNIST":
        num_classes=9
        data_mean=0.5
        data_std=0.5
    elif dataset_name == "CIFAR10":
        num_classes=9
        data_mean = (0.49139968, 0.48215827,0.44653124)
        data_std = (0.24703233, 0.24348505, 0.26158768)
    elif dataset_name == "CIFAR100":
        num_classes=99
        data_mean = (0.507, 0.4865, 0.4409)
        data_std = (0.2673, 0.2564, 0.2762)
    else:
        raise ValueError("Unknown dataset name")
    
    if class_dropped==None:
        class_dropped = jax.random.randint(key_class_dropped, (), 0, num_classes+1)
        class_dropped = class_dropped.item()
    
    # activation
    if activation == None:
        activations = ["relu", "leakyrelu", "tanh", "sigmoid", "silu", "gelu"]
        activation = activations[jax.random.randint(key_act, (), 0, len(activations))]
    
    # init
    if init=="random":
        inits = [None, "U", "N", "TN"]
        init = inits[jax.random.randint(key_init, (), 0, len(inits))]
    elif init=="None":
        init=None
    
    # batch
    if batch_size==None:
        batch_sizes = [32, 64, 128]
        batch_size = batch_sizes[jax.random.randint(key_batch, (), 0, len(batch_sizes))]
    
    # dropout
    if dropout==None:
        dropout = jax.random.uniform(key_dropout, (), minval=0.0, maxval=0.5)
        dropout = dropout.item()
    
    # weight decay
    if weight_decay==None:
        log_weight_decay = jax.random.uniform(key_weight_decay, (), minval=-4.0, maxval=-2.0)
        weight_decay = jnp.power(10.0, log_weight_decay)
        weight_decay = weight_decay.item()
    
    # learning rate
    if lr==None:
        log_lr = jax.random.uniform(key_lr, (), minval=-4.0, maxval=-3.0)
        lr = jnp.power(10.0, log_lr)
        lr = lr.item()
    
    # optionally fixed parameters
    # optimizer
    if opt == None:
        optimizers = ["adamW","sgd"]
        opt = optimizers[jax.random.randint(key_opt, (), 0, len(optimizers))]
        
    if num_epochs == None:
        num_epochs = 50
        
    if model_name==None:
        models = ["smallCNN", "largeCNN", "lenet5","alexnet"]
        model_name = models[jax.random.randint(key_model,(),0,len(models))]
    
    return new_key,Parameters(seed=seed, 
                      dataset=dataset_name, 
                      augment=augment,
                      num_classes=num_classes, 
                      class_dropped=class_dropped, 
                      model_name=model_name,
                      activation=activation,
                      init=init, 
                      data_mean=data_mean,
                      data_std = data_std,
                      batch_size=batch_size,
                      num_epochs=num_epochs,
                      optimizer=opt,
                      dropout=dropout,
                      weight_decay=weight_decay,
                      lr=lr)
    