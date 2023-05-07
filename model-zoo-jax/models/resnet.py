import numpy as np
import jax.numpy as jnp
from jax import jit
import haiku as hk
import functools

@functools.partial(jit, static_argnums=(1,2))
def norm_batch(x, mean, std):
    if isinstance(mean, tuple):
        # Vectorized version for multi-channel images
        mean = jnp.array(mean)
        std = jnp.array(std)
        return (x - mean[None,None, None, :]) / std[None,None, None, :]
    else:
        # Scalar version for single-channel images
        return (x - mean) / std
    
@functools.partial(jit, static_argnums=(1,2))
def augment_batch(x,mean=0.5,std=0.5):
    #x = x/255.0
    return norm_batch(x,mean,std)

def forward_resnet18(x:jnp.array,is_training: bool=True, num_cls=10, 
                     data_mean=(0.49139968, 0.48215827,0.44653124), 
                     data_std=(0.24703233, 0.24348505, 0.26158768), 
                     init:hk.initializers.Initializer=hk.initializers.RandomNormal(stddev=0.01)):
    x = augment_batch(x, data_mean, data_std)
    x = jnp.resize(x, (x.shape[0],224,224,x.shape[3]))
    
    net= hk.nets.ResNet18(num_classes=num_cls,
                          resnet_v2=False,
                          initial_conv_config={'w_init':init,'b_init':init})
    
    return net(x, is_training=is_training)

if __name__ == "__main__":
    import jax
    import numpy as np
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader

    
    def custom_transform(x):
        return np.array(x, dtype=jnp.float32)

    def custom_target_transform(x):
        return np.array(x,dtype = jnp.int32)
    
    model = hk.transform_with_state(forward_resnet18)
    
    train_dataset = CIFAR10(root='datasets/cifar10/train_cifar10', train=True, download=True, transform=custom_transform, target_transform=custom_target_transform)
    
    def numpy_collate(batch):
        """From https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple,list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)
    dataloader = DataLoader(train_dataset,16,shuffle=False,collate_fn=numpy_collate)
    dummy_x = next(iter(dataloader))[0]
    
    rng_key = jax.random.PRNGKey(3)
    key,subkey = jax.random.split(rng_key)

    params,state = model.init(rng=subkey, x=dummy_x, is_training=True)

    #test 
    print("param count:", sum(x.size for x in jax.tree_util.tree_leaves(params)))
    #print("param tree:", jax.tree_map(lambda x: x.shape, params))
    #print("param values:", [x for x in jax.tree_util.tree_leaves(params)])
    out = model.apply(params,state,rng_key,dummy_x,is_training=True)[0]
    print("model out:",out)
    print(out.shape)