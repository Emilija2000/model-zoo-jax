from model_zoo_jax.datasets.mnist import mnist_raw
from model_zoo_jax.datasets.cifar import cifar10_raw
from typing import Tuple
import jax.numpy as jnp
from jax import random
from jax.random import permutation

def load_dataset(name=str):
    if name == "CIFAR10":
        train_images, train_labels, test_images, test_labels = cifar10_raw()
        train_dataset = (jnp.array(train_images,dtype=jnp.float32),jnp.array(train_labels))
        test_dataset = (jnp.array(test_images,dtype=jnp.float32),jnp.array(test_labels))
    #elif name == "CIFAR100":
    #    train_dataset = CIFAR100(root='datasets/cifar100/train_cifar100', train=True, download=True, transform=custom_transform, target_transform=custom_target_transform)
    #    test_dataset = CIFAR100(root='datasets/cifar100/test_cifar100', train=False, download=True, transform=custom_transform,target_transform =custom_target_transform)
    elif name == "MNIST":
        train_images, train_labels, test_images, test_labels = mnist_raw()
        train_dataset = (jnp.array(train_images,dtype=jnp.float32),jnp.array(train_labels))
        test_dataset = (jnp.array(test_images,dtype=jnp.float32),jnp.array(test_labels))
    else:
        raise ValueError("Available datasets are CIFAR10, CIFAR100 and MNIST")
    
    datasets = {'train': train_dataset,
                'test':test_dataset}
    return datasets

def drop_class_from_dataset(d:Tuple[jnp.array], class_to_drop:int):
    data = d[0]
    targets = d[1]
    
    idx_to_keep = targets != class_to_drop
    #print(idx_to_keep.shape)
    #print(data.shape)
    data = data[idx_to_keep]
    targets = targets[idx_to_keep].tolist()
    targets = [target - int(target > class_to_drop) for target in targets]
    targets = jnp.array(targets)
    return (data,targets)
        

def drop_class_from_datasets(d: dict, class_to_drop:int):
    return {"train":drop_class_from_dataset(d['train'],class_to_drop),
            "test":drop_class_from_dataset(d['test'],class_to_drop)}

def load_drop_class_dataset(name:str, class_to_drop:int):
    datasets = load_dataset(name)
    return drop_class_from_datasets(datasets,class_to_drop)

def split_train_dataset(datasets, rng = 42, portion=0.1):
    pass

def dataloader(key, dataset, batch_size,shuffle=False,drop_last=True):
    data = dataset[0]
    labels = dataset[1]
    data_len = data.shape[0]
    i = 0
    key, subkey = random.split(key)
    order = jnp.arange(data_len)
    if shuffle:
        order = permutation(subkey, order)
    
    while i < data_len:
        if i+batch_size < data_len:
            yield data[order[i:i+batch_size]], labels[order[i:i+batch_size]]
        elif not(drop_last):
            yield data[order[i:]], labels[order[i:]]
        i += batch_size 
        
def get_dataloaders(datasets: dict, batch_size:int, rng:random.PRNGKey = random.PRNGKey(1),shuffle_train=False):
    dataloaders = {}
    for key, value in datasets.items():
        if key=='train':
            dataloaders[key] = list(dataloader(rng, value, batch_size, shuffle=shuffle_train, drop_last=True))
        else:
            dataloaders[key] = list(dataloader(rng, value, batch_size, shuffle=False, drop_last=True)) 
    return dataloaders

if __name__=="__main__":
    
    datasets_full = load_dataset('CIFAR10')
    datasets = drop_class_from_datasets(datasets_full, 0)
    
    flag = False
    for a in range(2):
        dataloaders = get_dataloaders(datasets, 32)
        for epoch in range(100):
            j=0
            for i,(img,label) in enumerate(dataloaders['train']):
                if not flag:
                    print(img.shape, label)
                    flag = True
                j = j +1
        print(a)