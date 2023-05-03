import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST, CIFAR10, CIFAR100

    
class DropClassDataset(Dataset):
    def __init__(self, dataset, drop_class):
        self.dataset = dataset
        self.drop_class = drop_class
        self.targets = np.array(self.dataset.targets if hasattr(self.dataset, 'targets') else self.dataset.labels)
    
        idx_to_keep = self.targets != self.drop_class
        self.data = self.dataset.data[idx_to_keep]
        self.targets = self.targets[idx_to_keep].tolist()
        self.targets = [target - int(target > self.drop_class) for target in self.targets]
        
        self.__name__ = f"{type(self).__name__}:{type(self.dataset).__name__}"

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if hasattr(self.dataset, 'transform') and not(self.dataset.transform is None):
            img = self.dataset.transform(img)
        if hasattr(self.dataset, 'target_transform') and not(self.dataset.target_transform is None):
            target = self.dataset.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.data)
     
    def __repr__(self):
        return f"{self.__name__},\nNumber of datapoints:{self.__len__()}\nDrop class:{self.drop_class}"
   
def custom_transform(x):
    return np.array(x, dtype=np.float32)

def custom_target_transform(x):
    return np.array(x,dtype = np.int32)

def numpy_collate(batch):
    """From https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html"""
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)
    
def load_dataset(name:str):
    if name == "CIFAR10":
        train_dataset = CIFAR10(root='datasets/cifar10/train_cifar10', train=True, download=True, transform=custom_transform, target_transform=custom_target_transform)
        test_dataset = CIFAR10(root='datasets/cifar10/test_cifar10', train=False, download=True, transform=custom_transform,target_transform =custom_target_transform)
    elif name == "CIFAR100":
        train_dataset = CIFAR100(root='datasets/cifar100/train_cifar100', train=True, download=True, transform=custom_transform, target_transform=custom_target_transform)
        test_dataset = CIFAR100(root='datasets/cifar100/test_cifar100', train=False, download=True, transform=custom_transform,target_transform =custom_target_transform)
    elif name == "MNIST":
        train_dataset = MNIST(root='datasets/mnist/train_mnist', train=True, download=True, transform=custom_transform, target_transform=custom_target_transform)
        test_dataset = MNIST(root='datasets/mnist/test_mnist', train=False, download=True, transform=custom_transform,target_transform =custom_target_transform)
    else:
        raise ValueError("Available datasets are CIFAR10, CIFAR100 and MNIST")
    
    datasets = {'train': train_dataset,
                'test':test_dataset}
    return datasets
    
def drop_class_from_datasets(d: dict, class_to_drop:int):
    return {'train': DropClassDataset(d['train'], drop_class=class_to_drop),
                'test': DropClassDataset(d['test'],drop_class=class_to_drop)}
    
def load_drop_class_dataset(name:str, class_to_drop:int):
    datasets = load_dataset(name)
    return drop_class_from_datasets(datasets,class_to_drop)

def split_train_dataset(datasets, rng = 42, portion=0.1):
    generator_val = torch.Generator().manual_seed(rng)
    train,val =  random_split(datasets['train'], [1-portion,portion], generator=generator_val)
    return {'train':train,'val':val, 'test':datasets['test']}
        
def get_dataloaders(datasets: dict, batch_size:int):
    dataloaders = {}
    for key, value in datasets.items():
        dataloaders[key] = DataLoader(value, batch_size, shuffle=False, collate_fn=numpy_collate,drop_last=True)
    return dataloaders
    
    
if __name__ =='__main__':
    BATCH_SIZE = 128
    
    datasets = load_drop_class_dataset("CIFAR10", 0)
    datasets = split_train_dataset(datasets)
    dataloaders = get_dataloaders(datasets)
    
    # test if all loaded
    print(len(datasets['train']))
    print(len(datasets['val']))
    print(len(datasets['test']))

    # test loading
    imgs, labels = next(iter(dataloaders['train']))
    print(imgs.shape, imgs[0].dtype, labels.shape, labels[0].dtype)

    plt.imshow(imgs[0]/255); plt.show()