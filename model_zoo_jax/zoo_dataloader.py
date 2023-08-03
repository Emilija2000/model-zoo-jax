from model_zoo_jax.logger import model_restore
import json
import jax.numpy as jnp
import jax
from jax import random
from jax.typing import ArrayLike
import numpy as np
from typing import Tuple, Union, Optional, List
import os
from math import ceil

unmap_info = {
    "dataset": {
        "CIFAR10": 0,
        "MNIST": 1,
    },
    "batch_size": {
        8: 0,
        16: 1,
        32: 2,
        64: 3,
        128: 4,
        256: 5,
    },
    "augment": {
        True: 0,
        False: 1,
    },
    "optimizer": {
        "adamW": 0,
        "sgd": 1,
        "scheduler": 2,
    },
    "activation": {
        "relu": 0,
        "leakyrelu": 1,
        "tanh": 2,
        "sigmoid": 3,
        "silu": 4,
        "gelu": 5
    },
    "init": {
        'U': 1,
        'N': 2,
        'TN': 3,
        None: 0
    },
    "model_name":{
        "smallCNN":0,
        "largeCNN":1,
        "lenet5":2,
        "alexnet":3,
        "resnet18":4
    }
}

def insert_all_targets(all_labels_dict: dict, config_dict:dict, metrics_dict:dict,
                       without:Optional[list] = ["data_mean","data_std","seed"]) -> dict:
    """
    Helper function: adds targets from all keys in config and metrics dictionaries,
    except for those specified by 'without'.
    """
    if len(all_labels_dict)==0:
        tasks = list(config_dict.keys()) + list(metrics_dict.keys())
        all_labels_dict = {key:[] for key in tasks if key not in without}
      
    for key in config_dict.keys():
        if key not in without:  
            if key in unmap_info.keys():
                data = unmap_info[key][config_dict[key]]
            else:
                data = config_dict[key]
            all_labels_dict[key].append(data)
    for key in metrics_dict.keys():
        all_labels_dict[key].append(metrics_dict[key])
    return all_labels_dict

def flatten_net(net):
    mylist = jax.tree_util.tree_flatten(net)[0]
    net = [item.flatten() for sublist in mylist for item in sublist]
    net = jnp.concatenate(net)
    return net

def load_nets(n:int=500, 
              data_dir:str='checkpoints/cifar10_lenet5_fixed_zoo', 
              flatten:bool=True,
              num_checkpoints:int =None,
              verbose:bool = False):
    """
    Load up to n networks from the model zoo, with all targets (hyperparameters 
    from config.json and metrics from specific training checkpoints). 
    
    Arguments:
        n (int): Number of checkpoints to load. If n==None, load all.
        data_dir (str): Path to model zoo created by running zoo.py 
        flatten (bool): If flatten=True return jnp.array of flattened network weights, otherwise return a list
                        of dicts (param trees)
        num_checkpoints (int): How many checkpoints from the same train run to load. If None read all. Nonvalid input 0 is converted to 1.
    """
    if num_checkpoints == 0:
        num_checkpoints = 1
    
    labels = {}
    
    nets = []
    
    current_config ={}
    
    for i, dir_info in enumerate(os.walk(data_dir)):
        if i == 0:
            continue
        subdir, dirs, files = dir_info
        dirs.sort(reverse=True) #the newest first
        
        # read config file
        for file_name in files:
            if file_name == "config.json":
                with open(os.path.join(subdir,file_name), 'r') as f:
                    current_config = json.load(f)
            
        # iterate through different epochs
        checkp = 0
        for dir_name in dirs:
            # load this model checkpoint
            dir_name = os.path.join(subdir,dir_name)
            net = model_restore(dir_name)
            
            # load train/test acc/loss
            with open(os.path.join(dir_name,"metrics.json"), 'r') as f:
                metrics = json.load(f)
             
            # append 
            nets.append(net)   
            labels = insert_all_targets(labels,current_config,metrics)
                
            checkp = checkp+1 
            if (num_checkpoints is not None) and (checkp >= num_checkpoints):
                break
            if n is not None and len(nets) == n:
                break
            
        if n is not None and len(nets) == n:
            break
    if verbose:
        print("Loaded", len(nets), "network parameters")

    if flatten:
        data_nets = [flatten_net(net) for net in nets] 
        data_nets = jnp.array(data_nets)
    else:
        data_nets = nets

    processed_labels = {}    
    for task in labels.keys():
        processed_labels[task] = jnp.array(labels[task])

    return data_nets, processed_labels

def load_multiple_datasets(dirs,num_networks=None, num_checkpoints=None,verbose=False,bs=None):
    """
    Load up to networks from multiple model zoos, with all targets (hyperparameters 
    from config.json and metrics from specific training checkpoints). The networks will 
    be loaded as params dicts
    
    Arguments:
        dirs (list): List of paths to different model zoo dirs to load
        num_networks (int): Number of checkpoints to load from each zoo. If n==None, load all.
        num_checkpoints (int): Number of checkpoints from a single training run, default: all
    """
    inputs_all = []
    all_labels_all = {}
    for dir in dirs:
        if verbose:
            print(f"Loading model zoo: {dir}")
        inputs, all_labels = load_nets(n=num_networks, 
                                   data_dir=dir,
                                   flatten=False,
                                   num_checkpoints=num_checkpoints)
        if bs!=None:
            num = ceil(len(inputs)/bs) * bs
            inputs = inputs[:num]
            all_labels = {key: all_labels[key][:num] for key in all_labels.keys()}
        inputs_all = inputs_all+inputs
        if len(all_labels_all.keys())==0:
            all_labels_all = all_labels
        else:
            all_labels_all = {key: jnp.concatenate([all_labels_all[key],all_labels[key]],axis=0) for key in all_labels.keys()}

    return inputs_all, all_labels_all

def shuffle_data(rng: jnp.ndarray, 
                 inputs: Union[jnp.ndarray,list], 
                 labels: Union[jnp.ndarray,dict],
                 chunks=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shuffle the data. Can handle flattened and unflattened network weights. 
    If type(labels)==dict, shuffle all target arrays
    If chunks!=None then be careful not to separate models from the same chunk -
    same training run
    As default behaviour, omits last part if len(inputs)%chunks!=0"""
    if chunks is not None:
        if len(inputs)%chunks!=0:
            a = len(inputs)%chunks
            inputs = inputs[:-a]
            if type(labels)==dict:
                labels = {key: value[:-a] for key,value in labels.items()}
            else:
                labels = labels[:-a]

        ''' index = jnp.arange(len(inputs)/chunks)
        index = random.permutation(rng,index)
        expanded_index = []
        for i in index:
            for j in range(chunks):
                expanded_index.append(i*chunks+j)
        index = jnp.array(expanded_index,dtype=jnp.int32)'''

        n_batches = len(inputs)//chunks
        index = jnp.arange(n_batches)
        index = random.permutation(rng, index)
        expanded_index = jnp.repeat(index, chunks) * chunks + jnp.tile(jnp.arange(chunks), n_batches)
        index = expanded_index.astype(jnp.int32)
        
    else:
        index = jnp.arange(len(inputs))
        index = random.permutation(rng,index)
    
    if type(inputs==list):
        inputs_new = [inputs[i] for i in index]
        inputs = inputs_new
    else:
        inputs = inputs[index]
        
    if type(labels)==dict:
        for key in labels.keys():
            labels[key] = labels[key][index]
    else:
        labels = labels[index]
    return inputs, labels

###################################
# Load train, val, test split with single task directly 

def split_data(data: list, labels: list, is_val:bool=True, chunks:int=1):
    if is_val:
        split_index = int(len(data)*0.7)
        split_index -= split_index % chunks
        split_index_1 = int(len(data)*0.85)
        split_index_1 -= split_index_1 % chunks
        
        if type(labels)==dict:
            labels_train = {}
            labels_val = {}
            labels_test = {}
            for key in labels.keys():
                labels_train[key] = labels[key][:split_index]
                labels_val[key] = labels[key][split_index:split_index_1]
                labels_test[key] = labels[key][split_index_1:]
        else:
            labels_train = labels[:split_index]
            labels_val = labels[split_index:split_index_1]
            labels_test = labels[split_index_1:]
        
        return (data[:split_index], labels_train, 
                data[split_index:split_index_1], labels_val,
                data[split_index_1:], labels_test)
    else:
        split_index = int(len(data)*0.8)
        split_index -= split_index % chunks
        
        if type(labels)==dict:
            labels_train = {}
            labels_test = {}
            for key in labels.keys():
                labels_train[key] = labels[key][:split_index]
                labels_test[key] = labels[key][split_index:]
        else:
            labels_train = labels[:split_index]
            labels_test = labels[split_index:]
        
        return (data[:split_index], labels_train, 
            data[split_index:], labels_test)

def flatten(x):
    return jax.flatten_util.ravel_pytree(x)[0]

def is_fine(params: dict):
    """Return false if std or mean is too high."""
    flat = flatten(params)
    if flat.std() > 5.0 or jnp.abs(flat.mean()) > 5.0:
        return False
    else:
        return True

def filter_data(data: List[dict], labels: List[ArrayLike]):
    """Given a list of net params, filter out those
    with very large means or stds."""
    assert len(data) == len(labels)
    f_data, f_labels = zip(*[(x, y) for x, y in zip(data, labels) if is_fine(x)])
    print(f"Filtered out {len(data) - len(f_data)} nets.\
          That's {100*(len(data) - len(f_data))/len(data):.2f}%.")
    return np.array(f_data), np.array(f_labels)

def load_data(rng,
              path: str,
              task: str, 
              n:int,
              num_checkpoints: int=1,
              is_flatten:bool=False,
              is_val:bool=True, 
              is_filter:bool=False, 
              verbose:bool=True):
    if verbose:
        print(f"Loading model zoo: {path}")
    inputs, all_labels = load_nets(n=n, 
                                   data_dir=path,
                                   flatten=is_flatten,
                                   num_checkpoints=num_checkpoints)
    if task is not None:
        if verbose:
            print(f"Training task: {task}.")
        labels = all_labels[task]
    else:
        labels = all_labels
    
    if is_filter:
        # Filter (high variance)
        inputs, labels = filter_data(inputs, labels)
    
     # Shuffle checkpoints before splitting
    rng, subkey = random.split(rng)
    inputs, labels = shuffle_data(subkey,inputs,labels,chunks=num_checkpoints)
    
    if is_val:
        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = split_data(inputs,labels,is_val=True,chunks=num_checkpoints)
    else:
        train_inputs, train_labels, val_inputs, val_labels = split_data(inputs, labels,is_val=False,chunks=num_checkpoints)
    
    if is_val:
        return train_inputs, train_labels, val_inputs, val_labels, test_inputs,test_labels
    return train_inputs, train_labels, val_inputs, val_labels

if __name__=="__main__":
    
    data,labels = load_nets(n=16,flatten=False)
    
    print(type(data))
    print(labels['class_dropped'])
    print(type(data[0]))
    print(len(data[0]))
    
    data,labels = shuffle_data(random.PRNGKey(42),data,labels,chunks=4)
    print(labels['class_dropped'])
    