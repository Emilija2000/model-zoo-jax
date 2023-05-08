from model_zoo_jax.logger import model_restore
import json
import jax.numpy as jnp
import jax
import os

unmap_info = {
    "dataset": {
        "CIFAR10": 0,
    },
    "batch_size": {
        32: 0,
        64: 1,
        128: 2,
        256: 3,
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
    "initialization": {
        None: 0,
        "U": 1,
        "N": 2,
        "TN": 3,
    },
    "model_name":{
        "smallCNN":0,
        "largeCNN":1,
        "lenet5":2,
        "alexnet":3,
        "resnet18":4
    }
}

def insert_all_targets(all_labels_dict, config_dict, metrics_dict,
                       without = ["data_mean","data_std","seed"]):
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

def load_nets(n=500, 
              data_dir='checkpoints/cifar10_lenet5_fixed_zoo', 
              flatten=True, 
              verbose=True):
    
    labels = {}
    
    nets = []
    
    current_config ={}
    
    for i, dir_info in enumerate(os.walk(data_dir)):
        if i == 0:
            continue
        subdir, dirs, files = dir_info
        
        # read config file
        for file_name in files:
            if file_name == "config.json":
                with open(os.path.join(subdir,file_name), 'r') as f:
                    current_config = json.load(f)
            
        # iterate through different epochs
        for dir_name in dirs:
            # load this model checkpoint
            dir_name = os.path.join(subdir,dir_name)
            net = model_restore(dir_name)
            #if has_nans(net):
            #    if verbose:
            #        print("Not loading params at:", dir_name, "since it contains nan values")
            #    continue
            # load train/test acc/loss
            with open(os.path.join(dir_name,"metrics.json"), 'r') as f:
                metrics = json.load(f)
             
            # append 
            nets.append(net)   
            labels = insert_all_targets(labels,current_config,metrics)
                 
            if n is not None and len(nets) == n:
                break
        if n is not None and len(nets) == n:
            break
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

if __name__=="__main__":
    
    data,labels = load_nets(n=3,flatten=True)
    print(type(data))
    print(labels['class_dropped'])
    print(type(data[0]))
    print(len(data[0]))
    print(data[0])