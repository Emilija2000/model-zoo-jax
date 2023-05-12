import jax.random
from jax import vmap
import numpy as np
import optax
import os

from model_zoo_jax.config import Parameters,sample_parameters
#from model_zoo_jax.datasets.dropclassdataset import load_dataset,drop_class_from_datasets, get_dataloaders
from model_zoo_jax.datasets.nontorch_dropclassdataset import load_dataset,drop_class_from_datasets, get_dataloaders
from model_zoo_jax.losses import CrossEntropyLoss
from model_zoo_jax.logger import Logger
from model_zoo_jax.models.models import get_model
from model_zoo_jax.train import Updater

import argparse
import time

#ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
#os.chdir(ROOT_PATH)

if __name__=='__main__':
    parser = argparse.ArgumentParser("Get random seed")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--log_wandb",type=bool,default=False)
    parser.add_argument("--augment",type=bool,default=False)
    parser.add_argument("--zoo_name",type=str, default="cifar10zoo")
    parser.add_argument("--dataset",type=str,default="CIFAR10")
    parser.add_argument("--num_models",type=int,default=1)
    parser.add_argument("--num_epochs",type=int,default=50)
    parser.add_argument("--model",type=str,default="lenet5")
    parser.add_argument("--activation",type=str,default=None)
    parser.add_argument("--init",type=str,default="random")
    parser.add_argument("--batch_size",type=int,default=None)
    parser.add_argument("--dropout",type=float, default=None)
    parser.add_argument("--weight_decay",type=float,default=None)
    parser.add_argument("--lr",type=float,default=None)
    parser.add_argument("--optimizer",type=str,default=None)
    parser.add_argument("--checkpoint_save_interval",type=int,default=20)
    args = parser.parse_args()
    
    key = jax.random.PRNGKey(args.seed)
    
    # load dataset
    datasets_full = load_dataset(args.dataset)
    
    for z in range(args.num_models):
        start = time.time()
        # config
        key, subkey = jax.random.split(key)
        subkey, zoo_config = sample_parameters(subkey,args.dataset,
                                            model_name=args.model, 
                                            activation=args.activation, 
                                            init=args.init, 
                                            batch_size=args.batch_size,
                                            dropout=args.dropout,
                                            weight_decay=args.weight_decay,
                                            lr=args.lr,
                                            opt=args.optimizer,
                                            num_epochs=args.num_epochs, 
                                            augment=args.augment)
                                
        # drop class
        datasets = drop_class_from_datasets(datasets_full, zoo_config.class_dropped)
        dataloaders = get_dataloaders(datasets, zoo_config.batch_size)
        
        loading_time=time.time()
        
        # model
        model,is_batch = get_model(zoo_config)
        if not(is_batch):
            batch_apply = vmap(model.apply, in_axes=(None,None,0,None),axis_name='batch')
            init_x = datasets['train'][0][0]
        else:
            batch_apply = model.apply
            init_x = next(iter(dataloaders['train']))[0]
    
        # loss
        evaluator = CrossEntropyLoss(batch_apply, zoo_config.num_classes)
        
        # optimizer
        if zoo_config.optimizer == 'adamW':
            optimizer = optax.adamw(learning_rate=zoo_config.lr, weight_decay=zoo_config.weight_decay)
        elif zoo_config.optimizer == 'sgd':
            optimizer = optimizer = optax.chain(optax.add_decayed_weights(zoo_config.weight_decay), optax.sgd(zoo_config.lr,momentum=0.99) 
        )
        else:
            raise ValueError('Unsupported optimizer')
    
        # logger
        checkpoints_subdir = "seed_"+str(args.seed)+"_iter_"+str(z)
        checkpoint_dir=os.path.join("checkpoints",args.zoo_name,checkpoints_subdir)
        logger = Logger(name=args.zoo_name+"_"+str(z), 
                        checkpoint_dir=checkpoint_dir, 
                        config=zoo_config,log_interval=500,
                        log_wandb=args.log_wandb,
                        save_interval=args.checkpoint_save_interval)
        logger.init()
        
        setup_time=time.time()

        # updater
        updater = Updater(opt=optimizer, evaluator=evaluator, model_init=model.init)
        state = updater.init_params(rng=zoo_config.seed,x=init_x)

        #training loop
        for epoch in range(zoo_config.num_epochs):
            train_all_acc = []
            train_all_loss = []
            for i,batch in enumerate(dataloaders['train']):
                state, train_metrics = updater.train_step(state, batch)
                logger.log(state, train_metrics)
                train_all_acc.append(train_metrics['train/acc'].item())
                train_all_loss.append(train_metrics['train/loss'].item())
            train_metrics = {'train/acc':np.mean(train_all_acc), 'train/loss':np.mean(train_all_loss)}
                
            test_acc = []
            test_loss = []
            for i,batch in enumerate(dataloaders['test']):
                state, test_metrics = updater.val_step(state, batch)
                test_acc.append(test_metrics['val/acc'].item())
                test_loss.append(test_metrics['val/loss'].item())
            test_metrics = {'test/acc':np.mean(test_acc), 'test/loss':np.mean(test_loss)}
            logger.log(state, train_metrics, test_metrics,last=(epoch==zoo_config.num_epochs-1))
        
        end = time.time()
        print("Trained model no. {} (time:{:.2f}, loading:{:.2f}, setup:{:.2f}):\t class missing: {}, final train_acc: {}, final test_acc: {}"
              .format(z,end-start,loading_time-start,setup_time-start,zoo_config.class_dropped,train_metrics['train/acc'],test_metrics['test/acc']))
        #print(zoo_config)
                  
        del updater, logger, evaluator