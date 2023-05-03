import jax.random
from jax import vmap
import numpy as np
import optax
import os

from config import Parameters,sample_parameters
from datasets.dropclassdataset import load_dataset,drop_class_from_datasets, get_dataloaders
from losses import CrossEntropyLoss
from logger import Logger
from models.models import get_model
from train import Updater

NUM_MODELS=1 #model zoo dataset size
SEED=42 #global seed

# fixed zoo parameters
NUM_EPOCHS = 50
OPTIMIZER = "adamW"
MODEL = "lenet5"
DATASET = "CIFAR10"
ZOO_NAME = "cifar10zoo"
LOG_WANDB = False
AUGMENT = False

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT_PATH)

if __name__=='__main__':
    
    key = jax.random.PRNGKey(SEED)
    
    # load dataset
    datasets_full = load_dataset(DATASET)
    
    for z in range(NUM_MODELS):
        # config
        key, subkey = jax.random.split(key)
        subkey, zoo_config = sample_parameters(subkey,DATASET,MODEL,OPTIMIZER,NUM_EPOCHS,AUGMENT)
        
        # drop class
        datasets = drop_class_from_datasets(datasets_full, zoo_config.class_dropped)
        dataloaders = get_dataloaders(datasets, zoo_config.batch_size)
        
        # model
        model = get_model(zoo_config)
        batch_apply = vmap(model.apply, in_axes=(None,None,0,None),axis_name='batch')
    
        # loss
        evaluator = CrossEntropyLoss(batch_apply, zoo_config.num_classes)
        
        # optimizer
        if zoo_config.optimizer == 'adamW':
            optimizer = optax.adamw(learning_rate=zoo_config.lr, weight_decay=zoo_config.weight_decay)
        elif zoo_config.optimizer == 'sgd':
            optimizer = optimizer = optax.chain(optax.add_decayed_weights(zoo_config.weight_decay), optax.sgd(zoo_config.lr) 
        )
        else:
            raise ValueError('Unsupported optimiser')
    
        # updater
        updater = Updater(opt=optimizer, evaluator=evaluator, model_init=model.init)
        state = updater.init_params(rng=zoo_config.seed,x=datasets['train'][0][0])
    
        # logger
        checkpoint_dir=os.path.join("checkpoints",ZOO_NAME,str(z))
        logger = Logger(name=ZOO_NAME+"_"+str(z), checkpoint_dir=checkpoint_dir, config=zoo_config,log_interval=500,log_wandb=LOG_WANDB)
        logger.init()

        #training loop
        for epoch in range(zoo_config.num_epochs):
            for batch in dataloaders['train']:
                state, train_metrics = updater.train_step(state, batch)
                logger.log(state, train_metrics)
                
            test_acc = []
            test_loss = []
            for batch in dataloaders['test']:
                state, test_metrics = updater.val_step(state, batch)
                test_acc.append(test_metrics['val/acc'].item())
                test_loss.append(test_metrics['val/loss'].item())
            test_metrics = {'test/acc':np.mean(test_acc), 'test/loss':np.mean(test_loss)}
            logger.log(state, train_metrics, test_metrics)
            
        print("Trained model no. {}: class missing: {}, final train_acc: {}, final test_acc: {}"
              .format(z,zoo_config.class_dropped,train_metrics['train/acc'],test_metrics['test/acc']))
                    