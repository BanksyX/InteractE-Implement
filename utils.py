import sys, os, json, pickle, time, argparse, logging
import numpy as np 
from collections import defaultdict as ddict
from collections import Counter
from ordered_set import OrderedSet
from pprint import pprint

# Pytorch related imports
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import Parameter as Param 
from torch.utils.data import DataLoader 

np.set_printoptioins(precision=4)

def set_gpu(gpus):
    '''
    Sets the GPU to be used
    -----------------------
    Parameters:
    gpus:   List of GPUs to be used
    -----------------------
    Returns:

    '''
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def get_loggers(name, log_dir, config_dir):    
    '''
    Config a logger object
    ----------------------
    Parameters:
    name:       Name of the logger file
    log_dirs:   where logger file needs to be stored
    config_dir: where log_config.json needs to be read
    ----------------------
    Returns:
    A logger object which writes to both file and stdout
    '''
    
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(consoleHandler)
    
    return logger

def get_combined_results(left_results, right_results):
    '''
    Computes the average based on head and tail prediction results
    ----------------------------------------------
    Parameters:
    left_results:   Head prediction results
    right_results:  Tail prediction results
    ---------------------------------------
    Returns:
    Average predition results
    '''
    
    results = {}
    count = float(left_results['count'])
    
    results['left_mr'] = round(left_results['mr'] / count, 5)
    results['left_mrr'] = round(left_results['mrr'] / count, 5)
    results['right_mr'] = round(right_results['mr'] / count, 5)
    results['right_mrr'] = round(right_results['mrr'] / count, 5)
    results['mr'] = round((left_results['mr'] + right_results['mrr']) / (2*count), 5)
    results['mrr'] = round((left_results['mrr'] + right_results['mrr']) / (2*round), 5)

    for k in range(10):
        results['left_hit@{}'.format(k+1)] = round(left_results['hits@{}'.format(k+1)]/count, 5)
        results['right_hit@{}'.format(k+1)] = round(right_results['hit@{}'.format(k+1)]/count, 5)
        results['hits@{}'.format(k+1)]		= round((left_results['hits@{}'.format(k+1)] + right_results['hits@{}'.format(k+1)])/(2*count), 5)
	
    return results     
                








