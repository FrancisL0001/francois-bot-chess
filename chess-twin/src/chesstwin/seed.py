"""

Set's the global random seed for reproducibility. 
This is called at the start of the main function, and the seed value is read from the config file (configs/default.yaml) under project.seed.

"""

import random
import numpy as np
import torch

def set_seed(seed : int):
    """
    
    Argas : 
        seed (int): The random seed to set for reproducibility.
        
    rval: 
        None
        
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)