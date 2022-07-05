import torch
import numpy as np
import random

SEED = 42
def seed_everything(SEED=SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True  # keep True if all the input have same size.


if __name__ == '__main__':
    # SEED = 42
    # seed_everything(SEED=SEED)
    seed_everything()