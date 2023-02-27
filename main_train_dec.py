import sys
import time
import platform

if str(platform.platform()).startswith('macOS'):
    sys.argv.extend(['--common.config-file', 'config/parcnet_dec.yaml'])
else:
    sys.argv.extend(['--common.config-file', 'config/parcnet_dec_colab.yaml'])

sys.path.append('parcnet')
from parcnet.data import *
from parcnet.main_train import *

# import torch
# from torch.nn import functional as F
# import torchvision.transforms as transforms
# from torch.cuda.amp import autocast
# import numpy as np
# import json

from PIL import Image, ImageDraw

if __name__ == "__main__":
    main_worker()
    
    