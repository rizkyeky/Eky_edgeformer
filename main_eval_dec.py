import sys
import os
import platform

if str(platform.platform()).startswith('macOS'):
    sys.argv.extend(['--common.config-file', 'config/parcnet_dec_eval.yaml'])
    sys.argv.extend(['--evaluation.detection.mode', 'test_set'])
    
else:
    sys.argv.extend(['--common.config-file', 'config/parcnet_dec_colab_eval.yaml'])
    sys.argv.extend(['--evaluation.detection.mode', 'test_set'])

sys.path.append('parcnet')
from parcnet.data import *
from parcnet.eval_det import *

# import torch
# from torch.nn import functional as F
# import torchvision.transforms as transforms
# from torch.cuda.amp import autocast
# import numpy as np
# import json

from PIL import Image, ImageDraw

if __name__ == "__main__":
    if not str(platform.platform()).startswith('macOS'):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main_worker_detection()
    
    