import torch 
from torch.autograd import Variable 
import torch.utils.data
import numpy as np 


# create a variable => borrowed from jgc123
def variable(obj, volatile=False):
    if isinstance(obj, (list, tuple)):
        return [variable(o, volatile=volatile) for o in obj]

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)

    obj = cuda(obj)
    obj = Variable(obj, volatile=volatile)
    return obj

# make cuda object 
def cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()
    return obj
