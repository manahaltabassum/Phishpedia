from pkg_resources import add_activation_listener
import torch
import torch.nn as nn

import torchattacks
from torchattacks import Pixle
from torch.autograd import Variable
import copy

def pixle(model, image, label):
  # print(image)
  pert_image = copy.deepcopy(image)
  x = Variable(pert_image, requires_grad=True)
  atk = Pixle(model)
  adv_image = atk(x, label)
  return adv_image