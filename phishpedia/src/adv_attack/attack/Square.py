from pkg_resources import add_activation_listener
import torch
import torch.nn as nn

import torchattacks
from torchattacks import Square
from torchattacks import FGSM
from torch.autograd import Variable
import copy

'''Used the following package to run the Square attack: https://github.com/Harry24k/adversarial-attacks-pytorch'''

def square(model, image, label):
  # print(image)
  pert_image = copy.deepcopy(image)
  x = Variable(pert_image, requires_grad=True)
  # atk = FGSM(model, eps=0.05)
  atk = Square(model, norm='Linf', eps=0.031, n_queries=5000, n_restarts=1, p_init=.8, loss='margin', resc_schedule=True, seed=0, verbose=True)
  adv_image = atk(x, label)
  return adv_image