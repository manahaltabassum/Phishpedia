import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import numpy as np
import copy
import collections
import os
from torchvision import transforms

def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)

def fgsm(model, method, image, label, criterion, img_path, max_iter=100, epsilon=0.05, clip_min=-1.0, clip_max=1.0):
    '''
    https://pytorch.org/tutorials/beginner/fgsm_tutorial.html
    FGSM attack
    :param model: subject model
    :param method: fgsm|stepll
    :param image: input image
    :param label: original class
    :param criterion: loss function to use
    :param max_iter: maximum iteration allowed
    :param epsilon: perturbation strength
    :param clip_min:  minimum/maximum value a pixel can take
    :param clip_max:
    :return: perturbed images
    '''

    # initialize perturbed image
    pert_image = copy.deepcopy(image)
    x = Variable(pert_image, requires_grad=True)

    output = model(x)
    pred = output.max(1, keepdim=True)[1]
    iter_ct = 0
    

    # loop until attack is successful
    while pred == label:
        if method == 'fgsm':
            loss = criterion(output, label)  # loss for ground-truth class
        else:
            ll = output.min(1, keepdim=True)[1][0]
            loss = criterion(output, ll)  # Loss for least-likely class

        # Back propogation
        zero_gradients(x)
        model.zero_grad()
        loss.backward()

        # Collect the sign of the data gradient
        sign_data_grad = torch.sign(x.grad.data.detach())
        # test that if zero change to 1 or -1 (choose one)
        

        # Create the perturbed image by adjusting each pixel of the input image
        if method == 'fgsm':
            # print('Checking equality')
            # print(epsilon)
            # print(sign_data_grad)
            # print(x.data)
            # new = torch.add(x.data, (epsilon * sign_data_grad))
            # print(new)
            # print(torch.equal(x.data, new))
            x.data = x.data + epsilon * sign_data_grad
        else:
            x.data = x.data - epsilon * sign_data_grad

        # Adding clipping to maintain [0,1] range
        x.data = torch.clamp(x.data, clip_min, clip_max)

        # Pass perturbed image through original model and get new classification
        output = model(x)
        pred = output.max(1, keepdim=True)[1]

        os.makedirs('./fgsm_walkthrough/{}'.format(img_path[0]), exist_ok=True)
        MEAN = torch.tensor([0.5, 0.5, 0.5])
        STD = torch.tensor([0.5, 0.5, 0.5])
        reverse = transforms.ToPILImage()
        pert = copy.deepcopy(x.data)
        perturbed_data = Variable(pert_image, requires_grad=True)
        perturbed_data = perturbed_data.squeeze().detach().cpu() * STD[:, None, None] + MEAN[:, None, None]
        perturbed_data = reverse(perturbed_data)
        perturbed_data.save('./fgsm_walkthrough/{}/{}.png'.format(img_path[0], iter_ct))

        iter_ct += 1
        if iter_ct >= max_iter:
            break

    # Return perturbed image once iteration limit has been reached or once an
    # image has been successfully generated to defeat the model
    return x.data
