from .FGSM import fgsm
from .JSMA import jsma
from .DeepFool import deepfool
from .CWL2 import cw
from .Square import square
from .Pixle import pixle
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torchvision import transforms
from PIL import Image, ImageOps


# define the adversarial attack as a class
class adversarial_attack():
    '''
    Perform adversarial attack
    '''
    def __init__(self, method, model, dataloader, device, num_classes=10, save_data=True):
        '''
        :param method: Which attack method to use
        :param model: subject model to attack (Siamese model?)
        :param dataloader: dataloader
        :param device: cuda/cpu
        :param num_classes: number of classes for classification model
        :param save_data: save data or not
        '''
        self.method = method
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.num_classes = num_classes
        self.save_data = save_data
        
    def batch_attack(self, model_type):
        '''
        Run attack on a batch of data
        '''
        print('Testing the following attack: ' + self.method)

        # Accuracy counter
        correct = 0
        total = 0
        adv_examples = []
        ct_save = 0
        # adv_cat = torch.tensor([])
        counter = 0

        # Loop over all examples in test set
        for ct, (data, label, img_path) in tqdm(enumerate(self.dataloader)):
            # if counter > 0:
            #     break
            data = data.to(self.device, dtype=torch.float) 
            label = label.to(self.device, dtype=torch.long)
            
            # Forward pass the data through the model
            # Determine the predicted brand by the Siamese model
            output = self.model(data)
            self.model.zero_grad()
            init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

            if init_pred.item() != label.item():  # initially was incorrect --> no need to generate adversary
                # print('\nModel Initially Incorrect')
                total += 1
                # print(ct)
                continue

            # Call Attack
            # Generate the perturbed data 
            if self.method in ['fgsm', 'stepll']:
                criterion = nn.CrossEntropyLoss()
                perturbed_data = fgsm(self.model, self.method, data, label, criterion, img_path)
                
            elif self.method == 'jsma':
                # randomly select a target class
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                # print(target_class)
                perturbed_data = jsma(self.model, self.num_classes, data, target_class)
                
            elif self.method == 'deepfool':
                f_image = output.detach().cpu().numpy().flatten()
                I = (np.array(f_image)).flatten().argsort()[::-1]
                perturbed_data = deepfool(self.model, self.num_classes, data, label, I)
                
            elif self.method == 'cw':
                target_class = init_pred
                while target_class == init_pred:
                    target_class = torch.randint(0, self.num_classes, (1,)).to(self.device)
                # print(target_class)
                perturbed_data = cw(self.model, self.device, data, label, target_class)
            
            elif self.method == 'square':
                perturbed_data = square(self.model, data, label)

            elif self.method == 'pixle':
                perturbed_data = pixle(self.model, data, label)

            else:
                print('Attack method is not supported, please choose your attack from [fgsm|stepll|jsma|deepfool|cw]')
                
                
            # Re-classify the perturbed image
            self.model.zero_grad()
            self.model.eval()
            with torch.no_grad():
                output = self.model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1]
            # print('Checking if tensors are equal')
            # print(torch.equal(data, perturbed_data))
            if final_pred.item() == init_pred.item():
                # print('\nAttack unsuccessful')
                correct += 1  # still correct
            else: # save successful attack
                # print('Attack successful')
                # print(final_pred)
                # print(init_pred)
                if self.save_data:
                    # print('Saving an attack')
                    # os.makedirs('./adv_attacks/normal_relu_{}'.format(self.method), exist_ok=True)
                    os.makedirs('./successful_attacks/{}_{}'.format(model_type, self.method), exist_ok=True)
                    MEAN = torch.tensor([0.5, 0.5, 0.5])
                    STD = torch.tensor([0.5, 0.5, 0.5])
                    reverse = transforms.ToPILImage()
                    # Save the original instance
                    # x = data.squeeze().detach().cpu() * STD[:, None, None] + MEAN[:, None, None]
                    # x = reverse(x)
                    # x.save('./adv_attacks/normal_steprelu_{}/{}.png'.format(self.method, ct_save))
                    # Save the adversarial example
                    x = perturbed_data.squeeze().detach().cpu() * STD[:, None, None] + MEAN[:, None, None]
                    x = reverse(x)
                    x.save('./successful_attacks/{}_{}/{}'.format(model_type, self.method, img_path[0]))
                    ct_save += 1

            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
            total += 1
            # print(ct)
            # print("Test Accuracy = {}".format(correct/float(total)))

            counter += 1

        # Calculate final accuracy
        # final_acc = correct / float(len(self.dataloader))
        final_acc = correct / float(total)
        print("Test Accuracy = {} / {} = {}".format(correct, total, final_acc))

        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
            
