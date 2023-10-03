import torch
from torch import nn
from torch.utils.data import DataLoader
import cv2

from torchvision import transforms, models, datasets

import numpy as np
import matplotlib.pyplot as plt

import copy
import json
from PIL import Image

# Varieble for train status tracking
train = {'epoch': 0}

# Variables for directories paths
DATASET_DIR_PATH = 'images/tiny-imagenet'
TRAIN_IMG_DIR_PATH = 'images/tiny-imagenet/train'
VAL_IMG_DIR_PATH = 'images/tiny-imagenet/val'

# CPU or GPU select
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

if use_cuda:
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.benchmark = True


# Default image augmentation and nornalization parameters
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Default image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(30,),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Data loaders
def dataloader_create(data, transform):
    if data is None: 
        return None
    
    if transform is None:
        dataset = datasets.ImageFolder(data, transform=transforms.ToTensor())
    else:
        dataset = datasets.ImageFolder(data, transform=transform)

    if use_cuda:
        kwargs = {'pin_memory': True}
    else:
        kwargs = {}
    
    dataloader = DataLoader(dataset, batch_size=64, 
                        shuffle=(True), 
                        **kwargs)

    return dataloader

def init_dataloaders():
    dataloaders = {
        'train': dataloader_create(data=TRAIN_IMG_DIR_PATH, transform=data_transforms['train']),
        'val': dataloader_create(data=VAL_IMG_DIR_PATH, transform=data_transforms['val'])
    }

    # Index - Name dictionary
    with open('cat_to_name.json', 'r') as f:
        cat_name_dict = json.load(f)

    cat_label_dict = dataloaders['train'].dataset.class_to_idx

    label_class_dict = {}
    for cat in cat_label_dict:
        cla = cat_name_dict[cat]
        label = cat_label_dict[cat]
        label_class_dict[label] = cla

    return dataloaders, label_class_dict

# Core functions declare
def model_create(class_count=0):
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

    feature_in_count = model.fc.in_features
    feature_out_count = int(feature_in_count/4)
    
    custom_fc = nn.Sequential(
        nn.Linear(feature_in_count, feature_out_count),
        nn.ReLU(inplace=True),
        nn.Linear(int(feature_out_count), class_count)
        )
    
    model.fc = custom_fc

    return model

def model_change_class_count(model, class_count):
    feature_in_count = models.resnext50_32x4d().fc.in_features
    feature_out_count = int(feature_in_count/4)
    
    custom_fc = nn.Sequential(
        nn.Linear(feature_in_count, feature_out_count),
        nn.ReLU(inplace=True),
        nn.Linear(int(feature_out_count), class_count)
        )
    
    model.fc = custom_fc

    return model    

def model_train(model, dataloaders, criterion, optimizer, scheduler, epoch_count=15, train_state={}):
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = .0

    dataset_sizes = {
        'train': len(dataloaders['train'].dataset), 
        'val': len(dataloaders['val'].dataset)
    }

    for epoch in range(epoch_count):
        train_state['epoch'] = epoch + 1
 
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()                

            run_loss = .0
            run_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                        
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                run_loss += loss.item() * inputs.size(0)
                run_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                    
            loss = run_loss / dataset_sizes[phase]
            acc = run_corrects.double() / dataset_sizes[phase]
            
            if phase == 'val' and acc > best_acc:
                best_acc = acc
                best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    return model

def create_checkpoint(model, label_class_dict, criterion, optimizer, scheduler, path='./model.tar'):
    model.to('cpu')
    
    checkpoint = {
                  'model_state_dict': model.state_dict(),
                  'label_class_dict': label_class_dict,
                  'criterion': criterion.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()
                  }

    torch.save(checkpoint, path)

def load_checkpoint(path='./model.tar'):
    checkpoint = torch.load(path, map_location=device)
    model = model_create()
   
    label_class_dict = checkpoint['label_class_dict']
    model_change_class_count(model, len(label_class_dict))
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    
    return model, label_class_dict, criterion, optimizer, scheduler

def image_transforming(image):
    image = data_transforms['val'](image)
    return image

def image_denoising(path):
    image = cv2.imread(path)
    image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = transforms.ToPILImage()(image)
    image = image_transforming(image)
    return image

def image_process(path):
    image = Image.open(path)
    image = image_transforming(image)
    image_dn = image_denoising(path)

    return image, image_dn

def image_show(image, ax=None, title=None, titlecolor='k'):
    if ax is None:
        _, ax = plt.subplots()
    
    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.axis('off')
    ax.imshow(image)
    ax.grid(False)
    if title:
        ax.set_title(title, color=titlecolor)
    
    return ax

def predict(model, image, topk=5):    
    with torch.no_grad():
        model.eval()
        
        image = image.view(1,3,224,224)
        image = image.to(device)
        
        predictions = model(image)
        sm = torch.nn.Softmax(dim=1)
        probabilities = sm(predictions) 
        top_ps, top_class = probabilities.topk(topk, dim=1)
    
    return top_ps, top_class