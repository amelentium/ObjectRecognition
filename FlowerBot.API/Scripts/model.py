import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
import cv2
import os
import time
import json
import copy
from PIL import Image

# CPU or GPU select
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Default image augmentation and nornalization parameters
mean = [0.484, 0.454, 0.401]
std = [0.225, 0.221, 0.221]

# Default image transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(30,),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

# Data load
data_dir = '../Images/flower_images'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'valid']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=os.cpu_count(), pin_memory=True) for x in ['train', 'valid']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

# Flower names load 
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

class_to_idx = image_datasets['train'].class_to_idx

cat_label_to_name = {}
for cat, label in class_to_idx.items():
    name = cat_to_name.get(cat)
    cat_label_to_name[label] = name

# Core functions declare
def model_create():
    model = models.resnext50_32x4d(pretrained=True)

    feature_in_count = model.fc.in_features
    feature_out_count = int(feature_in_count/4)
    class_count = 102

    custom_fc = nn.Sequential(
        nn.Linear(feature_in_count, feature_out_count),
        nn.ReLU(inplace=True),
        nn.Linear(feature_out_count, int(feature_out_count/2)),
        nn.ReLU(inplace=True),
        nn.Linear(int(feature_out_count/2), class_count)
        )

    model.fc = custom_fc
    return model

def model_train(model, criterion, optimizer, scheduler, epoch_count=15):
    start = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = .0

    for epoch in range(1, epoch_count+1):
        print(f'Epoch {epoch}/{epoch_count}')

        run_start = time.time()
 
        for phase in ['train', 'valid']:
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

            print(f'{phase} Loss: {loss:.4f} Acc: {acc:.4f}')
            
            if phase == 'valid' and acc > best_acc:
                best_acc = acc
                best_wts = copy.deepcopy(model.state_dict())
        
        spent_time = time.time() - run_start
        print(f'Epoch time: {(spent_time//60):.0f}m {(spent_time%60):.2f}s')
        print()

    spent_time = time.time() - start
    print(f'Training complete in {(spent_time//60):.0f}m {(spent_time%60):.2f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_wts)
    return model

def create_checkpoint(model, criterion, optimizer, scheduler, path='./model.tar'):
    model.to('cpu')
    
    checkpoint = {
                  'model_state_dict': model.state_dict(),
                  'criterion': criterion.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict()
                  }

    torch.save(checkpoint, path)

def load_checkpoint(path='./model.tar'):
    checkpoint = torch.load(path, map_location=device)
    model = model_create()
   
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion = checkpoint['criterion']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    
    return model, criterion, optimizer, scheduler

def image_transforming(image):
    image = TF.resize(image, 256)    
    upper_pixel = (image.height - 224) // 2
    left_pixel = (image.width - 224) // 2
    image = TF.crop(image, upper_pixel, left_pixel, 224, 224)    
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean, std)
    return image

def image_denoising(path):
    image = cv2.imread(path)
    image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = TF.to_tensor(image)
    image = TF.to_pil_image(image)
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