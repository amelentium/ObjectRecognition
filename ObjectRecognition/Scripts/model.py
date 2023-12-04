from re import VERBOSE
from tabnanny import verbose
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

import os
import copy
import time
import cv2
from PIL import Image

# CPU or GPU select
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

if use_cuda:
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.benchmark = True
  os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

def init_dataloaders(train_images_path, val_images_path):
    if train_images_path is None or val_images_path is None:
        return None

    dataloaders = {
        'train': dataloader_create(data=train_images_path, transform=data_transforms['train']),
        'val': dataloader_create(data=val_images_path, transform=data_transforms['val'])
    }

    return dataloaders

# Core functions declare
def model_create(dataloaders=None):
    model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

    feature_in_count = model.fc.in_features
    class_count = len(dataloaders['train'].dataset.classes) if dataloaders != None else 0

    model.fc = nn.Linear(feature_in_count, class_count)

    return model

def model_change_class_count(model, class_count):
    feature_in_count = model.fc.in_features

    model.fc = nn.Linear(feature_in_count, class_count)

    return model    

def model_train(model, dataloaders, criterion, optimizer, scheduler, epoch_count=18, verbose=False):
    if verbose:
        print(f'Training started, total number of epochs - {epoch_count} ...')
    
    best_wts = copy.deepcopy(model.state_dict())
    best_acc = .0

    dataset_sizes = {
        'train': len(dataloaders['train'].dataset), 
        'val': len(dataloaders['val'].dataset)
    }

    for epoch in range(epoch_count):
        start = time.time()
        
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
        
        end = time.time()

        if verbose:
        
            minutes = (end - start) // 60
            seconds = (end - start) % 60
            print('Epoch {0} fineshed in {1}{2}.'.format(epoch + 1, f'{minutes} min ' if minutes > 0 else '', f'{seconds:4.2f} sec'))

    model.load_state_dict(best_wts)
    
    if verbose:
        print(f'Training is complete with accuracy - {(best_acc * 100):5.2f}%')
        
    return model

def create_checkpoint(model, dataloaders, criterion, optimizer, scheduler, path='./model.tar'):
    model.to('cpu')

    data_classes =  dataloaders['train'].dataset.class_to_idx
    label_class_dict = {val: key for key, val in data_classes.items()}
    
    checkpoint = {
                  'model_state_dict': model.state_dict(),
                  'label_class_dict': label_class_dict,
                  }

    torch.save(checkpoint, path)
    model.to(device)

def load_checkpoint(path='./model.tar'):
    model = model_create()
    checkpoint = torch.load(path, map_location=device)

    model_change_class_count(model, len(checkpoint['label_class_dict']))

    model.load_state_dict(checkpoint['model_state_dict'])
    label_class_dict = checkpoint['label_class_dict']
    
    return model, label_class_dict

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
    image = Image.open(path).convert('RGB')
    image = image_transforming(image)
    image_dn = image_denoising(path)

    return image, image_dn


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

def make_prediction(model_path, image_path):
    model, label_class_dict = load_checkpoint(model_path)
    model = model.to(device)

    image, image_dn = image_process(image_path)

    topk = 5 if len(label_class_dict) > 5 else len(label_class_dict)

    probs, classes = predict(model, image, topk)
    probs = probs.data.cpu()
    probs = probs.numpy().squeeze()

    probs_dn, classes_dn = predict(model, image_dn, topk)
    probs_dn = probs_dn.data.cpu()
    probs_dn = probs_dn.numpy().squeeze()

    if (probs_dn[0] > probs[0]):
        probs = probs_dn
        classes = classes_dn

    classes = classes.data.cpu()
    classes = classes.numpy().squeeze()
    classes = [label_class_dict[_class_].title() for _class_ in classes]

    return classes, probs

def train(model_path, train_images_path, val_images_path):
    model = criterion = optimizer = scheduler = None
    dataloaders = init_dataloaders(train_images_path, val_images_path)
    
    if (os.path.exists(model_path)):
        model, _ = load_checkpoint(model_path)
    else:
        model = model_create(dataloaders)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    model = model_train(model, dataloaders, criterion, optimizer, scheduler, epoch_count=18, verbose=True)

    create_checkpoint(model, dataloaders, criterion, optimizer, scheduler, path=model_path)
