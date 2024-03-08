# Imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import utilities
import os
import time

# import data visualization
import matplotlib.pyplot as plt

# import pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD,Adam,lr_scheduler

import torchvision
from torchvision import transforms, datasets,models 
from torch.utils.data import DataLoader

# define transformations for train
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=.40),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# define transformations for test
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_datasets(inPath, imageSize):
    """
    Function to prepare the Datasets.
    :param pretrained: Boolean, True or False.
    Returns the training and validation datasets along 
    with the class names.
    """
    

    train_dataset = datasets.ImageFolder(
        root=os.path.join(inPath,"train"),
        transform=train_transform
    )
# validation dataset
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(inpath,"val"),
        transform=test_transform
    )
    

    return train_dataset, valid_dataset, train_dataset.classes

def get_training_dataloader(train_dataset, batch_size=128, num_workers=8, shuffle=True):
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    
    return train_loader 
    

    

# define test dataloader
def get_testing_dataloader(valid_dataset, batch_size=128, num_workers=8, shuffle=True):
   valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
   
   return valid_loader



def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    
    """
    path = os.path.join(outPath,"model_"+str(epochs)+".pth")
    print("model safed to: " + path) 
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, path)


if __name__ == '__main__':


    

    inpath = r"/mnt/projects/neuropath_hd/data/splits/non_glial/256_10x/kryo"
    outPath = r"/mnt/projects/neuropath_hd/data/modelCollection/kryo/denseNet/non-glial/256_10x"
    # number of epochs
    epochs = 100
    # learning rate
    learning_rate = 0.0001
    # device to use
    # don't forget to turn on GPU on kernel's settings
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    ImageSize = 256

  
    
    trainSet, valSet, classes = get_datasets(inpath,ImageSize)


    trainloader = get_training_dataloader(trainSet)
    testloader = get_testing_dataloader(valSet)

    classes_dict = {0 : 'Lym', 1 : 'MB', 2: 'MEL', 3 : 'MEN', 4 : 'MET', 5: 'PIT', 6:'SCHW' }



    model = models.densenet121(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.require_grad = False
    fc = nn.Sequential(
        nn.Linear(1024, 460),
        nn.ReLU(),
        nn.Dropout(0.4),
    
        nn.Linear(460,7),
        nn.LogSoftmax(dim=1)
    
    )
    model.classifier = fc

    criterion = nn.CrossEntropyLoss()

    # set optimizer, only train the classifier parameters, feature parameters are frozen
    optimizer = Adam(model.classifier.parameters(), lr=learning_rate)



    train_stats = pd.DataFrame(columns = ['Epoch', 'Time per epoch', 'Avg time per step', 'Train loss', 'Train accuracy', 'Train top-3 accuracy','Test loss', 'Test accuracy', 'Test top-3 accuracy']) 

    #train the model
    model.to(device)

    steps = 0
    running_loss = 0
    for epoch in range(epochs):
    
        since = time.time()
    
        train_accuracy = 0
        top3_train_accuracy = 0 
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            # calculate train top-1 accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
            # Calculate train top-3 accuracy
            np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
            target_numpy = labels.cpu().numpy()
            top3_train_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])
        
        time_elapsed = time.time() - since
    
        test_loss = 0
        test_accuracy = 0
        top3_test_accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate test top-1 accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                # Calculate test top-3 accuracy
                np_top3_class = ps.topk(3, dim=1)[1].cpu().numpy()
                target_numpy = labels.cpu().numpy()
                top3_test_accuracy += np.mean([1 if target_numpy[i] in np_top3_class[i] else 0 for i in range(0, len(target_numpy))])

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Time per epoch: {time_elapsed:.4f}.. "
              f"Average time per step: {time_elapsed/len(trainloader):.4f}.. "
              f"Train loss: {running_loss/len(trainloader):.4f}.. "
              f"Train accuracy: {train_accuracy/len(trainloader):.4f}.. "
              f"Top-3 train accuracy: {top3_train_accuracy/len(trainloader):.4f}.. "
              f"Test loss: {test_loss/len(testloader):.4f}.. "
              f"Test accuracy: {test_accuracy/len(testloader):.4f}.. "
              f"Top-3 test accuracy: {top3_test_accuracy/len(testloader):.4f}")

        #train_stats = train_stats.append({'Epoch': epoch, 'Time per epoch':time_elapsed, 'Avg time per step': time_elapsed/len(trainloader), 'Train loss' : running_loss/len(trainloader), 'Train accuracy': train_accuracy/len(trainloader), 'Train top-3 accuracy':top3_train_accuracy/len(trainloader),'Test loss' : test_loss/len(testloader), 'Test accuracy': test_accuracy/len(testloader), 'Test top-3 accuracy':top3_test_accuracy/len(testloader)}, ignore_index=True)
        save_model(epoch, model, optimizer, criterion)
        running_loss = 0
        model.train()
    

    #train_stats.to_csv(os.path.join(outPath,"stats.csv"));







    
