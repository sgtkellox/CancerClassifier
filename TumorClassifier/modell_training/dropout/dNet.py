import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch.optim as optim

import time

from model import CNN_dropout
from utils import save_model
from utils import save_plots

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# the validation transforms
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = datasets.ImageFolder(root = r'C:\Users\felix\Desktop\neuro\kryo\val',transform = train_transform)

# Load validation dataset
test_dataset = datasets.ImageFolder(root = r'C:\Users\felix\Desktop\neuro\kryo\test',transform = valid_transform)

batch_size = 4
# Create a data loader for loading training dataset
train_load = torch.utils.data.DataLoader(dataset = train_dataset, 
                                         batch_size = batch_size,
                                         shuffle = True,
                                         num_workers=2
                                        )

# Create a data loader for loading testing dataset
test_load = torch.utils.data.DataLoader(dataset = test_dataset, 
                                         batch_size = batch_size,
                                         shuffle = False,
                                       num_workers=2
                                       )

def train_model(lr,model, weight_decay=0, num_epochs = 200,):
    CUDA = torch.cuda.is_available()
    if CUDA:
        model = model.to('cuda')
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss function
    criterion = nn.CrossEntropyLoss()

    #Define the lists to store the results of loss and accuracy
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    
    # Training the CNN
    for epoch in range(1,num_epochs+1): 
        #Reset these below variables to 0 at the begining of every epoch
        start = time.time()
        correct = 0
        iterations = 0
        iter_loss = 0.0

        model.train()                   # Put the network into training mode

        for i, (inputs, labels) in enumerate(train_load):

            # If we have GPU, shift the data to GPU
            CUDA = torch.cuda.is_available()
            if CUDA:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

            optimizer.zero_grad()
            print(inputs.shape)
            outputs = model(inputs)         
            loss = criterion(outputs, labels) 
            iter_loss += loss.item()       # Accumulate the loss
            loss.backward()                 # Backpropagation 
            optimizer.step()                # Update the weights

            # Record the correct predictions for training data 
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum()
            iterations += 1

        # Record the training loss
        train_loss.append(iter_loss/iterations)
        # Record the training accuracy
        train_accuracy.append((correct.item() / len(train_dataset)))

        #Testing
        loss = 0.0
        correct = 0
        iterations = 0
        iter_loss = 0.0

        
        with torch.no_grad():
            
            model.eval()                    # Put the network into evaluation mode

            for i, (inputs, labels) in enumerate(test_load):

                CUDA = torch.cuda.is_available()
                if CUDA:
                    inputs = inputs.to('cuda')
                    labels = labels.to('cuda')

                outputs = model(inputs)     
                loss = criterion(outputs, labels) # Calculate the loss
                iter_loss += loss.item()
                # Record the correct predictions for training data
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()

                iterations += 1

        # Record the Testing loss
        test_loss.append(iter_loss/iterations)
        # Record the Testing accuracy
        test_accuracy.append((correct.item() / len(test_dataset)))
        stop = time.time()

        
        print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Testing Loss: {:.3f}, Testing Acc: {:.3f}, Time: {:.3f}s'
                .format(epoch, num_epochs, train_loss[-1], train_accuracy[-1], 
                        test_loss[-1], test_accuracy[-1], stop-start))

        save_model(num_epochs,model,optimizer,criterion,epoch)
        save_plots(train_accuracy, test_accuracy, train_loss, test_loss)
            
    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    train_accuracy = np.array(train_accuracy)
    test_accuracy = np.array(test_accuracy)
    return model,train_loss,test_loss,train_accuracy,test_accuracy


def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, labels, colors,
                       loss_legend_loc='upper center', acc_legend_loc='upper left', legend_font=15,
                       fig_size=(16, 8), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2)):
    
    plt.rcParams["figure.figsize"] = fig_size
    plt.figure
    
    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])
    
    for i in range(len(train_loss)):
        x_train = range(len(train_loss[i]))
        x_val = range(len(val_loss[i]))
        
        min_train_loss = train_loss[i].min()
        
        min_val_loss = val_loss[i].min()
        
        plt.plot(x_train, train_loss[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN ({0:.4}): {1}".format(min_train_loss, labels[i]))
        plt.plot(x_val, val_loss[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID ({0:.4}): {1}".format(min_val_loss, labels[i]))
        
    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc, prop={'size': legend_font})
    plt.title('Training and Validation Loss')
        
    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])
    
    for i in range(len(train_acc)):
        x_train = range(len(train_acc[i]))
        x_val = range(len(val_acc[i]))
        
        max_train_acc = train_acc[i].max() 
        
        max_val_acc = val_acc[i].max() 
        
        plt.plot(x_train, train_acc[i], linestyle='-', color='tab:{}'.format(colors[i]), 
                 label="TRAIN ({0:.4}): {1}".format(max_train_acc, labels[i]))
        plt.plot(x_val, val_acc[i], linestyle='--' , color='tab:{}'.format(colors[i]), 
                 label="VALID ({0:.4}): {1}".format(max_val_acc, labels[i]))
        
    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc, prop={'size': legend_font})
    plt.title('Training and Validation Accuracy')
    
    plt.show()
    
    return



if __name__ == '__main__':

    model = CNN_dropout()

    model_dropout,train_loss_2,val_loss_2,train_acc_2,val_acc_2 = train_model(0.01,model, weight_decay=0, num_epochs = 500)

    
     