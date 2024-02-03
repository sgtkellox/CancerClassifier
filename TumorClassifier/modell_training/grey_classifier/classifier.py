
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import timeit
import unittest

from Net import Net

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
transforms.CenterCrop(26),
transforms.Resize((28,28)),
transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
transforms.RandomRotation(10),      
transforms.RandomAffine(5),

# convert the image to a pytorch tensor
transforms.ToTensor(), 

# normalise the images with mean and std of the dataset
transforms.Normalize((0.1307,), (0.3081,)) 
])


train_dataset = datasets.MNIST('./data',train=True,transform=transform,download=True)
test_dataset =  datasets.MNIST('./data',train=False,transform=transform,download=True)

train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

model = Net().to(device)

print(model.parameters)

losses_1 = []
losses_2 = []


pos_weight = torch.FloatTensor ([4.5]).to(device) 

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
        
    for batch_idx, (data, target) in enumerate(train_loader):
        # send the image, target to the device
        data, target = data.to(device), target.to(device)
        # flush out the gradients stored in optimizer
        optimizer.zero_grad()
        # pass the image to the model and assign the output to variable named output
        output = model(data)
        # calculate the loss (use nll_loss in pytorch)
        loss = F.nll_loss(output, target)
        # do a backward pass
        loss.backward()
        # update the weights
        optimizer.step()
          
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            losses_1.append(loss.item())
            losses_2.append(100. * batch_idx / len(train_loader))
            
accuracy = []
avg_loss = []
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
          
            # send the image, target to the device
            data, target = data.to(device), target.to(device)
            # pass the image to the model and assign the output to variable named output
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
          
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    avg_loss.append(test_loss)
    accuracy.append(100. * correct / len(test_loader.dataset))
    

learning_rate = []
def adjust_learning_rate(optimizer, iter, each):
    # sets the learning rate to the initial LR decayed by 0.1 every 'each' iterations
    lr = 0.001 * (0.95 ** (iter // each))
    state_dict = optimizer.state_dict()
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    print("Learning rate = ",lr)
    return lr


## Define Adam Optimiser with a learning rate of 0.01
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

start = timeit.default_timer()
for epoch in range(1,100):
  lr = adjust_learning_rate(optimizer, epoch, 1.616)
  learning_rate.append(lr)
  train(model, device, train_dataloader, optimizer, epoch)
  test(model, device, test_dataloader)
stop = timeit.default_timer()
print('Total time taken: {} seconds'.format(int(stop - start)))
            
    
            
                
                
