import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import CustomCNN
import os

from dataset import train_loader, valid_loader
from utils import (
    save_model, save_plots,
    get_outputs_binary_list, 
    binary_accuracy, calculate_f1_score
)
# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=4,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-lr', '--learning-rate', type=float,
    dest='learning_rate', default=0.001,
    help='Learning rate for training the model'
)
args = vars(parser.parse_args())

dataPath = r"C:\Users\felix\Desktop\split"


# Training function.
def train(model, trainloader, optimizer, criterion, scheduler=None, epoch=None):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    y_true = []
    y_pred = [] 
    iters = len(trainloader)
    for i, data in enumerate(trainloader):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        labels = labels.float()
        loss = criterion(outputs, labels.view(-1, 1))
        train_running_loss += loss.item()
        # Get the binary predictions, 0 or 1.
        outputs_binary_list = get_outputs_binary_list(
            outputs.clone().detach().cpu()
        )
        # Calculate the accuracy.
        train_running_correct = binary_accuracy(
            labels, outputs, train_running_correct
        )
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
        if i%100 ==0:
            if scheduler is not None:
                scheduler.step(epoch + i / iters)
        y_true.extend(labels.detach().cpu().numpy())
        y_pred.extend(outputs_binary_list)
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    # F1 score.
    f1_score = calculate_f1_score(y_true, y_pred)
    return epoch_loss, epoch_acc, f1_score

# Validation function.
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    y_true = []
    y_pred = []
    counter = 0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            counter += 1
            
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            labels = labels.float()
            loss = criterion(outputs, labels.view(-1, 1))
            valid_running_loss += loss.item()
            # Get the binary predictions, 0 or 1.
            outputs_binary_list = get_outputs_binary_list(
                outputs.clone().detach().cpu()
            )
            # Calculate the accuracy.
            valid_running_correct = binary_accuracy(
                labels, outputs, valid_running_correct
            )
            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(outputs_binary_list)
        
    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    # F1 score.
    f1_score = calculate_f1_score(y_true, y_pred)
    return epoch_loss, epoch_acc, f1_score


def load_ckp(modelPath, model, optimizer):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']


if __name__ == '__main__':
    
    
   
    # Learning_parameters. 
    lr = 1e-3
    epochs = 1000
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    model = CustomCNN(num_classes=1)
    #checkpoint = torch.load(r'C:\Users\felix\Desktop\AutoEncoder\models3\47.pth', map_location='cuda')
    #model.load_state_dict(checkpoint['model_state_dict'])
    model.classifier[2] = nn.Linear(in_features=128, out_features=1)
    model = model.to(device)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Loss function.

    pos_weight = torch.FloatTensor ([4.0]).to(device) 
   
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=25, 
        T_mult=1,
        verbose=False
    )
    #model, optimizer, epoch = load_ckp(r"C:\Users\felix\Desktop\AutoEncoder\models\model83.pth", model, optimizer)
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    train_f1_score, valid_f1_score = [], []
    # Start the training.
    epoch = 0
    while epoch in range(epochs):
        epoch+=1
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc, train_epoch_f1_score = train(
            model, train_loader, 
            optimizer, criterion,
            scheduler=scheduler, epoch=epoch
        )
        valid_epoch_loss, valid_epoch_acc, valid_epoch_f1_score = validate(
            model, valid_loader,  criterion
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        train_f1_score.append(train_epoch_f1_score)
        valid_f1_score.append(valid_epoch_f1_score)
        print(
            f"Training loss: {train_epoch_loss:.3f},",
            f"training acc: {train_epoch_acc:.3f},",
            f"training f1-score: {train_epoch_f1_score:.3f}"
            )
        print(
            f"Validation loss: {valid_epoch_loss:.3f},",
            f"validation acc: {valid_epoch_acc:.3f},",
            f"validation f1-score: {valid_epoch_f1_score:.3f}"
            )
        #print(f"LR at end of epoch {epoch+1} {scheduler.get_last_lr()[0]}")
        print('-'*50)
        
    # Save the trained model weights.
        save_model(epoch, model, optimizer, criterion)
    # Save the loss and accuracy plots.
        save_plots(
            train_acc, valid_acc, 
            train_loss, valid_loss, 
            train_f1_score, valid_f1_score
        )
    print('TRAINING COMPLETE')