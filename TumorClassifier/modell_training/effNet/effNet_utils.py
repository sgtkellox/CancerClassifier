import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
import os

outPath = r"/mnt/projects/neuropath_hd/data/modelCollection/kryo/effNet/v1_40x_448_ar_sn"
def save_model(epochs, model, optimizer, criterion, pretrained):
    """
    Function to save the trained model to disk.
    
    
    """
    
    safePath = os.path.join(outPath,"model"+ str(epochs)+".pth")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, safePath)

def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
    safePath = os.path.join(outPath,"acc.png")
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(safePath)
    plt.close()
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    safePath = os.path.join(outPath,"loss.png")
    plt.savefig(safePath)
    plt.close()