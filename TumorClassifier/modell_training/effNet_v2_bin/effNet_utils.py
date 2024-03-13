import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import os

outPath = r"/mnt/projects/neuropath_hd/data/modelCollection/smear/effNet/other/384_10x"
def save_model(epochs, model, optimizer, criterion, pretrained):
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

def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained):
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
    path = os.path.join(outPath,"acc.png")
    plt.savefig(path)
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
    path = os.path.join(outPath,"loss.png")
    plt.savefig(path)
    plt.close()