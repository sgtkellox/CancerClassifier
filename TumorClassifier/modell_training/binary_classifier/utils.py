import torch
import matplotlib
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score
matplotlib.style.use('ggplot')
def save_model(epoch, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, r"C:\Users\felix\Desktop\AutoEncoder\models\model"+str(epoch)+".pth")
def save_plots(
    train_acc, valid_acc, 
    train_loss, valid_loss,
    train_f1_score, valid_f1_score
):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # Accuracy plots
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
    plt.savefig(r"C:\Users\felix\Desktop\AutoEncoder\models\accuracy.png")
    
    # Loss plots
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
    plt.savefig(r"C:\Users\felix\Desktop\AutoEncoder\models\loss.png")
    # F1 score plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_f1_score, color='purple', linestyle='-', 
        label='train f1 score'
    )
    plt.plot(
        valid_f1_score, color='olive', linestyle='-', 
        label='validataion f1 score'
    )
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.savefig(r"C:\Users\felix\Desktop\AutoEncoder\models\f1_score.png")

def get_outputs_binary_list(outputs):
    """
    Function to generate a list of binary values depending on the 
    outputs of the model.
    """
    outputs = torch.sigmoid(outputs)
    binary_list = []
    for i, output in enumerate(outputs):
        if outputs[i] < 0.5:
            binary_list.append(0.)
        elif outputs[i] >= 0.5:
            binary_list.append(1.)
    return binary_list
def binary_accuracy(labels, outputs, train_running_correct):
    """
    Function to calculate the binary accuracy of the model.
    """
    outputs = torch.sigmoid(outputs)
    for i, label in enumerate(labels):
        if label < 0.5 and outputs[i] < 0.5:
            train_running_correct += 1
        elif label >= 0.5 and outputs[i] >= 0.5:
            train_running_correct += 1
    return train_running_correct
def calculate_f1_score(y_true, y_pred):
    """
    Function returns F1-Score for predictions and true labels.
    """
    return f1_score(y_true, y_pred)