import os
import PIL
import sys
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from torchsummary import summary
import torch
from torch.autograd import Variable
from torch import nn
import torchvision
from torchvision import transforms,datasets,models
#import splitfolders
#from monai.networks.nets import DenseNet201,DenseNet121
from sklearn.metrics import accuracy_score,classification_report


    

if __name__ == '__main__':
    

    path = r"/mnt/projects/neuropath_hd/data/splits/non_glial/448_10x/kryo"
    safePath = r"/mnt/projects/neuropath_hd/data/modelCollection/kryo/mobileNetV3/non_glial/448_10x"

    test_path = os.path.join(path,'test')
    train_path = os.path.join(path,'train')
    val_path = os.path.join(path,'val')


    data_transforms = {
            "train": transforms.Compose([
                transforms.Resize((224,224)), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-3,3)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ]),
            "val": transforms.Compose([
                transforms.Resize((224,224)),          
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                       ]),
            "test":transforms.Compose([
                transforms.Resize((224,224)),           
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
    }
    

    data_sets = {
        "train":datasets.ImageFolder(train_path,transform = data_transforms['train']),
        "val":datasets.ImageFolder(val_path,transform = data_transforms['val']),
        "test":datasets.ImageFolder(test_path,transform = data_transforms['test']),
    }
    num_train = len(data_sets["train"])
    num_val = len(data_sets["val"])
    datasizes = {'train':num_train,'val':num_val}

    data_dicts = {
        'train':[],
        'test':[],
        'val':[]
    }
    data_labels = {
        'train':[],
        'test':[],
        'val':[]
    }

    for phase in ['train','val','test']:
        if phase == 'train':
            data_dir = train_path
        elif phase == 'val':
            data_dir = val_path
        else:
            data_dir = test_path
        class_names = sorted(x for x in os.listdir(data_dir)
                                if os.path.isdir(os.path.join(data_dir, x)))
        num_class = len(class_names) 
        image_files = [
            [
                os.path.join(data_dir, class_names[i], x)
                for x in os.listdir(os.path.join(data_dir, class_names[i]))
            ]
            for i in range(num_class)
        ]
        num_each = [len(image_files[i]) for i in range(num_class)]
        image_files_list = [] 
        image_class = []
        for i in range(num_class):
            image_files_list.extend(image_files[i])
            image_class.extend([i] * num_each[i])
        data_dicts[phase] = image_files_list
        data_labels[phase] = image_class
        num_total = len(image_class)
        image_width, image_height = PIL.Image.open(image_files_list[0]).size
        print(f"{phase} information")
        print(f"Image dimensions: {image_width} x {image_height}")
        print(f"Label names: {class_names}")
        print(f"Label counts: {num_each}")
        print(f"Total image count: {num_total}")
        print('--'*8)
    
    df = pd.DataFrame({'path': data_dicts['train'],'label': data_labels['train']})
    sorted_counts  = df.value_counts('label')
    plt.title('distribution in train dataset')
    plt.pie(sorted_counts, labels = class_names,startangle = 90,counterclock = False,autopct="%.1f%%")

    batch_size = 300
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    dataloaders = {
        "train":torch.utils.data.DataLoader(data_sets['train'],batch_size=batch_size,shuffle=True,num_workers=nw),
        "val":torch.utils.data.DataLoader(data_sets['val'],batch_size=batch_size,shuffle=False,num_workers=nw),
        "test":torch.utils.data.DataLoader(data_sets['test'],batch_size=batch_size,shuffle=False,num_workers=nw)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=models.mobilenet_v3_small(weights="MobileNet_V3_Small_Weights.IMAGENET1K_V1")
    num_features=model.classifier[0].in_features
    model.classifier=nn.Sequential(
        nn.Linear(in_features=num_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=7, bias=True)
      )
    model.to(device)

    optimizer=torch.optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
    loss_function=nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    num_epochs = 800
    best_acc = 0.0
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []


    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                #train_bar = tqdm(dataloaders[phase], file=sys.stdout)
            
            
                for step,data in enumerate(dataloaders[phase]):
                    optimizer.zero_grad()
                    X,y_true = data
                    with torch.set_grad_enabled(phase == 'train'):
                        y_predict = model(X.to(device))
                        _, pred_labels = torch.max(y_predict, 1)
                        # _, pred_labels = torch.max(y_predict.item(), 1)
                        loss = loss_function(y_predict, y_true.to(device))
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * X.size(0)#batch_size
                    running_corrects += torch.sum(pred_labels == y_true.data.to(device))
                    # running_corrects += torch.sum(pred_labels == y_true.data)
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = running_loss / datasizes[phase]
                epoch_acc = running_corrects.double() / datasizes[phase]
                if phase == 'train':
                    train_loss_list.append(epoch_loss)
                    train_acc_list.append(epoch_acc)
                elif phase == 'val':
                    val_loss_list.append(epoch_loss)
                    val_acc_list.append(epoch_acc)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(),os.path.join(safePath,"model"+str(epoch)+".pth"))
                
    print('Best val Acc: {:4f}'.format(best_acc)) 

    acc_dict = {'train':torch.tensor(train_acc_list, device='cpu') .tolist(),'val':torch.tensor(val_acc_list, device='cpu') .tolist()}
    loss_dict = {'train':torch.tensor(train_loss_list, device='cpu') .tolist(),'val':torch.tensor(val_loss_list, device='cpu').tolist()}


    max_accvalue = max(acc_dict['val'])
    max_accidx = acc_dict['val'].index(max_accvalue)+1
    min_lossvalue = min(loss_dict['val'])
    min_lossidx = loss_dict['val'].index(min_lossvalue)+1

    plt.figure(figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title('loss')
    plt.xlabel('epoch')
    plt.grid(alpha=0.4,linestyle=':')
    x = [i + 1 for i in range(num_epochs)]
    for phase in ['train','val']:
        plt.plot(x,loss_dict[phase],'-o',label=phase)
        plt.legend()
        plt.plot(min_lossidx,min_lossvalue,'-ro')
        plt.annotate(round(min_lossvalue,2), xy=(min_lossidx, min_lossvalue))
    plt.subplot(1,2,2)
    plt.title('acc')
    plt.xlabel('epoch')
    plt.grid(alpha=0.4,linestyle=':')
    for phase in ['train','val']:
        plt.plot(x,acc_dict[phase],'-o',label=phase)
        plt.legend()
        plt.plot(max_accidx,max_accvalue,'-ro')
        # plt.annotate(round(max_value,2), xy=(max_idx, max_value),xytext=(max_idx, max_idx))
        plt.annotate(round(max_accvalue,2), xy=(max_accidx, max_accvalue))



