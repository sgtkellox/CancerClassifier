
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import os 

#from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
#from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from torchvision.models import resnet18, ResNet18_Weights




import cv2





def trainResNet():
    


    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor(),
                                           ])    
    test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          ])    
    train_data = datasets.ImageFolder(r"C:\Users\felix\Desktop\kryoSplitSN\kryo\train",       
                        transform=train_transforms)
    test_data = datasets.ImageFolder(r"C:\Users\felix\Desktop\kryoSplitSN\kryo\val",
                        transform=test_transforms)    
    num_train = len(train_data)

    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True,num_workers=2, batch_size=50)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=50)
 


    print(trainloader.dataset.classes)
    print("cuda " + str(torch.cuda.is_available()))

    device = torch.device("cuda" if torch.cuda.is_available() 
                                      else "cpu")


    model = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")

    print(model)

    for param in model.parameters():
        param.requires_grad = False


    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(fc_inputs, 256),
                                     nn.ReLU(),
                                     nn.Dropout(0.5),
                                     nn.Linear(256, 3),
                                     nn.LogSoftmax(dim=1))
    criterion = nn.CrossEntropyLoss()
    print(model)
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
    model.to(device)

    epochs = 1000
    steps = 0
    running_loss = 0
    print_every = 1000
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(steps)
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
                train_losses.append(running_loss/len(trainloader))
                test_losses.append(test_loss/len(testloader))                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()
        modelPath = os.path.join(r'C:\Users\felix\Desktop\resNet\models',"model"+str(epoch) + ".pth")
        torch.save(model, modelPath)
        plt.plot(train_losses, label='Training loss')
        plt.plot(test_losses, label='Validation loss')
        plt.legend(frameon=False)
        plt.savefig(r'C:\Users\felix\Desktop\reNet\loss.png', bbox_inches='tight')

def predict_image(image,test_transforms,device, model):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)
    output = model(input)
    index = output.data.cpu().numpy().argmax()
    return index

def get_random_images(num,data_dir, test_transforms):
    data = datasets.ImageFolder(data_dir, transform=test_transforms)
    classes = data.classes
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    idx = indices[:num]
    from torch.utils.data.sampler import SubsetRandomSampler
    sampler = SubsetRandomSampler(idx)
    loader = torch.utils.data.DataLoader(data, 
                   sampler=sampler, batch_size=num)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    return images, labels

def runInference():
    classes = ['Astro', 'GBM', 'Oligo']
    data_dir = r'C:\Users\felix\Desktop\Neuro\Images\test'
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=torch.load(r'C:\Users\felix\Desktop\Neuro\resNet\models\model27.pth')
    model.eval()
    to_pil = transforms.ToPILImage()
    images, labels = get_random_images(5,data_dir,test_transforms)
    fig=plt.figure(figsize=(10,10))
    for ii in range(len(images)):
        image = to_pil(images[ii])
        index = predict_image(image,test_transforms,device,model)
        sub = fig.add_subplot(1, len(images), ii+1)
        res = int(labels[ii]) == index
        sub.set_title(str(classes[index]) + ":" + str(res))
        plt.axis('off')
        plt.imshow(image)
    plt.show()


def makeTestRun(path):
    

    # the computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    # list containing all the class labels
    labels = [
        'Astro', 'GBM', 'Oligo'
        ]

    # initialize the model and load the trained weights
    
    model=torch.load(r'D:\ClassifierResults\resNet50Kryo\models\model19.pth')
    model.eval()

    # define preprocess transforms
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])

    count = 0
    res = []
    gt = []

    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            resOfCurrentClass = []
            gt_class = file
            testImgs = os.listdir(d)
            gtOfCurrentClass = [file] * len(testImgs)
            for testImg in testImgs:
                if not testImg.endswith(".jpg"):
                    continue
                imgPath = os.path.join(d,testImg)
                image = cv2.imread(imgPath)
                orig_image = image.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                to_pil = transforms.ToPILImage()
                image = to_pil(image)

                image_tensor = test_transforms(image).float()
                image_tensor = image_tensor.unsqueeze_(0)
                input = Variable(image_tensor)
                input = input.to(device)
                output = model(input)
                index = output.data.cpu().numpy().argmax()
                #output_label = torch.topk(outputs, 1)
              
                pred_class = labels[index]
                resOfCurrentClass.append(pred_class)
                cv2.putText(orig_image, 
                f"GT: {gt_class}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (0, 255, 0), 2, cv2.LINE_AA
                )
                cv2.putText(orig_image, 
                    f"Pred: {pred_class}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 0, 255), 2, cv2.LINE_AA
                )
                
                if index == count:
                    outpath = os.path.join(r"D:\ClassifierResults\resNet50Kryo\sucess",testImg)
                    
                else:
                    outpath = os.path.join(r"D:\ClassifierResults\resNet50Kryo\fail",testImg)
                cv2.imwrite(outpath, orig_image)
                    
            count+=1
            res+=resOfCurrentClass
            gt+= gtOfCurrentClass

    return res, gt


def makeGradCam():

    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])
    classes = ['Astro', 'GBM', 'Oligo']
    data_dir = r'C:\Users\felix\Desktop\Neuro\KryoSplit\test\Astro\A2-N17-1686K_60001_20001.jpg'
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkPoint =torch.load(r'D:\ClassifierResults\resNet50Kryo\models\model18.pth')
    
    model  = models.resnet50(pretrained=True)
    

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(512, 10),
                                     nn.LogSoftmax(dim=1))
    
    model.load_state_dict(checkPoint['model_state_dict'])

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    model.eval()
    model.to(device)
    image = cv2.imread(data_dir)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    to_pil = transforms.ToPILImage()
    image = to_pil(image)

    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor)
    input = input.to(device)            
    
    
    #targets  = [ClassifierOutputTarget(3)]
    
    
    
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
       
if __name__ == '__main__':
    trainResNet()
   