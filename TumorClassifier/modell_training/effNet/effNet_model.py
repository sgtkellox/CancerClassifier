import torchvision.models as models
import torch.nn as nn


def build_model(pretrained=True, fine_tune=True, num_classes=3):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b0(weights="EfficientNet_B0_Weights.IMAGENET1K_V1")
    
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.

    model.classifier = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512 , 256),
            nn.Linear(256 , num_classes)
        )
    
    
    return model