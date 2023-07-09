import torchvision.models as models
import torch.nn as nn

import timm
def build_model(pretrained=True, fine_tune=True, num_classes=3):
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    else:
        print('[INFO]: Not loading pre-trained weights')
    model = models.efficientnet_b5(weights="EfficientNet_B5_Weights.IMAGENET1K_V1")
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
    # Change the final classification head.
    model.classifier[1] = nn.Linear(in_features=2048, out_features=num_classes)
    return model