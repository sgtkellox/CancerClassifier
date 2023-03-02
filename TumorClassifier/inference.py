import torch
import cv2
import torchvision.transforms as transforms
import argparse
from model import CNNModel
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', 
    default=r'E:\models\Images\test\Oligo\O3-N21-463Q_10001_44001.jpg',
    help='path to the input image')
args = vars(parser.parse_args())

# the computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# list containing all the class labels
labels = [
    'Astro', 'GBM', "Oligo"
    ]

# initialize the model and load the trained weights
model = CNNModel().to(device)
checkpoint = torch.load(r'C:\Users\felix\Desktop\Neuro\models\model286.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# define preprocess transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])  # read and preprocess the image
image = cv2.imread(args['input'])
# get the ground truth class
gt_class = 'Oligo'
orig_image = image.copy()
# convert to RGB format
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = transform(image)
# add batch dimension
image = torch.unsqueeze(image, 0)
with torch.no_grad():
    outputs = model(image.to(device))
output_label = torch.topk(outputs, 1)
pred_class = labels[int(output_label.indices)]
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
print(f"GT: {gt_class}, pred: {pred_class}")
cv2.imshow('Result', orig_image)
cv2.waitKey(0)
cv2.imwrite(r'E:\models\res.jpg',
    orig_image)

