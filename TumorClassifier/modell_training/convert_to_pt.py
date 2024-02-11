import torch
import torch.nn.functional as F
import torchvision.models as models
from effNet_v2.effNet_model import build_model


def load_ckp(modelPath, model):
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['epoch']

modelPath=r""


model = build_model(
        pretrained=True, 
        fine_tune=True, 
        num_classes=7
    )      # We now have an instance of the pretrained model

model, start_epoch = load_ckp(r"D:\non_glial\non_glial\v2_384_10x\model_60.pth", model)
r18_scripted = torch.jit.script(model)         # *** This is the TorchScript export
dummy_input = torch.rand(1, 3, 384, 384)

unscripted_output = model(dummy_input)         # Get the unscripted model's prediction...
scripted_output = r18_scripted(dummy_input)  # ...and do the same for the scripted version

unscripted_top5 = F.softmax(unscripted_output, dim=1).topk(5).indices
scripted_top5 = F.softmax(scripted_output, dim=1).topk(5).indices


print('Python model top 5 results:\n  {}'.format(unscripted_top5))
print('TorchScript model top 5 results:\n  {}'.format(scripted_top5))

r18_scripted.save(r'D:\non_glial\non_glial\test.pt')