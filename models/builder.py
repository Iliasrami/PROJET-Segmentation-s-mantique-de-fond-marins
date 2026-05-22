from torchvision.models.segmentation import deeplabv3_resnet50
import torch
from backboned_unet import Unet

def get_model(model_name, num_classes, backbone_name="resnet50"):
    if model_name == "UNet":
        model = Unet(backbone_name=backbone_name, classes=num_classes)
    elif model_name == "DeepLabV3":
        model = deeplabv3_resnet50(pretrained=True)
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise ValueError(f"Model name {model_name} not recognized")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model