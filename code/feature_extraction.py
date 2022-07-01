import PIL.Image
import torch, torchvision
import torchvision.transforms as transforms
import clip
from PIL import Image
import numpy as np


def feature_extraction(keyframes, frames_num, method="CLIP"):
    # keyframes: list of key frame

    img_features = []
    if method == "CLIP":
        with torch.no_grad():
            model, preprocess = clip.load("ViT-B/32")
            img_preprocess = torch.cat([preprocess(Image.fromarray(item)).unsqueeze(0) for item in keyframes], dim=0)
            img_features = np.array(model.encode_image(img_preprocess))
            feature2index = {str(img_features[i]): i for i in range(frames_num)}

    elif method == "ResNet":
        with torch.no_grad():
            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((224, 224)),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                ])
            model = torchvision.models.resnet50(pretrained=True)

            img_preprocess = torch.cat([transform(Image.fromarray(item)).unsqueeze(0) for item in keyframes], dim=0)
            img_features = np.array(model(img_preprocess))
            feature2index = {str(img_features[i]): i for i in range(frames_num)}

    else:
        raise NotImplementedError

    return img_features, feature2index



