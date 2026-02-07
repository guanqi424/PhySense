import os
import torch
from matplotlib import pyplot as plt
from torch import autograd
from torchvision import transforms
from torch.autograd import Variable

from .yolov3.utils.utils import non_max_suppression, load_classes
from PIL import Image

from .yolov3.models import Darknet


def showtensor(img_tensor):
    img_tensor_display = img_tensor.detach().permute(1, 2, 0).cpu()
    image = img_tensor_display.numpy()
    plt.imshow(image)
    plt.show()


def pils_to_tensor(imgs, img_size):
    # Image preprocessing and detection
    for i, img in enumerate(imgs):
        ratio = min(img_size / img.size[0], img_size / img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([
            transforms.Resize((imh, imw)),
            transforms.Pad((max(int((imh - imw) / 2), 0), max(int((imw - imh) / 2), 0), max(int((imh - imw) / 2), 0),
                            max(int((imw - imh) / 2), 0)), (128, 128, 128)),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        image_tensor = img_transforms(img).float()
        imgs[i] = image_tensor
    # image_tensor = image_tensor.unsqueeze_(0)
    image_tensor = torch.stack(imgs, dim=0)
    return image_tensor


def detect_image(imgs_tensor, Tensor, model, conf_thres, nms_thres, device):
    input_img = imgs_tensor.type(Tensor).to(device)
    detections = model(input_img)
    detections = non_max_suppression(detections, 80, conf_thres, nms_thres)
    # detections = [d for d in detections if d is not None]
    '''
    if detections:
        detections = torch.stack(detections, dim=0)
    else:
        detections = torch.tensor([[]])
    '''
    # detections = torch.stack(detections, dim=0)
    return detections


class yolov3():
    def __init__(self, device):
        # YOLOv3 model configuration
        base_path = os.path.dirname(os.path.abspath(__file__))
        
        self.config_path = os.path.join(base_path, 'yolov3/config/yolov3.cfg')
        self.weights_path = os.path.join(base_path, 'yolov3/config/yolov3.weights')
        self.class_path = os.path.join(base_path, 'yolov3/config/coco.names')
        self.img_size = 416
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.Tensor = torch.cuda.FloatTensor
        self.device = device

        # Load YOLOv3 model
        self.model = Darknet(self.config_path, img_size=self.img_size)
        self.model.load_weights(self.weights_path)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        self.classes = load_classes(self.class_path)

    def predict(self, imgs_tensor):
        """return: [batch_size, num_instance*[x, y, w, h, confidence, class_score, class_index]]"""
        detections = detect_image(imgs_tensor, self.Tensor, self.model, self.conf_thres, self.nms_thres, self.device)
        return detections
