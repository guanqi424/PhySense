import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle
import torch
import numpy as np
import torchvision as tv
from scipy.optimize import linear_sum_assignment

from tracker import *
from utils.utils import *
from config import *
from util import yolov3_pytorch
from PIL import Image
from torchvision import transforms

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
            transforms.ToTensor(),
        ])
        image_tensor = img_transforms(img).float()
        imgs[i] = image_tensor
    image_tensor = torch.stack(imgs, dim=0)
    return image_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myyolo = yolov3_pytorch.yolov3(device)

with open('./data/nusc/attacked_sequences.pkl', 'rb') as f:
    sequences = pickle.load(f)


# Initialize model

classes = load_classes("./object_track/coco.names")
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

asr_yolo = []
asr_sort = []
frame_lens = []

# Loop through each image file
for seq_idx, image_files in tqdm(enumerate(sequences), total=len(sequences)):
    if 'car' in image_files[1]:
        this_label = 2
    elif 'ped' in image_files[1]:
        this_label = 0
    else:
        continue
    tracking_list = []
    attacked = 0
    total = len(image_files[1:])
    frame_lens.append(total)

    for frame_idx, image_file in enumerate(image_files[1:]):
        # Read the image
        if 'attacked' in image_file:
            filename = './data/nusc/'+image_file[8:]
        else:
            filename = './data/nusc/bbox_dataset/b'+image_file[8:]
        pert_image = Image.open(filename)
        if pert_image.mode == 'RGBA':
            pert_image = pert_image.convert('RGB')
        pert_image_tensor = pils_to_tensor([pert_image], myyolo.img_size)
        # Process the image
        detections = []
        non_overlaps = []
        with torch.no_grad():
            pred = myyolo.predict(pert_image_tensor)[0] if myyolo.predict(pert_image_tensor)[0] is not None else None
            if pred is not None and len(pred):
                pred[:, :4] = scale_coords(img_size, pred[:, :4], frame_size)
                pred = pred.cpu().numpy()
                for *xyxy, conf, _, cls in pred:
                    if cls == this_label:
                        detections.append(xyxy)
                detections = np.array(detections, dtype=np.float32)
                if len(tracking_list) != 0 and detections.size > 0:
                    tracks = [(t.predict()[0] if t.is_mature() else t.get_state()) for t in tracking_list]
                    tracks = torch.from_numpy(np.array(tracks, dtype=np.float32))
                    ious = tv.ops.box_iou(torch.from_numpy(detections), tracks).numpy()
                    ious *= ious >= iou_thres
                    row_ind, col_ind = linear_sum_assignment(-ious)
                    for i in range(len(detections)):
                        if i in row_ind:
                            ind, = np.where(row_ind == i)
                            col = int(col_ind[ind])
                            if ious[i, col] > 0.:
                                tracking_list[col].update(detections[i])
                                tracking_list[col].detected = True
                                continue
                        tracking_list.append(KalmanObjectTracker(detections[i]))
                        tracking_list[-1].color = colors[np.random.randint(low=0, high=len(colors))]
                else:
                    for i in range(len(detections)):
                        tracking_list.append(KalmanObjectTracker(detections[i]))
                        tracking_list[-1].color = colors[np.random.randint(low=0, high=len(colors))]

        new_tl = tracking_list[:]
        for tracker in tracking_list:
            if tracker.is_mature() and not tracker.expired() and tracker.detected:
                try:
                    xyxy, lbl, color = tracker.get_state(), str(tracker.id), tracker.color
                except ValueError:
                    print("NaN occurred.")
                    new_tl.remove(tracker)
            elif tracker.expired():
                new_tl.remove(tracker)
            tracker.detected = False
        tracking_list = new_tl
        if len(tracking_list) == 0:
            attacked += 1 
    asr_sort.append(attacked/total)
    asr_yolo.append(image_files[0])
print()
print(f'The average number of frames per object instance is {np.mean(frame_lens):.1f}, and the mean attack successful rate is {np.mean(asr_sort)*100:.2f}%.')
