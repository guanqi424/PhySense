import argparse
import pygad
import copy
import os
import pickle
import re
import time
import cv2

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from tqdm import tqdm, trange

from util import yolov3_pytorch
from util import sticker_apply
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

from phySense_initial import phySense_adaptive as phySense
from phySense_initial.phySense import GraphDataset_addgt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Adaptive attack evaluation")

parser.add_argument(
    '--option', 
    choices=['defenseoutput', 'texture_behavior', 'temporalgraph'], 
    default='defenseoutput', 
)

args = parser.parse_args()


def tensor2cv2(input_tensor: torch.Tensor):
    my_input_tensor = input_tensor.detach()
    my_input_tensor = my_input_tensor.to(torch.device('cpu')).numpy()
    in_arr = np.transpose(my_input_tensor, (1, 2, 0))
    cv2img = cv2.cvtColor(np.uint8(in_arr * 255), cv2.COLOR_RGB2BGR)
    return cv2img


def showtensor(img_tensor, save_dir):
    img_tensor_display = img_tensor.detach().permute(1, 2, 0).cpu()

    image = img_tensor_display.numpy()
    plt.imshow(image)
    plt.savefig(save_dir)
    plt.show()

label_names = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']
coco_to_labels = {1:0, # bicycle
                  5:1, # bus
                  2:2, # car
                  0:3, # pedestrian
                  7:4} # truck
labels_to_coco = {0:1, # bicycle
                  1:5, # bus
                  2:2, # car
                  3:0, # pedestrian
                  4:7} # truck

class yolo_with_guard_car_detection():
    def __init__(self, device, lstm_label_df, trust_frame_num):
        self.device = device
        self.yolo = yolov3_pytorch.yolov3(device)
        self.phySense = phySense.phySense(lstm_model_path='weights/at_bilstm.pth',
                                                lstm_label_path='weights/lstm_label_to_int.pkl',
                                                file_path='weights/bayesian_dataset_nusc.json',
                                                num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                bayesian_model_path='weights/xgb_lpb_model.json',
                                                label_encoder_path='weights/label_encoder_classes.npy',
                                                edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                num_states=5, num_actions=15, num_inter_actions=1,
                                                beam_size=5, device=device, lstm_label_df=lstm_label_df, 
                                                small_labelspace=True, use_precompute=False, trust_frame_num=trust_frame_num)

        crf_state_dict = torch.load('weights/crf_model_best.pth')
        self.phySense.crf.load_state_dict(crf_state_dict)
        self.phySense.to(device)

    def inference(self, indices, filenames, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks,
                  batch_edges, interactions, graph_nums, batch_edge_masks):
        # filenames should be already pert-applied images
        guard_indices = []
        with torch.no_grad():
            target_images = []
            for i, filename in enumerate(filenames):
                for j in indices[i]:
                    guard_indices.append(j)
                    image = Image.open(filename[j]).convert("RGB")
                    image = TF.to_tensor(image).to(self.device)
                    target_images.append(image)
            target_images = torch.stack(target_images, dim=0)

            target_images = sticker_apply.resize_and_pad_tensors(target_images, self.yolo.img_size)

            outputYOLO = self.yolo.predict(target_images)
            results_YOLO = []

            if outputYOLO is not None:
                for detections in outputYOLO:
                    class_pred = -1
                    is_target_class = False
                    if detections is not None:
                        for detection in detections:
                            class_pred = int(detection[-1])
                            break
                            is_target_class = (int(class_pred) == 2)
                            if is_target_class:
                                break

                    results_YOLO.append(class_pred)
            else:
                raise ValueError('YOLO returns nothing')
            
            outputGuard, times, texture_prob, behavior_prob = self.phySense.decode(filenames, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks,
                                                           batch_edges, interactions, graph_nums, batch_edge_masks, self.device, None) # copy.deepcopy(results_YOLO))
            behavior_prob = behavior_prob.sum(axis=-1)
            output_guard = []
            texture_prob_res = []
            behavior_prob_res = []
            for i in guard_indices:
                output_guard.append(int(outputGuard[0][i]))
                texture_prob_res.append(texture_prob[i])
                behavior_prob_res.append(behavior_prob[0][i])
            return results_YOLO, output_guard, texture_prob_res, behavior_prob_res

# 1. Model perpare

with open('./weights/lstm_label_df.pkl', 'rb') as file:
    lstm_label_df = pickle.load(file)
myGuard = yolo_with_guard_car_detection(device, lstm_label_df, 3)

with open('./data/nusc/attacked_selected/advpath_phys_attacked_instances_car_cornershash.pkl', 'rb') as f:
    phys_attacked_instances_car = pickle.load(f)
    phys_attacked_instances_car = dict(sorted(phys_attacked_instances_car.items()))
with open('./data/nusc/scene_token_chain.pkl', 'rb') as file:
    scene_token_chain = pickle.load(file)

attacked_car_phys = {}

for key, value in phys_attacked_instances_car.items():
    if value[0] < 0.7:
        continue
    attacked_pics = [item for item in value[1] if item['attacked']]
    scenenum = int(key[5:8])
    token_chain = scene_token_chain[scenenum]
    if scenenum not in attacked_car_phys:
        attacked_car_phys[scenenum] = {}
    added_pics = 0
    for i, frame_token in enumerate(token_chain):
        if frame_token not in attacked_car_phys[scenenum]:
            attacked_car_phys[scenenum][frame_token] = []
        for pic_dict in attacked_pics:
            if pic_dict['sample_token'] == frame_token:
                attacked_car_phys[scenenum][frame_token].append(pic_dict['path_adv'])
                added_pics += 1
    assert added_pics == len(attacked_pics)

with open('./data/nusc/project_point_dict_qd3dt_cornershash_car.pkl', 'rb') as file:
    project_point_dict_nongt = pickle.load(file)

mask = Image.open('./data/masks/mask_1024x1024_adaptive.png')
mask = TF.to_tensor(mask).to(device)

# prepare dataset

with open('./data/nusc/crf_dataset_full_part_lengths.pkl', 'rb') as file:
    part_lengths = pickle.load(file)

dataset_per_scene = GraphDataset_addgt('./data/nusc/crf_dataset_full/', 'weights/label_encoder_classes.npy', 10, part_lengths)

def obj_func(solution_tensor, part_start, part_end):
    total_pert = 0
    missed = 0
    attacked_res = {}

    for scene_num, frames_len in tqdm(enumerate(part_lengths[part_start:part_end]), total=part_end-part_start):# len(part_lengths)):
        attacked_res[scene_num] = {}
        for cur_frame_num in range(frames_len):
            datapoint = dataset_per_scene[cur_frame_num+sum(part_lengths[:part_start+scene_num])]
            datapoint = copy.deepcopy(datapoint)

            # We need to use actual frame num durning apply attack
            actual_frame_num = -1
            parts = datapoint[8][0].split('/')
            scene_part = parts[3].split('#')
            this_sample_token = scene_part[0]
            for index, frame_token in enumerate(scene_token_chain[part_start+scene_num]):
                if frame_token == this_sample_token:
                    actual_frame_num = index
                    break
            assert actual_frame_num != -1

            # change filenames to pert image
            pert_indices = []
            pert_filenames = []
            change_flag = False

            ori_filenames = copy.deepcopy(datapoint[8])

            for i, path in enumerate(datapoint[8]):
                datapoint[8][i] = './data/nusc/bbox_dataset/b'+path[8:]
                
            if part_start+scene_num in attacked_car_phys:
                for attacked_car in attacked_car_phys[part_start+scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(ori_filenames):
                        if attacked_car.split('/')[-1] not in ori_filename:
                            continue

                        target_img = Image.open(datapoint[8][pert_index]).convert("RGB")
                        target_img = TF.to_tensor(target_img).to(device)

                        transformed_masks_tensor, transformed_masks_tensor_unpadded, flags = sticker_apply.project_mask_pytorch_batch_worigin(
                            target_img.unsqueeze(0), [ori_filenames[pert_index]], solution_tensor,
                            False,
                            project_point_dict_nongt,
                            myGuard.yolo.img_size)

                        transformed_masks_tensor = transformed_masks_tensor.to(device)

                        stickers_transformed_tensor, stickers_transformed_tensor_unpadded, _ = sticker_apply.project_mask_pytorch_batch_worigin(
                            target_img.unsqueeze(0),
                            [ori_filenames[pert_index]],
                            mask, True,
                            project_point_dict_nongt,
                            myGuard.yolo.img_size)
                        stickers_transformed_tensor = stickers_transformed_tensor.to(device)

                        pert_imgs = (sticker_apply.resize_and_pad_tensors(target_img.unsqueeze(0), myGuard.yolo.img_size).to(device)
                                    * (1 - stickers_transformed_tensor) + transformed_masks_tensor * stickers_transformed_tensor)
                        
                        pert_filename = './adaptive_attack/gad_pert/' + attacked_car.split('/')[-1]
                        cv2.imwrite(pert_filename, tensor2cv2(pert_imgs[0]))
                        datapoint[8][pert_index] = pert_filename
                        change_flag = True
                        pert_indices.append(pert_index)
                        pert_filenames.append([part_start+scene_num, this_sample_token, attacked_car])
                        break
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
            
            if len(pert_indices) == 0:
                continue
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint[:-1]],n_states=5)

            padded_beh_seqs, batch_edges, batch_edge_masks, size_xs, size_ys, size_zs, padded_labels, batch_masks, interactions = \
                padded_beh_seqs.to(device), batch_edges.to(device), batch_edge_masks.to(device), \
                    size_xs.to(device), size_ys.to(device), size_zs.to(device), padded_labels.to(
                    device), batch_masks.to(
                    device), interactions.to(device)
            outputYOLO, outputGuard, texture_prob, behavior_prob = \
                myGuard.inference([pert_indices], filenames, size_xs, size_ys, size_zs, padded_beh_seqs,
                                  lengths, batch_masks, batch_edges,
                                  interactions, graph_num, batch_edge_masks)
            total_pert += len(pert_indices)
            for i in range(len(outputYOLO)):
                if outputYOLO[i] not in coco_to_labels:
                    pass
                else:
                    perception_result = coco_to_labels[outputYOLO[i]]
                    # if attack success:
                    if perception_result != 2:
                        # if bypassed
                        if perception_result == outputGuard[i]:
                            missed += 1
                            if cur_frame_num not in attacked_res[scene_num]:
                                attacked_res[scene_num][cur_frame_num] = []
                            attacked_res[scene_num][cur_frame_num].append(pert_filenames[i])
    print(f"{int((missed/total_pert)*100)} % attacked object frames bypassed our defense.")
    return attacked_res

if not os.path.exists(f'./adaptive_attack/best_solution_{args.option}.npy'):
    print()
    print(f"./adaptive_attack/best_solution_{args.option}.npy which will be generated by running adaptive attack does not exist! If you want to reproduce our result, please run attack_reproduced.py.")
    myGuard.phySense.bayesianUnary.texture_predictor.shutdown()
    exit()
    
solution = np.load(f'./adaptive_attack/best_solution_{args.option}.npy')
solution_tensor = torch.tensor(solution.reshape((3, 600, 600)), dtype=torch.float32, device=device)
attacked_res = obj_func(solution_tensor, 0, 150)

with open(f'./adaptive_attack/{args.option}_attacked.pkl', 'wb') as f:
    pickle.dump(attacked_res, f)

myGuard.phySense.bayesianUnary.texture_predictor.shutdown()
exit()
