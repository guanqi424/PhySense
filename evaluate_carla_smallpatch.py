import copy
import gc
import os
import pickle
import re
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from util import yolov3_pytorch
from PIL import Image
from torchvision import transforms

from phySense import phySense
from scripts.interactions_construct_carla import main as interaction_mean

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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

def create_files_list(directory):
    files_dict = []
    for root, dirs, files in os.walk(directory):
        if files:
            folder_name = os.path.basename(root)
            match = re.match(r'scene0*(\d+)', folder_name)
            extracted = 'scene' + match.group(1)
            for i in files:
                files_dict.append((extracted, i))
    return files_dict


def load_list_from_parts(file_prefix, parts):
    """
    Loads a large list from multiple parts saved on disk using pickle, based on the given file prefix and number of parts.

    :param file_prefix: The prefix used when the files were saved.
    :param parts: The number of parts to load.
    :return: The concatenated list from all the loaded parts.
    """
    combined_list = []

    for i in tqdm(range(parts)):
        filename = f'{file_prefix}_part{i}.pkl'
        with open(filename, 'rb') as file:
            part_list = pickle.load(file)
            combined_list.extend(part_list)

    return combined_list


from multiprocessing import set_start_method
if __name__ == '__main__':
    set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Model perpare

    with open('weights/lstm_label_df.pkl', 'rb') as file:
        lstm_label_df = pickle.load(file)

    myyolo = yolov3_pytorch.yolov3(device)
    crf_model_path = 'weights/crf_model_best.pth'
    # 2. Read attacked instances

    with open('data/carla/capatch_attacked_instances_car_small.pkl', 'rb') as f:
        capatch_attacked_instances_car_3 = pickle.load(f)
        capatch_attacked_instances_car_3 = dict(sorted(capatch_attacked_instances_car_3.items()))
    with open('data/carla/phys_attacked_instances_car_small.pkl', 'rb') as f:
        phys_attacked_instances_car = pickle.load(f)
        phys_attacked_instances_car = dict(sorted(phys_attacked_instances_car.items()))
    with open('data/carla/slap_attacked_instances_car_small.pkl', 'rb') as f:
        slap_attacked_instances_car = pickle.load(f)
        slap_attacked_instances_car = dict(sorted(slap_attacked_instances_car.items()))
    with open('data/carla/capatch_attacked_instances_ped_small.pkl', 'rb') as f:
        capatch_attacked_instances_ped_3 = pickle.load(f)
        capatch_attacked_instances_ped_3 = dict(sorted(capatch_attacked_instances_ped_3.items()))
    with open('data/carla/phys_attacked_instances_ped_small.pkl', 'rb') as f:
        phys_attacked_instances_ped = pickle.load(f)
        phys_attacked_instances_ped = dict(sorted(phys_attacked_instances_ped.items()))
    with open('data/carla/slap_attacked_instances_ped_small.pkl', 'rb') as f:
        slap_attacked_instances_ped = pickle.load(f)
        slap_attacked_instances_ped = dict(sorted(slap_attacked_instances_ped.items()))

    with open('./data/carla/crf_idx_to_scene.pkl', 'rb') as f:
        crf_idx_to_scene = pickle.load(f)

    have_scene = set()
    for key, value in crf_idx_to_scene.items():
        have_scene.add(value[0])

    with open('./data/carla/carla_dataset.pkl', 'rb') as f:
        carla_dataset = pickle.load(f)

    all_car_instances = set()
    all_car_instance_cnt = 0

    for key, value in carla_dataset.items():
        if '_'.join(key.split('_')[:-2]) not in have_scene:
            continue
        if 'car' not in key:
            continue
        else:
            all_car_instance_cnt += 1
        all_car_instances.add(key)

    all_ped_instances = set()
    all_ped_instance_cnt = 0
    for key, value in carla_dataset.items():
        if '_'.join(key.split('_')[:-2]) not in have_scene:
            continue
        if 'ped' not in key:
            continue
        else:
            all_ped_instance_cnt += 1
        all_ped_instances.add(key)

    def attacked_total_number(phys_attacked_instances_car, capatch_attacked_instances_car, slap_attacked_instances_car, phys_attacked_instances_ped, capatch_attacked_instances_ped, slap_attacked_instances_ped):

        phys_car_attacked = set()
        for key, value in phys_attacked_instances_car.items():
            if '_'.join(key.split('_')[:-2]) not in have_scene:
                continue
            if value[0] >= 0.7:
                phys_car_attacked.add(key)
                
        phys_ped_attacked = set()
        for key, value in phys_attacked_instances_ped.items():
            if '_'.join(key.split('_')[:-2]) not in have_scene:
                continue
            if value[0] >= 0.7:
                phys_ped_attacked.add(key)

        phys_car_benign = all_car_instances - phys_car_attacked
        phys_ped_benign = all_ped_instances - phys_ped_attacked

        capatch_car_attacked = set()
        for key, value in capatch_attacked_instances_car.items():
            if '_'.join(key.split('_')[:-2]) not in have_scene:
                continue
            if value[0] >= 0.7:
                capatch_car_attacked.add(key)

        capatch_ped_attacked = set()
        for key, value in capatch_attacked_instances_ped.items():
            if '_'.join(key.split('_')[:-2]) not in have_scene:
                continue
            if value[0] >= 0.7:
                capatch_ped_attacked.add(key)

        capatch_car_benign = all_car_instances - capatch_car_attacked
        capatch_ped_benign = all_ped_instances - capatch_ped_attacked

        slap_car_attacked = set()
        for key, value in slap_attacked_instances_car.items():
            if '_'.join(key.split('_')[:-2]) not in have_scene:
                continue
            if value[0] >= 0.7:
                slap_car_attacked.add(key)

        slap_ped_attacked = set()
        for key, value in slap_attacked_instances_ped.items():
            if '_'.join(key.split('_')[:-2]) not in have_scene:
                continue
            if value[0] >= 0.7:
                slap_ped_attacked.add(key)

        slap_car_benign = all_car_instances - slap_car_attacked
        slap_ped_benign = all_ped_instances - slap_ped_attacked
        
        print("# Obj")
        print(f"Total: {len(phys_car_attacked)+len(capatch_car_attacked)+len(slap_car_attacked)+len(phys_ped_attacked)+len(capatch_ped_attacked)+len(slap_ped_attacked)+len(phys_car_benign|capatch_car_benign|slap_car_benign|phys_ped_benign|capatch_ped_benign|slap_ped_benign)}")
        print(f"Attacked: {len(phys_car_attacked)+len(phys_ped_attacked)+len(capatch_car_attacked)+len(capatch_ped_attacked)+len(slap_car_attacked)+len(slap_ped_attacked)}")
        print()
        
    attacked_total_number(phys_attacked_instances_car, capatch_attacked_instances_car_3, slap_attacked_instances_car, phys_attacked_instances_ped, capatch_attacked_instances_ped_3, slap_attacked_instances_ped)

    with open('data/carla/scene_name_dict_carla.pkl', 'rb') as file:
        scene_name_dict = pickle.load(file)
    scene_name_dict = {value: key for key, value in scene_name_dict.items()}

    # Make indices against attacked file, organized by frame
    attacked_car_capatch = {}
    attacked_ped_capatch = {}
    attacked_car_phys = {}
    attacked_ped_phys = {}
    attacked_car_slap = {}
    attacked_ped_slap = {}

    for key, value in capatch_attacked_instances_car_3.items():
        if value[0] < 0.7:
            continue
        attacked_pics = value[1]
        scenenum = scene_name_dict['_'.join(key.split('_')[:3])]
        if scenenum not in attacked_car_capatch:
            attacked_car_capatch[scenenum] = {}
        for pic_dict in attacked_pics:
            instance_id = int(pic_dict['path'].split('_')[-2])
            frame_num = int(pic_dict['path'].split('_')[-3].split('/')[-1])
            if frame_num not in attacked_car_capatch[scenenum]:
                attacked_car_capatch[scenenum][frame_num] = []
            if pic_dict['attacked']:
                attacked_car_capatch[scenenum][frame_num].append(pic_dict['path'])

    for key, value in phys_attacked_instances_car.items():
        if value[0] < 0.7:
            continue
        attacked_pics = value[1]
        scenenum = scene_name_dict['_'.join(key.split('_')[:3])]
        if scenenum not in attacked_car_phys:
            attacked_car_phys[scenenum] = {}
        for pic_dict in attacked_pics:
            instance_id = int(pic_dict['path'].split('_')[-2])
            frame_num = int(pic_dict['path'].split('_')[-3].split('/')[-1])
            if frame_num not in attacked_car_phys[scenenum]:
                attacked_car_phys[scenenum][frame_num] = []
            if pic_dict['attacked']:
                attacked_car_phys[scenenum][frame_num].append(pic_dict['path'])

    for key, value in slap_attacked_instances_car.items():
        if value[0] < 0.7:
            continue
        attacked_pics = value[1]
        scenenum = scene_name_dict['_'.join(key.split('_')[:3])]
        if scenenum not in attacked_car_slap:
            attacked_car_slap[scenenum] = {}
        for pic_dict in attacked_pics:
            instance_id = int(pic_dict['path'].split('_')[-2])
            frame_num = int(pic_dict['path'].split('_')[-3].split('/')[-1])
            if frame_num not in attacked_car_slap[scenenum]:
                attacked_car_slap[scenenum][frame_num] = []
            if pic_dict['attacked']:
                attacked_car_slap[scenenum][frame_num].append(pic_dict['path'])

    for key, value in capatch_attacked_instances_ped_3.items():
        if value[0] < 0.7:
            continue
        attacked_pics = value[1]
        scenenum = scene_name_dict['_'.join(key.split('_')[:3])]
        if scenenum not in attacked_ped_capatch:
            attacked_ped_capatch[scenenum] = {}
        for pic_dict in attacked_pics:
            instance_id = int(pic_dict['path'].split('_')[-2])
            frame_num = int(pic_dict['path'].split('_')[-3].split('/')[-1])
            if frame_num not in attacked_ped_capatch[scenenum]:
                attacked_ped_capatch[scenenum][frame_num] = []
            if pic_dict['attacked']:
                attacked_ped_capatch[scenenum][frame_num].append(pic_dict['path'])

    for key, value in phys_attacked_instances_ped.items():
        if value[0] < 0.7:
            continue
        attacked_pics = value[1]
        scenenum = scene_name_dict['_'.join(key.split('_')[:3])]
        if scenenum not in attacked_ped_phys:
            attacked_ped_phys[scenenum] = {}
        for pic_dict in attacked_pics:
            instance_id = int(pic_dict['path'].split('_')[-2])
            frame_num = int(pic_dict['path'].split('_')[-3].split('/')[-1])
            if frame_num not in attacked_ped_phys[scenenum]:
                attacked_ped_phys[scenenum][frame_num] = []
            if pic_dict['attacked']:
                attacked_ped_phys[scenenum][frame_num].append(pic_dict['path'])

    for key, value in slap_attacked_instances_ped.items():
        if value[0] < 0.7:
            continue
        attacked_pics = value[1]
        scenenum = scene_name_dict['_'.join(key.split('_')[:3])]
        if scenenum not in attacked_ped_slap:
            attacked_ped_slap[scenenum] = {}
        for pic_dict in attacked_pics:
            instance_id = int(pic_dict['path'].split('_')[-2])
            frame_num = int(pic_dict['path'].split('_')[-3].split('/')[-1])
            if frame_num not in attacked_ped_slap[scenenum]:
                attacked_ped_slap[scenenum][frame_num] = []
            if pic_dict['attacked']:
                attacked_ped_slap[scenenum][frame_num].append(pic_dict['path'])

    with open('data/carla/carla_box_slap/car_mapping.pkl', 'rb') as f:
        car_mapping = pickle.load(f)
    car_mapping = {v: k for k, v in car_mapping.items()}
    with open('data/carla/carla_box_slap/ped_mapping.pkl', 'rb') as f:
        ped_mapping = pickle.load(f)
    ped_mapping = {v: k for k, v in ped_mapping.items()}

    # prepare dataset

    with open('data/carla/dataset_part_lengths_carla.pkl', 'rb') as file:
        part_lengths = pickle.load(file)

    dataset_per_scene = phySense.GraphDataset_addgt('./data/carla/crf_dataset/', './data/carla/label_encoder_classes_carla.npy', 10, part_lengths, True)

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


    label_names = ['bicycle', 'bus', 'car', 'ped', 'truck']
    perception_correct_phys = 0
    perception_correct_infer_attack_phys = 0
    perception_wrong_phys = 0
    perception_wrong_infer_equal_phys = 0
    perception_wrong_infer_attack_phys = 0
    perception_wrong_infer_true_phys = 0
    total_attacked_phys = 0
    total_attacked_detected_phys = 0
    total_attacked_recovered_phys = 0

    perception_correct_capatch = 0
    perception_correct_infer_attack_capatch = 0
    perception_wrong_capatch = 0
    perception_wrong_infer_equal_capatch = 0
    perception_wrong_infer_attack_capatch = 0
    perception_wrong_infer_true_capatch = 0
    total_attacked_capatch = 0
    total_attacked_detected_capatch = 0
    total_attacked_recovered_capatch = 0

    perception_correct_slap = 0
    perception_correct_infer_attack_slap = 0
    perception_wrong_slap = 0
    perception_wrong_infer_equal_slap = 0
    perception_wrong_infer_attack_slap = 0
    perception_wrong_infer_true_slap = 0
    total_attacked_slap = 0
    total_attacked_detected_slap = 0
    total_attacked_recovered_slap = 0

    perception_correct_benign = 0
    perception_correct_infer_attack_benign = 0
    perception_wrong_benign = 0
    perception_wrong_infer_equal_benign = 0
    perception_wrong_infer_attack_benign = 0
    perception_wrong_infer_true_benign = 0

    times_records = []
    total_time = 0
    total_time_frac = 0

    with open('./data/carla/crf_idx_to_scene.pkl', 'rb') as f:
        crf_idx_to_scene = pickle.load(f)
    print('Start Evaluation')


    myGuard_benign = phySense.phySense(lstm_model_path='weights/at_bilstm.pth',
                                                    lstm_label_path='weights/lstm_label_to_int.pkl',
                                                    file_path='weights/bayesian_dataset_nongt_carla.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='weights/xgb_lpb_model_nongt_carla_8.json',
                                                    label_encoder_path='weights/label_encoder_classes_nongt_carla_8.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path=crf_model_path,
                                                    small_labelspace=True, use_precompute=False, trust_frame_num=3)
                                                    

    myGuard_phys = phySense.phySense(lstm_model_path='weights/at_bilstm.pth',
                                                    lstm_label_path='weights/lstm_label_to_int.pkl',
                                                    file_path='weights/bayesian_dataset_nongt_carla.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='weights/xgb_lpb_model_nongt_carla_8.json',
                                                    label_encoder_path='weights/label_encoder_classes_nongt_carla_8.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path=crf_model_path,
                                                    small_labelspace=True, use_precompute=False, trust_frame_num=3)
                                                    

    for scene_num, frames_len in tqdm(enumerate(part_lengths), total=len(part_lengths)):
        # 1. prepare all the frame data in one list
        benign_frames = []
        phys_frames = []
        benign_perceptions = []
        phys_perceptions = []
        benign_gt = []
        phys_gt = []
        benign_gt_correct = []
        phys_gt_correct = []
        sample_token_list = []
        benign_perts = []
        phys_perts = []
        actual_scene_num = crf_idx_to_scene[scene_num][1]
        for cur_frame_num in range(frames_len):
            datapoint = dataset_per_scene[cur_frame_num+sum(part_lengths[:scene_num])]
            gt_correct = datapoint[9]
            datapoint = datapoint[:-1]

            # We need to use actual frame num durning apply attack
            actual_frame_num = int(datapoint[8][0].split('/')[-1].split('_')[0])

            sample_token_list.append(actual_frame_num)

            # 1. Benign
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint],n_states=5)
            
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'attacked' not in j:
                        filenames[-1].append('./data/carla/carla_box/'+'/'.join(j.split('/')[-3:]))
                    else:
                        filenames[-1].append(j)
            
            perception_results = []
            groundtruth_labels = []
            for img_filename in filenames[0]:
                with Image.open(img_filename) as img:
                    pert_image = img.copy()
                # pert_image = Image.open(img_filename)
                if pert_image.mode == 'RGBA':
                    pert_image = pert_image.convert('RGB')
                pert_image_tensor = pils_to_tensor([pert_image], myyolo.img_size)
                pert_image_output = myyolo.predict(pert_image_tensor)[0].tolist() if myyolo.predict(pert_image_tensor)[0] is not None else None
                for i, name in enumerate(label_names):
                    if name in img_filename.replace('carla', '*****'):
                        groundtruth_labels.append(i)
                        break
                this_perce_res = []
                if pert_image_output is not None:
                    for i in pert_image_output:
                        this_perce_res.append(int(i[-1]))
                perception_results.append(this_perce_res)

            assert len(perception_results) == len(groundtruth_labels)
            
            filenames_flattened = [item for sublist in filenames for item in sublist]
            images = []
            for path in filenames_flattened:
                with Image.open(path) as img:
                    images.append(img.convert('L'))
            # images = [Image.open(path).convert('L') for path in filenames_flattened]
            benign_frames.append((images, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks, batch_edges,
                                    interactions, graph_num, batch_edge_masks, device, copy.deepcopy(perception_results)))
            
            benign_perceptions.append(perception_results)
            benign_gt.append(groundtruth_labels)
            benign_gt_correct.append(gt_correct)
            
            # 2. Attacked by phys

            datapoint_phys = copy.deepcopy(datapoint)
            # Substitute attacked image
            pert_indices = []
            change_flag = False
            
            if actual_scene_num in attacked_car_phys and actual_frame_num in attacked_car_phys[actual_scene_num]:
                for attacked_car in attacked_car_phys[actual_scene_num][actual_frame_num]:
                    for pert_index, ori_filename in enumerate(datapoint_phys[8]):
                        if not(ori_filename.split('/')[-1] == attacked_car.split('/')[-1] and ori_filename.split('/')[-2] == attacked_car.split('/')[-2]):
                            continue
                        attacked_path = './data/carla/car_phys_applied_small/' + '/'.join(attacked_car.split('/')[-2:])
                        datapoint_phys[8][pert_index] = attacked_path
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError("Pert Image not in")
            
            if actual_scene_num in attacked_ped_phys and actual_frame_num in attacked_ped_phys[actual_scene_num]:
                for attacked_ped in attacked_ped_phys[actual_scene_num][actual_frame_num]:
                    for pert_index, ori_filename in enumerate(datapoint_phys[8]):
                        if not(ori_filename.split('/')[-1] == attacked_ped.split('/')[-1] and ori_filename.split('/')[-2] == attacked_ped.split('/')[-2]):
                            continue
                        attacked_path = './data/carla/ped_phys_applied_small/' + '/'.join(attacked_ped.split('/')[-2:])
                        datapoint_phys[8][pert_index] = attacked_path
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError("Pert Image not in")

            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint_phys],n_states=5)
            
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'phys' not in j:
                        filenames[-1].append('./data/carla/carla_box/'+'/'.join(j.split('/')[-3:]))
                    else:
                        filenames[-1].append(j)
            
            perception_results = []
            groundtruth_labels = []
            for img_filename in filenames[0]:
                with Image.open(img_filename) as img:
                    pert_image = img.copy()
                # pert_image = Image.open(img_filename)
                if pert_image.mode == 'RGBA':
                    pert_image = pert_image.convert('RGB')
                pert_image_tensor = pils_to_tensor([pert_image], myyolo.img_size)
                pert_image_output = myyolo.predict(pert_image_tensor)[0].tolist() if myyolo.predict(pert_image_tensor)[0] is not None else None
                for i, name in enumerate(label_names):
                    if name in img_filename.replace('carla', '*****'):
                        groundtruth_labels.append(i)
                        break
                this_perce_res = []
                if pert_image_output is not None:
                    for i in pert_image_output:
                        this_perce_res.append(int(i[-1]))
                perception_results.append(this_perce_res)

            assert len(perception_results) == len(groundtruth_labels)
            
            filenames_flattened = [item for sublist in filenames for item in sublist]
            images = []
            for path in filenames_flattened:
                with Image.open(path) as img:
                    images.append(img.convert('L'))
            # images = [Image.open(path).convert('L') for path in filenames_flattened]
            phys_frames.append((images, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks, batch_edges,
                                    interactions, graph_num, batch_edge_masks, device, copy.deepcopy(perception_results)))
            
            phys_perceptions.append(perception_results)
            phys_gt.append(groundtruth_labels)
            phys_gt_correct.append(gt_correct)
            phys_perts.append(pert_indices)

        # 2. put the sequence into the defense at once
        benign_results, benign_runtime = myGuard_benign.decode_pipeline(benign_frames)
        phys_results, phys_runtime = myGuard_phys.decode_pipeline(phys_frames)
        times_records.append(benign_runtime)
        times_records.append(phys_runtime)

        myGuard_benign.clear_trustworthy_info()
        myGuard_phys.clear_trustworthy_info()
            
        # 3. perform evaluate at once.
        for i, (benign_result, this_sample_token) in enumerate(zip(benign_results, sample_token_list)):
            output = benign_result[0]
            for index, (perception_result, groundtruth_label) in enumerate(zip(benign_perceptions[i], benign_gt[i])):
                perception_correct_flag = False
                for detection in perception_result:
                    if detection in coco_to_labels and coco_to_labels[detection] == groundtruth_label:
                        perception_correct_benign += 1
                        perception_correct_flag = True
                        break
                if perception_correct_flag:
                    if int(output[index]) != groundtruth_label and labels_to_coco[int(output[index])] not in benign_perceptions[i][index]:
                        perception_correct_infer_attack_benign += 1
                
                else:
                    perception_wrong_benign += 1
                    perception_wrong_infer_equal_flag = False
                    for detection in perception_result:
                        if detection == labels_to_coco[int(output[index])]:
                            perception_wrong_infer_equal_benign += 1
                            perception_wrong_infer_equal_flag = True
                            break
                    if not perception_wrong_infer_equal_flag:
                        perception_wrong_infer_attack_benign += 1
                    if int(output[index]) == groundtruth_label:
                        perception_wrong_infer_true_benign += 1 

        
        for i, (phys_result, this_sample_token) in enumerate(zip(phys_results, sample_token_list)):
            output = phys_result[0]

            if actual_scene_num in attacked_car_phys and this_sample_token in attacked_car_phys[actual_scene_num] or actual_scene_num in attacked_ped_phys and this_sample_token in attacked_ped_phys[actual_scene_num]:
                for index, (perception_result, groundtruth_label) in enumerate(zip(phys_perceptions[i], phys_gt[i])):
                    if not phys_gt_correct[i][index]:
                        continue
                    perception_correct_flag = False
                    for detection in perception_result:
                        if detection in coco_to_labels and coco_to_labels[detection] == groundtruth_label:
                            perception_correct_phys += 1
                            perception_correct_flag = True
                            break
                    if perception_correct_flag:
                        if int(output[index]) != groundtruth_label and labels_to_coco[int(output[index])] not in phys_perceptions[i][index]:
                            perception_correct_infer_attack_phys += 1
                    
                    else:
                        perception_wrong_phys += 1
                        perception_wrong_infer_equal_flag = False
                        for detection in perception_result:
                            if detection == labels_to_coco[int(output[index])]:
                                perception_wrong_infer_equal_phys += 1
                                perception_wrong_infer_equal_flag = True
                                break
                        if not perception_wrong_infer_equal_flag:
                            perception_wrong_infer_attack_phys += 1
                        if int(output[index]) == groundtruth_label:
                            perception_wrong_infer_true_phys += 1
                for pert_index in phys_perts[i]:
                    if not phys_gt_correct[i][pert_index]:
                        continue
                    if labels_to_coco[phys_gt[i][pert_index]] not in phys_perceptions[i][pert_index]:
                        total_attacked_phys += 1
                        if labels_to_coco[phys_gt[i][pert_index]] not in phys_perceptions[i][pert_index]:
                            total_attacked_detected_phys += 1
                        if int(output[pert_index]) == phys_gt[i][pert_index]:
                            total_attacked_recovered_phys += 1
                

    myGuard_benign.shutdown()
    myGuard_phys.shutdown()
    del myGuard_benign
    del myGuard_phys
    torch.cuda.empty_cache()
    gc.collect()
                            

    myGuard_capatch = phySense.phySense(lstm_model_path='weights/at_bilstm.pth',
                                                    lstm_label_path='weights/lstm_label_to_int.pkl',
                                                    file_path='weights/bayesian_dataset_nongt_carla.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='weights/xgb_lpb_model_nongt_carla_8.json',
                                                    label_encoder_path='weights/label_encoder_classes_nongt_carla_8.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path=crf_model_path,
                                                    small_labelspace=True, use_precompute=False, trust_frame_num=3)
                                                    

    myGuard_slap = phySense.phySense(lstm_model_path='weights/at_bilstm.pth',
                                                    lstm_label_path='weights/lstm_label_to_int.pkl',
                                                    file_path='weights/bayesian_dataset_nongt_carla.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='weights/xgb_lpb_model_nongt_carla_8.json',
                                                    label_encoder_path='weights/label_encoder_classes_nongt_carla_8.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path=crf_model_path,
                                                    small_labelspace=True, use_precompute=False, trust_frame_num=3)


    for scene_num, frames_len in tqdm(enumerate(part_lengths), total=len(part_lengths)):
        # 1. prepare all the frame data in one list
        capatch_frames = []
        slap_frames = []
        capatch_perceptions = []
        slap_perceptions = []
        capatch_gt = []
        slap_gt = []
        capatch_gt_correct = []
        slap_gt_correct = []
        sample_token_list = []
        capatch_perts = []
        slap_perts = []
        actual_scene_num = crf_idx_to_scene[scene_num][1]
        for cur_frame_num in range(frames_len):
            datapoint = dataset_per_scene[cur_frame_num+sum(part_lengths[:scene_num])]
            gt_correct = datapoint[9]
            datapoint = datapoint[:-1]

            # We need to use actual frame num durning apply attack
            actual_frame_num = int(datapoint[8][0].split('/')[-1].split('_')[0])

            sample_token_list.append(actual_frame_num)
                
            # 3. Attacked by capatch

            datapoint_capatch = copy.deepcopy(datapoint)
            # Substitute attacked image
            pert_indices = []
            change_flag = False
                
            if actual_scene_num in attacked_car_capatch and actual_frame_num in attacked_car_capatch[actual_scene_num]:
                for attacked_car in attacked_car_capatch[actual_scene_num][actual_frame_num]:
                    for pert_index, ori_filename in enumerate(datapoint_capatch[8]):
                        if not(ori_filename.split('/')[-1] == attacked_car.split('/')[-1] and ori_filename.split('/')[-2] == attacked_car.split('/')[-2]):
                            continue
                        attacked_path = './data/carla/carla_car_capatch_small/' + '/'.join(attacked_car.split('/')[-2:])
                        datapoint_capatch[8][pert_index] = attacked_path
                        pert_indices.append(pert_index)
                        change_flag = True
                        break
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError("Pert Image not in")
                
            if actual_scene_num in attacked_ped_capatch and actual_frame_num in attacked_ped_capatch[actual_scene_num]:
                for attacked_ped in attacked_ped_capatch[actual_scene_num][actual_frame_num]:
                    for pert_index, ori_filename in enumerate(datapoint_capatch[8]):
                        if not(ori_filename.split('/')[-1] == attacked_ped.split('/')[-1] and ori_filename.split('/')[-2] == attacked_ped.split('/')[-2]):
                            continue
                        attacked_path = './data/carla/carla_ped_capatch_small/' + '/'.join(attacked_ped.split('/')[-2:])
                        datapoint_capatch[8][pert_index] = attacked_path
                        pert_indices.append(pert_index)
                        change_flag = True
                        break
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError("Pert Image not in")
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint_capatch],n_states=5)
            
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'capatch' not in j:
                        filenames[-1].append('./data/carla/carla_box/'+'/'.join(j.split('/')[-3:]))
                    else:
                        filenames[-1].append(j)
            
            perception_results = []
            groundtruth_labels = []
            for img_filename in filenames[0]:
                with Image.open(img_filename) as img:
                    pert_image = img.copy()
                # pert_image = Image.open(img_filename)
                if pert_image.mode == 'RGBA':
                    pert_image = pert_image.convert('RGB')
                pert_image_tensor = pils_to_tensor([pert_image], myyolo.img_size)
                pert_image_output = myyolo.predict(pert_image_tensor)[0].tolist() if myyolo.predict(pert_image_tensor)[0] is not None else None
                for i, name in enumerate(label_names):
                    if name in img_filename.replace('carla', '*****'):
                        groundtruth_labels.append(i)
                        break
                this_perce_res = []
                if pert_image_output is not None:
                    for i in pert_image_output:
                        this_perce_res.append(int(i[-1]))
                perception_results.append(this_perce_res)

            assert len(perception_results) == len(groundtruth_labels)
            
            filenames_flattened = [item for sublist in filenames for item in sublist]
            images = []
            for path in filenames_flattened:
                with Image.open(path) as img:
                    images.append(img.convert('L'))
            # images = [Image.open(path).convert('L') for path in filenames_flattened]
            capatch_frames.append((images, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks, batch_edges,
                                    interactions, graph_num, batch_edge_masks, device, copy.deepcopy(perception_results)))
            
            capatch_perceptions.append(perception_results)
            capatch_gt.append(groundtruth_labels)
            capatch_perts.append(pert_indices)
            capatch_gt_correct.append(gt_correct)
            
            # 4. Attacked by slap
            # print('slap'

            datapoint_slap = copy.deepcopy(datapoint)
            # Substitute attacked image
            pert_indices = []
            change_flag = False
                
            if actual_scene_num in attacked_car_slap and actual_frame_num in attacked_car_slap[actual_scene_num]:
                for attacked_car in attacked_car_slap[actual_scene_num][actual_frame_num]:
                    for pert_index, ori_filename in enumerate(datapoint_slap[8]):
                        if not(ori_filename.split('/')[-1] == attacked_car.split('/')[-1] and ori_filename.split('/')[-2] == attacked_car.split('/')[-2]):
                            continue
                        attacked_path = './data/carla/car_applied_carla_small/' + car_mapping['./'+'/'.join(attacked_car.split('/')[2:])]
                        datapoint_slap[8][pert_index] = attacked_path
                        pert_indices.append(pert_index)
                        change_flag = True
                        break
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError("Pert Image not in")
                
            if actual_scene_num in attacked_ped_slap and actual_frame_num in attacked_ped_slap[actual_scene_num]:
                for attacked_ped in attacked_ped_slap[actual_scene_num][actual_frame_num]:
                    for pert_index, ori_filename in enumerate(datapoint_slap[8]):
                        if not(ori_filename.split('/')[-1] == attacked_ped.split('/')[-1] and ori_filename.split('/')[-2] == attacked_ped.split('/')[-2]):
                            continue
                        attacked_path = './data/carla/ped_applied_carla_small/' + ped_mapping['./'+'/'.join(attacked_ped.split('/')[2:])]
                        datapoint_slap[8][pert_index] = attacked_path
                        pert_indices.append(pert_index)
                        change_flag = True
                        break
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError("Pert Image not in")
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint_slap],n_states=5)
            
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'applied' not in j:
                        filenames[-1].append('./data/carla/carla_box/'+'/'.join(j.split('/')[-3:]))
                    else:
                        filenames[-1].append(j)
            
            perception_results = []
            groundtruth_labels = []
            for img_filename in filenames[0]:
                with Image.open(img_filename) as img:
                    pert_image = img.copy()
                # pert_image = Image.open(img_filename)
                if pert_image.mode == 'RGBA':
                    pert_image = pert_image.convert('RGB')
                pert_image_tensor = pils_to_tensor([pert_image], myyolo.img_size)
                pert_image_output = myyolo.predict(pert_image_tensor)[0].tolist() if myyolo.predict(pert_image_tensor)[0] is not None else None
                for i, name in enumerate(label_names):
                    if name in img_filename.replace('carla', '*****'):
                        groundtruth_labels.append(i)
                        break
                this_perce_res = []
                if pert_image_output is not None:
                    for i in pert_image_output:
                        this_perce_res.append(int(i[-1]))
                perception_results.append(this_perce_res)

            assert len(perception_results) == len(groundtruth_labels)
            
            filenames_flattened = [item for sublist in filenames for item in sublist]
            images = []
            for path in filenames_flattened:
                with Image.open(path) as img:
                    images.append(img.convert('L'))
            # images = [Image.open(path).convert('L') for path in filenames_flattened]
            slap_frames.append((images, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks, batch_edges,
                                    interactions, graph_num, batch_edge_masks, device, copy.deepcopy(perception_results)))
            
            slap_perceptions.append(perception_results)
            slap_gt.append(groundtruth_labels)
            slap_perts.append(pert_indices)
            slap_gt_correct.append(gt_correct)

        # 2. put the sequence into the defense at once
        capatch_results, capatch_runtime = myGuard_capatch.decode_pipeline(capatch_frames)
        slap_results, slap_runtime = myGuard_slap.decode_pipeline(slap_frames)
        times_records.append(capatch_runtime)
        times_records.append(slap_runtime)

        myGuard_capatch.clear_trustworthy_info()
        myGuard_slap.clear_trustworthy_info()

        # 3. perform evaluate at once.
        for i, (capatch_result, this_sample_token) in enumerate(zip(capatch_results, sample_token_list)):
            output = capatch_result[0]
            if actual_scene_num in attacked_car_capatch and this_sample_token in attacked_car_capatch[actual_scene_num] or actual_scene_num in attacked_ped_capatch and this_sample_token in attacked_ped_capatch[actual_scene_num]:
                for index, (perception_result, groundtruth_label) in enumerate(zip(capatch_perceptions[i], capatch_gt[i])):
                    if not capatch_gt_correct[i][index]:
                        continue
                    perception_correct_flag = False
                    for detection in perception_result:
                        if detection in coco_to_labels and coco_to_labels[detection] == groundtruth_label:
                            perception_correct_capatch += 1
                            perception_correct_flag = True
                            break
                    if perception_correct_flag:
                        if int(output[index]) != groundtruth_label and labels_to_coco[int(output[index])] not in capatch_perceptions[i][index]:
                            perception_correct_infer_attack_capatch += 1
                    
                    else:
                        perception_wrong_capatch += 1
                        perception_wrong_infer_equal_flag = False
                        for detection in perception_result:
                            if detection == labels_to_coco[int(output[index])]:
                                perception_wrong_infer_equal_capatch += 1
                                perception_wrong_infer_equal_flag = True
                                break
                        if not perception_wrong_infer_equal_flag:
                            perception_wrong_infer_attack_capatch += 1
                        if int(output[index]) == groundtruth_label:
                            perception_wrong_infer_true_capatch += 1
                
                for pert_index in capatch_perts[i]:
                    if not capatch_gt_correct[i][pert_index]:
                        continue
                    if labels_to_coco[capatch_gt[i][pert_index]] not in capatch_perceptions[i][pert_index]:
                        total_attacked_capatch += 1
                        if labels_to_coco[int(output[pert_index])] not in capatch_perceptions[i][pert_index]:
                            total_attacked_detected_capatch += 1
                        if int(output[pert_index]) == capatch_gt[i][pert_index]:
                            total_attacked_recovered_capatch += 1

        
        for i, (slap_result, this_sample_token) in enumerate(zip(slap_results, sample_token_list)):
            output = slap_result[0]
            if actual_scene_num in attacked_car_slap and this_sample_token in attacked_car_slap[actual_scene_num] or actual_scene_num in attacked_ped_slap and this_sample_token in attacked_ped_slap[actual_scene_num]:
                for index, (perception_result, groundtruth_label) in enumerate(zip(slap_perceptions[i], slap_gt[i])):
                    if not slap_gt_correct[i][index]:
                        continue
                    perception_correct_flag = False
                    for detection in perception_result:
                        if detection in coco_to_labels and coco_to_labels[detection] == groundtruth_label:
                            perception_correct_slap += 1
                            perception_correct_flag = True
                            break
                    if perception_correct_flag:
                        if int(output[index]) != groundtruth_label and labels_to_coco[int(output[index])] not in slap_perceptions[i][index]:
                            perception_correct_infer_attack_slap += 1
                    
                    else:
                        perception_wrong_slap += 1
                        perception_wrong_infer_equal_flag = False
                        for detection in perception_result:
                            if detection == labels_to_coco[int(output[index])]:
                                perception_wrong_infer_equal_slap += 1
                                perception_wrong_infer_equal_flag = True
                                break
                        if not perception_wrong_infer_equal_flag:
                            perception_wrong_infer_attack_slap += 1
                        if int(output[index]) == groundtruth_label:
                            perception_wrong_infer_true_slap += 1
                
                for pert_index in slap_perts[i]:
                    if not slap_gt_correct[i][pert_index]:
                        continue
                    if labels_to_coco[slap_gt[i][pert_index]] not in slap_perceptions[i][pert_index]:
                        total_attacked_slap += 1
                        if labels_to_coco[int(output[pert_index])] not in slap_perceptions[i][pert_index]:
                            total_attacked_detected_slap += 1
                        if int(output[pert_index]) == slap_gt[i][pert_index]:
                            total_attacked_recovered_slap += 1

    int_inference_time, graph_construction_time = interaction_mean(runtime_test=True)
    
    myGuard_capatch.shutdown()
    myGuard_slap.shutdown()
    del myGuard_capatch
    del myGuard_slap
    torch.cuda.empty_cache()
    gc.collect()

    print()
    print("Overall Metric:")
    print('Detection Acc.: ', (total_attacked_detected_phys+total_attacked_detected_capatch+total_attacked_detected_slap)/(total_attacked_phys+total_attacked_capatch+total_attacked_slap))
    print('Correction Acc.: ', (total_attacked_recovered_phys+total_attacked_recovered_capatch+total_attacked_recovered_slap)/(total_attacked_phys+total_attacked_capatch+total_attacked_slap))
    print('FPR: ', (perception_correct_infer_attack_benign+perception_correct_infer_attack_phys+perception_correct_infer_attack_capatch+perception_correct_infer_attack_slap)/(perception_correct_benign+perception_correct_phys+perception_correct_capatch+perception_correct_slap))
    print('FNR: ', (perception_wrong_infer_equal_benign+perception_wrong_infer_equal_phys+perception_wrong_infer_equal_capatch+perception_wrong_infer_equal_slap)/(perception_wrong_benign+perception_wrong_phys+perception_wrong_capatch+perception_wrong_slap))


    print()
    print('Avg. Runtime for PhySense:', np.mean(times_records)+int_inference_time+graph_construction_time)
    