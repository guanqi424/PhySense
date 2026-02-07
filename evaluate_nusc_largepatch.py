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
from scripts.interactions_construct_nusc import main as interaction_mean
from scripts.interactions_construct_nusc import parse_arguments as interaction_parse_arguments

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('./data/nusc/attacked_selected/cars_name_dict.pkl','rb') as file:
    cars_name_dict = pickle.load(file)
cars_name_dict = {key.split('/')[-1]: value.split('/')[-1] for key, value in cars_name_dict.items()}
with open('./data/nusc/attacked_selected/peds_name_dict.pkl','rb') as file:
    peds_name_dict = pickle.load(file)
peds_name_dict = {key.split('/')[-1]: value.split('/')[-1] for key, value in peds_name_dict.items()}


from multiprocessing import set_start_method
if __name__ == '__main__':
    set_start_method('spawn')
    # 1. Model perpare

    with open('./weights/lstm_label_df.pkl', 'rb') as file:
        lstm_label_df = pickle.load(file)

    myyolo = yolov3_pytorch.yolov3(device)
    # 2. Read attacked instances


    with open('./data/nusc/attacked_selected/advpath_capatch_attacked_instances_car_3_cornershash_addslap.pkl', 'rb') as f:
        capatch_attacked_instances_car_3 = pickle.load(f)
        capatch_attacked_instances_car_3 = dict(sorted(capatch_attacked_instances_car_3.items()))
    with open('./data/nusc/attacked_selected/advpath_phys_attacked_instances_car_cornershash_addslap.pkl', 'rb') as f:
        phys_attacked_instances_car = pickle.load(f)
        phys_attacked_instances_car = dict(sorted(phys_attacked_instances_car.items()))
    with open('./data/nusc/attacked_selected/advpath_slap_attacked_instances_car_cornershash_addslap.pkl', 'rb') as f:
        slap_attacked_instances_car = pickle.load(f)
        slap_attacked_instances_car = dict(sorted(slap_attacked_instances_car.items()))
    with open('./data/nusc/attacked_selected/advpath_capatch_attacked_instances_ped_3_cornershash_addslap.pkl', 'rb') as f:
        capatch_attacked_instances_ped_3 = pickle.load(f)
        capatch_attacked_instances_ped_3 = dict(sorted(capatch_attacked_instances_ped_3.items()))
    with open('./data/nusc/attacked_selected/advpath_phys_attacked_instances_ped_cornershash_addslap.pkl', 'rb') as f:
        phys_attacked_instances_ped = pickle.load(f)
        phys_attacked_instances_ped = dict(sorted(phys_attacked_instances_ped.items()))
    with open('./data/nusc/attacked_selected/advpath_slap_attacked_instances_ped_cornershash_addslap.pkl', 'rb') as f:
        slap_attacked_instances_ped = pickle.load(f)
        slap_attacked_instances_ped = dict(sorted(slap_attacked_instances_ped.items()))

    with open('./data/nusc/nuscenes_dataset_qd3dt.pkl', 'rb') as f:
        nuscenes_dataset = pickle.load(f)
    all_instances = set(nuscenes_dataset.keys())
    car_instances = {s for s in nuscenes_dataset if "car" in s}
    ped_instances = {s for s in nuscenes_dataset if "ped" in s}

    phys_car_attacked = set()
    for key, value in phys_attacked_instances_car.items():
        if value[0] >= 0.7:
            phys_car_attacked.add(key)
            
    phys_ped_attacked = set()
    for key, value in phys_attacked_instances_ped.items():
        if value[0] >= 0.7:
            phys_ped_attacked.add(key)

    phys_car_benign = car_instances - phys_car_attacked
    phys_ped_benign = ped_instances - phys_ped_attacked

    capatch_car_attacked = set()
    for key, value in capatch_attacked_instances_car_3.items():
        if value[0] >= 0.7:
            capatch_car_attacked.add(key)

    capatch_ped_attacked = set()
    for key, value in capatch_attacked_instances_ped_3.items():
        if value[0] >= 0.7:
            capatch_ped_attacked.add(key)

    capatch_car_benign = car_instances - capatch_car_attacked
    capatch_ped_benign = ped_instances - capatch_ped_attacked

    slap_car_attacked = set()
    for key, value in slap_attacked_instances_car.items():
        if value[0] >= 0.7:
            slap_car_attacked.add(key)

    slap_ped_attacked = set()
    for key, value in slap_attacked_instances_ped.items():
        if value[0] >= 0.7:
            slap_ped_attacked.add(key)

    slap_car_benign = car_instances - slap_car_attacked
    slap_ped_benign = ped_instances - slap_ped_attacked
    
    print('# Obj')
    print(f"Total:  {len(phys_car_attacked)+len(capatch_car_attacked)+len(slap_car_attacked)+len(phys_ped_attacked)+len(capatch_ped_attacked)+len(slap_ped_attacked)+len(phys_car_benign|capatch_car_benign|slap_car_benign|phys_ped_benign|capatch_ped_benign|slap_ped_benign)}")
    print(f"Attacked: {len(phys_car_attacked)+len(phys_ped_attacked)+len(capatch_car_attacked)+len(capatch_ped_attacked)+len(slap_car_attacked)+len(slap_ped_attacked)}")

    # Sanity check against attack file:
    for objname, objseq in capatch_attacked_instances_car_3.items():
        objseq = objseq[1]

        substrings = [item["path"].split('#')[0] for item in objseq]
        is_unique = len(substrings) == len(set(substrings))
        if not is_unique:
            print("Duplicate detected")

    # Make indices against attacked file, organized by frame

    with open('./data/nusc/scene_token_chain.pkl', 'rb') as file:
        scene_token_chain = pickle.load(file)

    attacked_car_capatch = {}
    attacked_ped_capatch = {}
    attacked_car_phys = {}
    attacked_ped_phys = {}
    attacked_car_slap = {}
    attacked_ped_slap = {}

    for key, value in capatch_attacked_instances_car_3.items():
        if value[0] < 0.7:
            continue
        attacked_pics = [item for item in value[1] if item['attacked']]
        scenenum = int(key[5:8])
        token_chain = scene_token_chain[scenenum]
        if scenenum not in attacked_car_capatch:
            attacked_car_capatch[scenenum] = {}
        added_pics = 0
        for i, frame_token in enumerate(token_chain):
            if frame_token not in attacked_car_capatch[scenenum]:
                attacked_car_capatch[scenenum][frame_token] = []
            for pic_dict in attacked_pics:
                if pic_dict['sample_token'] == frame_token:
                    attacked_car_capatch[scenenum][frame_token].append(pic_dict['path_adv'])
                    added_pics += 1
            # attacked_car_capatch[scenenum][i].append(frame_token)
        assert added_pics == len(attacked_pics)

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

    for key, value in slap_attacked_instances_car.items():
        if value[0] < 0.7:
            continue
        attacked_pics = [item for item in value[1] if item['attacked']]
        scenenum = int(key[5:8])
        token_chain = scene_token_chain[scenenum]
        if scenenum not in attacked_car_slap:
            attacked_car_slap[scenenum] = {}
        added_pics = 0
        for i, frame_token in enumerate(token_chain):
            if frame_token not in attacked_car_slap[scenenum]:
                attacked_car_slap[scenenum][frame_token] = []
            for pic_dict in attacked_pics:
                if pic_dict['sample_token'] == frame_token:
                    attacked_car_slap[scenenum][frame_token].append(pic_dict['path_adv'])
                    added_pics += 1
        assert added_pics == len(attacked_pics)

    for key, value in capatch_attacked_instances_ped_3.items():
        if value[0] < 0.7:
            continue
        attacked_pics = [item for item in value[1] if item['attacked']]
        scenenum = int(key[5:8])
        token_chain = scene_token_chain[scenenum]
        if scenenum not in attacked_ped_capatch:
            attacked_ped_capatch[scenenum] = {}
        added_pics = 0
        for i, frame_token in enumerate(token_chain):
            if frame_token not in attacked_ped_capatch[scenenum]:
                attacked_ped_capatch[scenenum][frame_token] = []
            for pic_dict in attacked_pics:
                if pic_dict['sample_token'] == frame_token:
                    attacked_ped_capatch[scenenum][frame_token].append(pic_dict['path_adv'])
                    added_pics += 1
        assert added_pics == len(attacked_pics)

    for key, value in phys_attacked_instances_ped.items():
        if value[0] < 0.7:
            continue
        attacked_pics = [item for item in value[1] if item['attacked']]
        scenenum = int(key[5:8])
        token_chain = scene_token_chain[scenenum]
        if scenenum not in attacked_ped_phys:
            attacked_ped_phys[scenenum] = {}
        added_pics = 0
        for i, frame_token in enumerate(token_chain):
            if frame_token not in attacked_ped_phys[scenenum]:
                attacked_ped_phys[scenenum][frame_token] = []
            for pic_dict in attacked_pics:
                if pic_dict['sample_token'] == frame_token:
                    attacked_ped_phys[scenenum][frame_token].append(pic_dict['path_adv'])
                    added_pics += 1
        assert added_pics == len(attacked_pics)

    for key, value in slap_attacked_instances_ped.items():
        if value[0] < 0.7:
            continue
        attacked_pics = [item for item in value[1] if item['attacked']]
        scenenum = int(key[5:8])
        token_chain = scene_token_chain[scenenum]
        if scenenum not in attacked_ped_slap:
            attacked_ped_slap[scenenum] = {}
        added_pics = 0
        for i, frame_token in enumerate(token_chain):
            if frame_token not in attacked_ped_slap[scenenum]:
                attacked_ped_slap[scenenum][frame_token] = []
            for pic_dict in attacked_pics:
                if pic_dict['sample_token'] == frame_token:
                    attacked_ped_slap[scenenum][frame_token].append(pic_dict['path_adv'])
                    added_pics += 1
        assert added_pics == len(attacked_pics)

    # prepare dataset
    with open('./data/nusc/crf_dataset_full_part_lengths.pkl', 'rb') as file:
        part_lengths = pickle.load(file)
    
    dataset_per_scene = phySense.GraphDataset_addgt('./data/nusc/crf_dataset_full/', 'weights/label_encoder_classes.npy', 10, part_lengths)
    
    coco_to_labels = {1:0, # bicycle
                    5:1, # bus
                    2:2, # car
                    0:3, # pedestrian
                    7:4} # truck
    labels_to_coco = {0:1, # bicycle
                    1:5, # bus
                    2:2, # car
                    3:0, # pedestrian
                    4:7, # truck
                    5:-1}# None

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

    myGuard_benign = phySense.phySense(lstm_model_path='./weights/at_bilstm.pth',
                                                    lstm_label_path='./weights/lstm_label_to_int.pkl',
                                                    file_path='./weights/bayesian_dataset_nusc.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='./weights/xgb_lpb_model.json',
                                                    label_encoder_path='./weights/label_encoder_classes.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path='weights/crf_model_best.pth' ,
                                                    small_labelspace=True, use_precompute=False, trust_frame_num=3)
    
    myGuard_phys = phySense.phySense(lstm_model_path='./weights/at_bilstm.pth',
                                                    lstm_label_path='./weights/lstm_label_to_int.pkl',
                                                    file_path='./weights/bayesian_dataset_nusc.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='./weights/xgb_lpb_model.json',
                                                    label_encoder_path='./weights/label_encoder_classes.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path='weights/crf_model_best.pth' ,
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
        for cur_frame_num in range(frames_len):
            datapoint = dataset_per_scene[cur_frame_num+sum(part_lengths[:scene_num])]
            gt_correct = []
            for idx, gt_label_name in enumerate(datapoint[9]):
                if gt_label_name is not None:
                    for i, name in enumerate(label_names):
                        if name in gt_label_name:
                            break
                    if i == idx:
                        gt_correct.append(True)
                    else:
                        gt_correct.append(False)
                else:
                    gt_correct.append(False)
            datapoint = datapoint[:-1]

            # We need to use actual frame num durning apply attack
            actual_frame_num = -1
            parts = datapoint[8][0].split('/')
            scene_part = parts[3].split('#')
            this_sample_token = scene_part[0]
            for index, frame_token in enumerate(scene_token_chain[scene_num]):
                if frame_token == this_sample_token:
                    actual_frame_num = index
                    break
            assert actual_frame_num != -1

            sample_token_list.append(this_sample_token)

            # 1. Benign
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint],n_states=5)
        
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    filenames[-1].append('./data/nusc/bbox_dataset/b'+j[8:])
            
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
                    if name in img_filename:
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
                
            if scene_num in attacked_car_phys:
                for attacked_car in attacked_car_phys[scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(datapoint_phys[8]):
                        if attacked_car.split('/')[-1] not in ori_filename:
                            continue
                        datapoint_phys[8][pert_index] = './data/nusc/'+attacked_car[8:]
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                        
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
                
            if scene_num in attacked_ped_phys:
                for attacked_ped in attacked_ped_phys[scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(datapoint_phys[8]):
                        if attacked_ped.split('/')[-1] not in ori_filename:
                            continue
                        datapoint_phys[8][pert_index] = './data/nusc/'+attacked_ped[8:]
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                        
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint_phys],n_states=5)
        
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'attacked' not in j:
                        filenames[-1].append('./data/nusc/bbox_dataset/b'+j[8:])
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
                    if name in img_filename:
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

        # 3. performe evaluate at once.
        for i, (benign_result, this_sample_token) in enumerate(zip(benign_results, sample_token_list)):
            output = benign_result[0]
            for index, (perception_result, groundtruth_label) in enumerate(zip(benign_perceptions[i], benign_gt[i])):
                if not benign_gt_correct[i][index]:
                    continue
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
            if (scene_num in attacked_car_phys and len(attacked_car_phys[scene_num][this_sample_token]) > 0) or (scene_num in attacked_ped_phys and len(attacked_ped_phys[scene_num][this_sample_token]) > 0):
                for index, (perception_result, groundtruth_label) in enumerate(zip(phys_perceptions[i], phys_gt[i])):
                    if not benign_gt_correct[i][index]:
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
                    if not benign_gt_correct[i][pert_index]:
                        continue
                    if labels_to_coco[phys_gt[i][pert_index]] not in phys_perceptions[i][pert_index]:
                        total_attacked_phys += 1
                        if labels_to_coco[int(output[pert_index])] not in phys_perceptions[i][pert_index]:
                            total_attacked_detected_phys += 1
                        if int(output[pert_index]) == phys_gt[i][pert_index]:
                            total_attacked_recovered_phys += 1

    myGuard_benign.shutdown()
    myGuard_phys.shutdown()
    del myGuard_benign
    del myGuard_phys
    torch.cuda.empty_cache()
    gc.collect()

    myGuard_capatch = phySense.phySense(lstm_model_path='./weights/at_bilstm.pth',
                                                    lstm_label_path='./weights/lstm_label_to_int.pkl',
                                                    file_path='./weights/bayesian_dataset_nusc.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='./weights/xgb_lpb_model.json',
                                                    label_encoder_path='./weights/label_encoder_classes.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path='weights/crf_model_best.pth' ,
                                                    small_labelspace=True, use_precompute=False, trust_frame_num=3)
    
    myGuard_slap = phySense.phySense(lstm_model_path='./weights/at_bilstm.pth',
                                                    lstm_label_path='./weights/lstm_label_to_int.pkl',
                                                    file_path='./weights/bayesian_dataset_nusc.json',
                                                    num_size_x_bin=20, num_size_y_bin=20, num_size_z_bin=20,
                                                    bayesian_model_path='./weights/xgb_lpb_model.json',
                                                    label_encoder_path='./weights/label_encoder_classes.npy',
                                                    edge_buffer_ratio=(0.2, 0.2), num_regions=8,
                                                    region_size_ratio=(0.3, 0.3), behavior_mask_path='weights/tensor_behavior_mask.pt',
                                                    num_states=5, num_actions=15, num_inter_actions=1,
                                                    beam_size=5, device=device, lstm_label_df=lstm_label_df, crf_save_path='weights/crf_model_best.pth' ,
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
        for cur_frame_num in range(frames_len):
            datapoint = dataset_per_scene[cur_frame_num+sum(part_lengths[:scene_num])]
            gt_correct = []
            for idx, gt_label_name in enumerate(datapoint[9]):
                if gt_label_name is not None:
                    for i, name in enumerate(label_names):
                        if name in gt_label_name:
                            break
                    if i == idx:
                        gt_correct.append(True)
                    else:
                        gt_correct.append(False)
                else:
                    gt_correct.append(False)
            datapoint = datapoint[:-1]

            # We need to use actual frame num durning apply attack
            actual_frame_num = -1
            parts = datapoint[8][0].split('/')
            scene_part = parts[3].split('#')
            this_sample_token = scene_part[0]
            for index, frame_token in enumerate(scene_token_chain[scene_num]):
                if frame_token == this_sample_token:
                    actual_frame_num = index
                    break
            assert actual_frame_num != -1

            sample_token_list.append(this_sample_token)

            # 3. capatch

            datapoint_capatch = copy.deepcopy(datapoint)
            # Substitute attacked image
            pert_indices = []
            change_flag = False
                
            if scene_num in attacked_car_capatch:
                for attacked_car in attacked_car_capatch[scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(datapoint_capatch[8]):
                        if attacked_car.split('/')[-1] not in ori_filename:
                            continue
                        datapoint_capatch[8][pert_index] = './data/nusc/'+attacked_car[8:]
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                        
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
                
            if scene_num in attacked_ped_capatch:
                for attacked_ped in attacked_ped_capatch[scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(datapoint_capatch[8]):
                        if attacked_ped.split('/')[-1] not in ori_filename:
                            continue
                        datapoint_capatch[8][pert_index] = './data/nusc/'+attacked_ped[8:]
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                        
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint_capatch],n_states=5)
        
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'attacked' not in j:
                        filenames[-1].append('./data/nusc/bbox_dataset/b'+j[8:])
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
                    if name in img_filename:
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
            capatch_gt_correct.append(gt_correct)
            capatch_perts.append(pert_indices)
            # 4. Attacked by slap

            datapoint_slap = copy.deepcopy(datapoint)
            # Substitute attacked image
            pert_indices = []
            change_flag = False
                
            if scene_num in attacked_car_slap:
                for attacked_car in attacked_car_slap[scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(datapoint_slap[8]):
                        if cars_name_dict[attacked_car.split('/')[-1]] not in ori_filename:
                            continue
                        datapoint_slap[8][pert_index] = './data/nusc/attacked_selected/slap/' +  os.path.join(*attacked_car.split('/')[-2:])
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                        
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
                
            if scene_num in attacked_ped_slap:
                for attacked_ped in attacked_ped_slap[scene_num][this_sample_token]:
                    for pert_index, ori_filename in enumerate(datapoint_slap[8]):
                        if peds_name_dict[attacked_ped.split('/')[-1]] not in ori_filename:
                            continue
                        datapoint_slap[8][pert_index] = './data/nusc/attacked_selected/slap/' + os.path.join(*attacked_ped.split('/')[-2:])
                        change_flag = True
                        pert_indices.append(pert_index)
                        break
                        
                    
                    if change_flag:
                        change_flag = False
                    else:
                        raise ValueError('Pert Filename not in')
            
            (padded_beh_seqs, batch_edges, batch_edge_masks,
            size_xs, size_ys, size_zs, padded_labels, batch_masks, lengths, interactions, graph_num,
            filenames) = phySense.custom_collate_fn([datapoint_slap],n_states=5)
        
            old_filenames = copy.deepcopy(filenames)
            filenames = []
            for i in old_filenames:
                filenames.append([])
                for j in i:
                    if 'attacked' not in j:
                        filenames[-1].append('./data/nusc/bbox_dataset/b'+j[8:])
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
                    if name in img_filename:
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
            slap_gt_correct.append(gt_correct)
            slap_perts.append(pert_indices)

        # 2. put the sequence into the defense at once
        capatch_results, capatch_runtime = myGuard_capatch.decode_pipeline(capatch_frames)
        slap_results, slap_runtime = myGuard_slap.decode_pipeline(slap_frames)
        times_records.append(capatch_runtime)
        times_records.append(slap_runtime)

        myGuard_capatch.clear_trustworthy_info()
        myGuard_slap.clear_trustworthy_info()

        # 3. performe evaluate at once.
        for i, (capatch_result, this_sample_token) in enumerate(zip(capatch_results, sample_token_list)):
            output = capatch_result[0]
            if (scene_num in attacked_car_capatch and len(attacked_car_capatch[scene_num][this_sample_token]) > 0) or (scene_num in attacked_ped_capatch and len(attacked_ped_capatch[scene_num][this_sample_token]) > 0):
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
            if (scene_num in attacked_car_slap and len(attacked_car_slap[scene_num][this_sample_token]) > 0) or (scene_num in attacked_ped_slap and len(attacked_ped_slap[scene_num][this_sample_token]) > 0):
                for index, (perception_result, groundtruth_label) in enumerate(zip(slap_perceptions[i], slap_gt[i])):
                    if not capatch_gt_correct[i][index]:
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
                    if not capatch_gt_correct[i][pert_index]:
                        continue
                    if labels_to_coco[slap_gt[i][pert_index]] not in slap_perceptions[i][pert_index]:
                        total_attacked_slap += 1
                        if labels_to_coco[int(output[pert_index])] not in slap_perceptions[i][pert_index]:
                            total_attacked_detected_slap += 1
                        if int(output[pert_index]) == slap_gt[i][pert_index]:
                            total_attacked_recovered_slap += 1

    interaction_args = interaction_parse_arguments()
    interaction_args.runtime_test = True
    int_inference_time, graph_construction_time = interaction_mean(interaction_args)
    
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
