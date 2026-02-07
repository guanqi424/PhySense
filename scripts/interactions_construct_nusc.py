import argparse
import gc
import itertools
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from util.interaction_class_inference import inference_two_nodes
import copy

def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph-based dataset processing for interaction inference.")

    parser.add_argument('--device', type=str, default="cuda", help='Device to use for computation ("cuda" or "cpu")')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_scenes', type=int, default=150, help='Number of scenes to process')
    parser.add_argument('--label_space', type=str, nargs='+', default=['bicycle', 'bus', 'car', 'pedestrian', 'truck'], help='Label space')
    parser.add_argument('--threshold_deltaTTC', type=float, default=1, help='Threshold for delta Time to Collision (TTC)')
    parser.add_argument('--threshold_maxTTC', type=float, default=2, help='Threshold for maximum Time to Collision (TTC)')
    parser.add_argument('--threshold_diffVp', type=float, default=1, help='Threshold for difference in velocity parallel')
    parser.add_argument('--threshold_Vp', type=float, default=2, help='Threshold for velocity parallel')
    parser.add_argument('--threshold_Dp', type=float, default=2, help='Threshold for distance parallel')
    parser.add_argument('--threshold_follow_v', type=float, default=2, help='Threshold for follow velocity')
    parser.add_argument('--threshold_dist_parallel', type=float, default=3, help='Threshold for distance parallel')
    parser.add_argument('--threshold_angle_parallel', type=float, default=math.pi * 15 / 180, help='Threshold for angle parallel')
    parser.add_argument('--threshold_angle_overtake', type=float, default=math.pi * 30 / 180, help='Threshold for angle overtake')
    parser.add_argument('--threshold_Vc', type=float, default=7, help='Threshold for velocity crossing')
    parser.add_argument('--threshold_stop', type=float, default=0.1, help='Threshold for stop detection')
    parser.add_argument('--threshold_betweenV', type=float, default=2, help='Threshold for velocity between')
    parser.add_argument('--threshold_EOA', type=float, default=0.3, help='Threshold for end-of-action detection')
    parser.add_argument('--save_dir', type=str, default='./data/nusc/crf_dataset_train/', help='Directory to save the dataset parts')
    parser.add_argument('--save_parts', type=int, default=1, help='Number of parts to save dataset')
    parser.add_argument('--dataset_file', type=str, default='data/nusc/nuscenes_dataset_qd3dt.pkl', help='Dataset file path')
    parser.add_argument('--scene_token_chain', type=str, default='data/nusc/scene_token_chain.pkl', help='')
    parser.add_argument('--frames_meta', type=str, default='data/nusc/frames_meta.json', help='')
    parser.add_argument('--save_interval', type=int, default=1, help='')
    parser.add_argument('--enableVV', action='store_true', help='enable VV feature (default: False)')
    parser.add_argument('--runtime_test', action='store_true', help='Enable runtime test (default: False)')

    return parser.parse_args()

def save_list_in_parts(large_list, file_prefix, parts=5):
    part_size = len(large_list) // parts
    filenames = []
    for i in range(parts):
        start_index = i * part_size
        part = large_list[start_index:] if i == parts - 1 else large_list[start_index:start_index + part_size]
        filename = f'{file_prefix}_part{i + 1}.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(part, file)
        filenames.append(filename)
        del part
        gc.collect()
    return filenames

def save_data_to_parts(data, save_dir, prefix, num_parts=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    part_size = len(data) // num_parts
    file_paths = []
    for i in trange(num_parts):
        part_data = data[i * part_size: (i + 1) * part_size if i < num_parts - 1 else len(data)]
        file_path = os.path.join(save_dir, f'{prefix}_part{i}.pkl')
        file_paths.append(file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(part_data, f)
        del part_data
        gc.collect()
    return file_paths

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    np.random.seed(args.seed)
    random.seed(args.seed)

    prefixes = [f"scene{number:03d}" for number in range(args.num_scenes)]

    crf_full_dataset = []
    full_dataset_len = 0
    counter_con_full = 0

    LabelSpace = args.label_space
    label_mapping = {1:0, 2:1, 3:2, 6:3, 9:4}
    bidirectional_interactions = [0, 3, 8]
    interactions_to_index = {
        0: [(6, 6)],
        1: [(2, 6), (3, 6), (4, 6), (5, 6), (8, 6), (9, 6)],
        2: [(6, 2), (6, 3), (6, 4), (6, 5), (6, 8), (6, 9)],
        3: [(2, 2), (2, 3), (2, 4), (2, 5), (2, 8), (2, 9),
            (3, 2), (3, 3), (3, 4), (3, 5), (3, 8), (3, 9),
            (4, 2), (4, 3), (4, 4), (4, 5), (4, 8), (4, 9),
            (5, 2), (5, 3), (5, 4), (5, 5), (5, 8), (5, 9),
            (8, 2), (8, 3), (8, 4), (8, 5), (8, 8), (8, 9),
            (9, 2), (9, 3), (9, 4), (9, 5), (9, 8), (9, 9)],
        4: [(2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (9, 1)],
        5: [(1, 2), (1, 3), (1, 4), (1, 5), (1, 8), (1, 9)],
        6: [(1, 6)],
        7: [(6, 1)],
        8: [(1, 1)]
    }

    for index, pairs in interactions_to_index.items():
        interactions_to_index[index] = [(label_mapping[x], label_mapping[y]) for x, y in pairs if x in label_mapping and y in label_mapping]

    with open(args.scene_token_chain, 'rb') as file:
        scene_token_chain = pickle.load(file)
    with open(args.dataset_file, 'rb') as f:
        final_dataset_full = pickle.load(f)

    final_dataset_full = {key: value for key, value in final_dataset_full.items() if any(sub in key for sub in LabelSpace)}

    with open(args.frames_meta, 'r') as file:
        frames_meta = json.load(file)['frames']
    frames_meta_dict = {d["token"]: d for d in frames_meta}

    for key, value in final_dataset_full.items():
        for i, frame in enumerate(value):
            my_sample_token = frame['sample_token']
            final_dataset_full[key][i]['timestamp'] = frames_meta_dict[my_sample_token]['timestamp']

    total_inference_time = 0
    total_construction_time = 0
    start_time = time.time()

    for key, seq in final_dataset_full.items():
        for i in range(len(seq)):
            if i == len(seq) - 1:
                if i == 0:
                    acc = np.array([0, 0])
                elif i == 1:
                    acc = (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity'])) / np.array(seq[i]['timestamp'] - seq[i - 1]['timestamp']) if (seq[i]['timestamp'] - seq[i - 1]['timestamp']) != 0 else (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity']))
                else:
                    a1 = (np.array(seq[i - 1]['velocity']) - np.array(seq[i - 2]['velocity'])) / np.array(seq[i - 1]['timestamp'] - seq[i - 2]['timestamp']) if (seq[i - 1]['timestamp'] - seq[i - 2]['timestamp']) != 0 else (np.array(seq[i - 1]['velocity']) - np.array(seq[i - 2]['velocity']))
                    a2 = (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity'])) / np.array(seq[i]['timestamp'] - seq[i - 1]['timestamp']) if (seq[i]['timestamp'] - seq[i - 1]['timestamp']) != 0 else (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity']))
                    acc = 2 * a2 - a1
            else:
                acc = (np.array(seq[i + 1]['velocity']) - np.array(seq[i]['velocity'])) / (seq[i + 1]['timestamp'] - seq[i]['timestamp']) if (seq[i + 1]['timestamp'] - seq[i]['timestamp']) != 0 else (np.array(seq[i + 1]['velocity']) - np.array(seq[i]['velocity']))
            seq[i]['accelerate'] = acc.tolist()
    
    dataset_to_show = None
    for idx, scene in tqdm(enumerate(prefixes), total=args.num_scenes):
        full_instances = {key: value for key, value in final_dataset_full.items() if key.startswith(scene)}

        for framenum in scene_token_chain[idx]:
            my_graph = []
            for name, instance in full_instances.items():
                for instance_frame in instance:
                    if instance_frame['sample_token'] == framenum:
                        my_graph.append((name, instance_frame))
                        break

            if len(my_graph) > 0:
                my_graph_refined = ({'filename': [], 'pos': [], 'size_x': [], 'size_y': [], 'size_z': [], 'velo_x': [], 'velo_y': [], 'acc_x': [], 'acc_y': [], 'beh_seq': [], 'edges': [], 'graph_nums': 0, 'possibleInt': [], 'interactions': []}, [])
                for node in my_graph:
                    my_graph_refined[1].append(node[1]['name'])
                    my_graph_refined[0]['filename'].append(node[1]['path'])
                    my_graph_refined[0]['pos'].append(node[1]['translation'])
                    my_graph_refined[0]['size_x'].append(node[1]['size'][0])
                    my_graph_refined[0]['size_y'].append(node[1]['size'][1])
                    my_graph_refined[0]['size_z'].append(node[1]['size'][2])
                    my_graph_refined[0]['velo_x'].append(node[1]['velocity'][0])
                    my_graph_refined[0]['velo_y'].append(node[1]['velocity'][1])
                    my_graph_refined[0]['acc_x'].append(node[1]['accelerate'][0])
                    my_graph_refined[0]['acc_y'].append(node[1]['accelerate'][1])
                    my_beh_seq = []
                    init_x = full_instances[node[0]][0]['translation'][0]
                    init_y = full_instances[node[0]][0]['translation'][1]
                    for frame in full_instances[node[0]]:
                        my_beh_seq.append([frame['translation'][0] - init_x, frame['translation'][1] - init_y, frame['size'][0], frame['size'][1], frame['size'][2], frame['velocity'][0], frame['velocity'][1], frame['accelerate'][0], frame['accelerate'][1]])
                        if frame['sample_token'] == node[1]['sample_token']:
                            break
                    assert len(my_beh_seq) > 0
                    my_graph_refined[0]['beh_seq'].append(my_beh_seq)

                notFullyConnected = False
                for i in range(len(my_graph_refined[1])):
                    for j in range(len(my_graph_refined[1])):
                        if i == j:
                            continue
                        node1 = {'pos': [my_graph_refined[0]['pos'][i][0], my_graph_refined[0]['pos'][i][1]], 'size': [my_graph_refined[0]['size_x'][i], my_graph_refined[0]['size_y'][i]], 'acc': [my_graph_refined[0]['acc_x'][i], my_graph_refined[0]['acc_y'][i]], 'velocity': [my_graph_refined[0]['velo_x'][i], my_graph_refined[0]['velo_y'][i]]}
                        node2 = {'pos': [my_graph_refined[0]['pos'][j][0], my_graph_refined[0]['pos'][j][1]], 'size': [my_graph_refined[0]['size_x'][j], my_graph_refined[0]['size_y'][j]], 'acc': [my_graph_refined[0]['acc_x'][j], my_graph_refined[0]['acc_y'][j]], 'velocity': [my_graph_refined[0]['velo_x'][j], my_graph_refined[0]['velo_y'][j]]}
                        isConnected, possibleInteractions, inference_time = inference_two_nodes(
                            node1, node2, 
                            threshold_deltaTTC=args.threshold_deltaTTC,
                            threshold_maxTTC=args.threshold_maxTTC,
                            radius=2, 
                            threshold_diffVp=args.threshold_diffVp,
                            threshold_Vp=args.threshold_Vp,
                            threshold_Dp=args.threshold_Dp,
                            threshold_follow_v=args.threshold_follow_v,
                            threshold_dist_parallel=args.threshold_dist_parallel,
                            threshold_angle_parallel=args.threshold_angle_parallel,
                            threshold_angle_overtake=args.threshold_angle_overtake,
                            threshold_Vc=args.threshold_Vc,
                            threshold_stop=args.threshold_stop,
                            threshold_betweenV=args.threshold_betweenV,
                            thresholdEOA=args.threshold_EOA,
                            disableVV=not args.enableVV)
                        total_inference_time += inference_time
                        if isConnected:
                            my_graph_refined[0]['edges'].append((i, j))
                            my_graph_refined[0]['possibleInt'].append(possibleInteractions)
                        else:
                            notFullyConnected = True
                if notFullyConnected:
                    counter_con_full += 1

                node_pairs = my_graph_refined[0]['edges']

                if len(node_pairs) > 4:
                    continue

                interaction_types = my_graph_refined[0]['possibleInt']
                interaction_map = {pair: interactions for pair, interactions in zip(node_pairs, interaction_types)}

                for pair in list(interaction_map.keys()):
                    reverse_pair = (pair[1], pair[0])
                    if pair < reverse_pair and reverse_pair in interaction_map:
                        common_bidirectional = set(interaction_map[pair]) & set(interaction_map[reverse_pair]) & set(bidirectional_interactions)
                        for interaction in common_bidirectional:
                            interaction_map[pair].remove(interaction)
                            if not interaction_map[pair]:
                                del interaction_map[pair]

                my_graph_refined[0]['edges'] = list(interaction_map.keys())
                my_graph_refined[0]['possibleInt'] = list(interaction_map.values())

                for interaction_combination in itertools.product(*my_graph_refined[0]['possibleInt']):
                    my_graph_refined[0]['graph_nums'] += 1
                    edge_interaction = zip(my_graph_refined[0]['edges'], interaction_combination)
                    my_interaction_matrix = torch.zeros(len(my_graph_refined[1]), len(my_graph_refined[1]), len(LabelSpace), len(LabelSpace), 1)
                    for edges, interaction in edge_interaction:
                        for interacive_label in interactions_to_index[interaction]:
                            my_interaction_matrix[edges[0], edges[1], interacive_label[0], interacive_label[1], 0] = 1
                    my_graph_refined[0]['interactions'].append(my_interaction_matrix.numpy().tolist())

                if my_graph_refined[0]['graph_nums'] == 0:
                    my_graph_refined[0]['graph_nums'] = 1
                    my_interaction_matrix = torch.zeros(len(my_graph_refined[1]), len(my_graph_refined[1]), len(LabelSpace), len(LabelSpace), 1)
                    my_graph_refined[0]['interactions'].append(my_interaction_matrix.numpy().tolist())

                crf_full_dataset.append(my_graph_refined)

        if (idx + 1) % args.save_interval == 0:
            end_time = time.time()
            total_construction_time += end_time - start_time
            if not args.runtime_test:
                save_list_in_parts(crf_full_dataset, os.path.join(args.save_dir, f'scenes_{idx+1}'), args.save_parts)
            full_dataset_len += len(crf_full_dataset)

            if idx == 123: # scene to show
                dataset_to_show = copy.deepcopy(crf_full_dataset)
            
            del crf_full_dataset
            crf_full_dataset = []
            gc.collect()
            start_time = time.time()

    if not args.runtime_test:
        print(f"Interaction Inference time: {(total_inference_time) / (full_dataset_len)}s per graph")
        print(f"Graph construction time: {(total_construction_time - total_inference_time) / (full_dataset_len)}s per graph")
        print()
        print("Sample interaction pairs:")
        int_mapping = {
            0: 'Pedestrian-Pedestrian',
            1: 'Vehicle-Pedestrian',
            2: 'Pedestrian-Vehicle',
            3: 'Vehicle-Vehicle',
            4: 'Vehicle-Bicycle',
            5: 'Bicycle-Vehicle',
            6: 'Bicycle-Pedestrian',
            7: 'Pedestrian-Bicycle',
            8: 'Bicycle-Bicycle'
        }
        for graph in dataset_to_show[25:]:
            if len(graph[0]['edges']) != 0:
                print('Node', graph[0]['edges'][0][0], 'to', 'Node', graph[0]['edges'][0][1])
                print("Groundtruth Labels:", graph[1][graph[0]['edges'][0][0]], graph[1][graph[0]['edges'][0][1]])
                print('Potential object label pair inferred from interactions:')
                for interaction in graph[0]['possibleInt'][0]:
                    print(int_mapping[interaction])
                break
        return -1, -1
    else:
        return (total_inference_time) / (full_dataset_len), (total_construction_time - total_inference_time) / (full_dataset_len)

if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
