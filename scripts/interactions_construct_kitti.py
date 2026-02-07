import gc
import itertools
import math
import os
import pickle
import random
import time
import copy

import numpy as np
import torch
from tqdm import tqdm
from util.interaction_class_inference import inference_two_nodes


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

def main(runtime_test=False, save_interval=1, save_dir='./data/kitti/crf_dataset_train/', save_parts=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(42)
    random.seed(42)

    crf_full_dataset = []
    full_dataset_len = 0
    counter_con_gt = 0

    LabelSpace = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']
    label_mapping = {1:0, 2:1, 3:2, 6:3, 9:4}
    bidirectional_interactions = [0, 3, 8]
    interactions_to_index = {0: [(6, 6)],
                            1: [(2, 6), (3, 6), (4, 6), (5, 6), (8, 6), (9, 6)],
                            2: [(6, 2), (6, 3), (6, 4), (6, 5), (6, 8), (6, 9)],
                            3: [(2, 2), (2, 3), (2, 4), (2, 5), (2, 8), (2, 9),
                                (3, 2), (3, 3), (3, 4), (3, 5), (3, 8), (3, 9),
                                (4, 2), (4, 3), (4, 4), (4, 5), (4, 8), (4, 9),
                                (5, 2), (5, 3), (5, 4), (5, 5), (5, 8), (5, 9),
                                (8, 2), (8, 3), (8, 4), (8, 5), (8, 8), (8, 9),
                                (9, 2), (9, 3), (9, 4), (9, 5), (9, 8), (9, 9)],
                            4: [(2, 1), (3, 1), (4, 1), (5, 1), (8, 1), (9, 1)],
                            5: [(1, 2), (1, 3), (1, 4), (1, 4), (1, 8), (1, 9)],
                            6: [(1, 6)],
                            7: [(6, 1)],
                            8: [(1, 1)]}

    for index, pairs in interactions_to_index.items():
        interactions_to_index[index] = [(label_mapping[x], label_mapping[y]) for x, y in pairs if x in label_mapping and y in label_mapping]

    with open('./data/kitti/final_kitti_dataset_subtrain_aligned.pkl', 'rb') as f:
        final_dataset_gt = pickle.load(f)
    # modify label space
    final_dataset_gt = {key: value for key, value in final_dataset_gt.items() if any(sub in key for sub in LabelSpace)}
    total_inference_time = 0
    total_construction_time = 0
    start_time = time.time()

    # add velocity and acc. for datasets
    for key, seq in final_dataset_gt.items():
        for i in range(len(seq)):
            if i == 0:
                if len(seq) == 1:
                    velocity = np.array([0, 0])
                elif len(seq) == 2:
                    if seq[1]['timestamp'] - seq[0]['timestamp'] == 0:
                        velocity = np.array(seq[1]['translation'][:2]) - np.array(seq[0]['translation'][:2])
                    else:
                        velocity = (np.array(seq[1]['translation'][:2]) - np.array(seq[0]['translation'][:2])) / (
                                seq[1]['timestamp'] - seq[0]['timestamp'])
                else:
                    if (seq[1]['timestamp'] - seq[0]['timestamp']) != 0:
                        v1 = (np.array(seq[1]['translation'][:2]) - np.array(seq[0]['translation'][:2])) / (
                                seq[1]['timestamp'] - seq[0]['timestamp'])
                    else:
                        v1 = np.array(seq[1]['translation'][:2]) - np.array(seq[0]['translation'][:2])
                    if (seq[2]['timestamp'] - seq[1]['timestamp']) != 0:
                        v2 = (np.array(seq[2]['translation'][:2]) - np.array(seq[1]['translation'][:2])) / (
                                seq[2]['timestamp'] - seq[1]['timestamp'])
                    else:
                        v2 = np.array(seq[2]['translation'][:2]) - np.array(seq[1]['translation'][:2])
                    velocity = 2 * v1 - v2
            else:
                if (seq[i]['timestamp'] - seq[i - 1]['timestamp']) == 0:
                    velocity = np.array(seq[i]['translation'][:2]) - np.array(seq[i - 1]['translation'][:2])
                else:
                    velocity = (np.array(seq[i]['translation'][:2]) - np.array(seq[i - 1]['translation'][:2])) / (
                            seq[i]['timestamp'] - seq[i - 1]['timestamp'])
            seq[i]['velocity'] = velocity.tolist()

    for key, seq in final_dataset_gt.items():
        for i in range(len(seq)):
            if i == len(seq) - 1:
                if i == 0:
                    acc = np.array([0, 0])
                elif i == 1:
                    if (seq[i]['timestamp'] - seq[i - 1]['timestamp']) != 0:
                        acc = (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity'])) / np.array(
                            seq[i]['timestamp'] - seq[i - 1]['timestamp'])
                    else:
                        acc = (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity']))
                else:
                    if (seq[i - 1]['timestamp'] - seq[i - 2]['timestamp']) != 0:
                        a1 = (np.array(seq[i - 1]['velocity']) - np.array(seq[i - 2]['velocity'])) / np.array(
                            seq[i - 1]['timestamp'] - seq[i - 2]['timestamp'])
                    else:
                        a1 = (np.array(seq[i - 1]['velocity']) - np.array(seq[i - 2]['velocity']))
                    if (seq[i]['timestamp'] - seq[i - 1]['timestamp']) != 0:
                        a2 = (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity'])) / np.array(
                            seq[i]['timestamp'] - seq[i - 1]['timestamp'])
                    else:
                        a2 = (np.array(seq[i]['velocity']) - np.array(seq[i - 1]['velocity']))
                    acc = 2 * a2 - a1
            else:
                if (seq[i + 1]['timestamp'] - seq[i]['timestamp']) != 0:
                    acc = (np.array(seq[i + 1]['velocity']) - np.array(seq[i]['velocity'])) / (
                            seq[i + 1]['timestamp'] - seq[i]['timestamp'])
                else:
                    acc = (np.array(seq[i + 1]['velocity']) - np.array(seq[i]['velocity']))
            seq[i]['accelerate'] = acc.tolist()

    prefixes = ['0001', '0004', '0011', '0012', '0013', '0014', '0015', '0018']
    dataset_to_show = None
    num_scenes = len(prefixes)
    for idx, scene in tqdm(enumerate(prefixes), total=len(prefixes)):
        start_time = time.time()
        full_instances = {key: value for key, value in final_dataset_gt.items() if key.startswith(scene)}
        timestamp_list = []
        for key, value in full_instances.items():
            for frame_ins in value:
                if frame_ins['timestamp'] not in timestamp_list:
                    timestamp_list.append(frame_ins['timestamp'])
        timestamp_list.sort()
        for framenum in timestamp_list:
            my_graph = []
            for name, instance in full_instances.items():
                for instance_frame in instance:
                    if instance_frame['timestamp'] == framenum:
                        # this instance (for this frame) should be a node of the graph
                        my_graph.append((name, instance_frame))
                        break

            if len(my_graph) > 0:
                my_graph_refined = ({'filename': [],
                                    'pos': [],
                                    'size_x': [],
                                    'size_y': [],
                                    'size_z': [],
                                    'velo_x': [],
                                    'velo_y': [],
                                    'acc_x': [],
                                    'acc_y': [],
                                    'beh_seq': [],
                                    'edges': [],
                                    'graph_nums': 0,
                                    'possibleInt': [],
                                    'interactions': []}, [], [])
                for node in my_graph:
                    my_graph_refined[1].append(node[1]['name'])
                    if 'gt_aligned' in node[1]:
                        assert node[1]['gt_aligned']
                    my_graph_refined[2].append(node[1]['gt_aligned'] if 'gt_aligned' in node[1] else False)
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
                        my_beh_seq.append([frame['translation'][0] - init_x, frame['translation'][1] - init_y,
                                        frame['size'][0], frame['size'][1], frame['size'][2],
                                        frame['velocity'][0], frame['velocity'][1],
                                        frame['accelerate'][0], frame['accelerate'][1]])
                        if frame['timestamp'] == node[1]['timestamp']:
                            break
                    assert len(my_beh_seq) > 0
                    my_graph_refined[0]['beh_seq'].append(my_beh_seq)
                # Add edges to graph
                notFullyConnected = False
                for i in range(len(my_graph_refined[1])):
                    for j in range(len(my_graph_refined[1])):
                        if i == j:
                            continue
                        node1 = {'pos': [my_graph_refined[0]['pos'][i][0], my_graph_refined[0]['pos'][i][1]],
                                'size': [my_graph_refined[0]['size_x'][i], my_graph_refined[0]['size_y'][i]],
                                'acc': [my_graph_refined[0]['acc_x'][i], my_graph_refined[0]['acc_y'][i]],
                                'velocity': [my_graph_refined[0]['velo_x'][i], my_graph_refined[0]['velo_y'][i]]}
                        node2 = {'pos': [my_graph_refined[0]['pos'][j][0], my_graph_refined[0]['pos'][j][1]],
                                'size': [my_graph_refined[0]['size_x'][j], my_graph_refined[0]['size_y'][j]],
                                'acc': [my_graph_refined[0]['acc_x'][j], my_graph_refined[0]['acc_y'][j]],
                                'velocity': [my_graph_refined[0]['velo_x'][j], my_graph_refined[0]['velo_y'][j]]}
                        isConnected, possibleInteractions, inference_time = inference_two_nodes(node1, node2, threshold_deltaTTC=1,
                                                                                                threshold_maxTTC=2, radius=2,
                                                                                                threshold_diffVp=1,
                                                                                                threshold_Vp=2, threshold_Dp=2,
                                                                                                threshold_follow_v=2,
                                                                                                threshold_dist_parallel=3,
                                                                                                threshold_angle_parallel=math.pi * 15 / 180,
                                                                                                threshold_angle_overtake=math.pi * 30 / 180,
                                                                                                threshold_Vc=7, threshold_stop=0.1,
                                                                                                threshold_betweenV=2, thresholdEOA=0.3)
                        total_inference_time += inference_time
                        if isConnected:
                            my_graph_refined[0]['edges'].append((i, j))
                            my_graph_refined[0]['possibleInt'].append(possibleInteractions)
                        else:
                            notFullyConnected = True
                if notFullyConnected:
                    counter_con_gt += 1

                node_pairs = my_graph_refined[0]['edges']

                if len(node_pairs) > 4:
                    continue
                
                interaction_types = my_graph_refined[0]['possibleInt']

                interaction_map = {pair: interactions for pair, interactions in zip(node_pairs, interaction_types)}

                for pair in list(interaction_map.keys()):
                    reverse_pair = (pair[1], pair[0])
                    if pair < reverse_pair and reverse_pair in interaction_map:
                        common_bidirectional = set(interaction_map[pair]) & set(interaction_map[reverse_pair]) & set(
                            bidirectional_interactions)
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

        if (idx + 1) % save_interval == 0:
            end_time = time.time()
            total_construction_time += end_time - start_time
            if idx == 1:
                dataset_to_show = copy.deepcopy(crf_full_dataset)

            if not runtime_test:
                save_list_in_parts(crf_full_dataset, os.path.join(save_dir, f'scenes_{idx+1}'), save_parts)
            full_dataset_len += len(crf_full_dataset)
            if idx != num_scenes - 1:
                del crf_full_dataset
                crf_full_dataset = []
                gc.collect()
            start_time = time.time()

    if not runtime_test:
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
        for graph in dataset_to_show[138:]:
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


if __name__ == '__main__':
    os.makedirs('./data/kitti/crf_dataset_train/', exist_ok=True)
    main()
