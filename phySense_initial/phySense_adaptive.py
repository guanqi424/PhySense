import collections
import hashlib
import os
import pickle
import time
from itertools import chain

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .phySenseLSTM.ATBiLSTM import ATBiLSTM
from .phySenseCRF import phySenseCRF
from .bayesianUnary.bayesianUnary import BayesianUnary
from .phySenseCRF.torch_random_fields.constants import Inference, Learning
import torch.nn.functional as F


coco_to_labels = {1:0, # bicycle
                  5:1, # bus
                  2:2, # car
                  0:3, # pedestrian
                  7:4} # truck


def calculate_distance(center1, center2):
    return torch.norm(center1 - center2)

def calculate_iou(box1, box2):
    # box1 and box2 should be in the format [x, y, Size_x, Size_y, Size_z]
    x1, y1, Size_x1, Size_y1, Size_z1 = box1
    x2, y2, Size_x2, Size_y2, Size_z2 = box2
    
    x_min1, y_min1, z_min1 = x1 - Size_x1 / 2, y1 - Size_y1 / 2, -Size_z1 / 2
    x_max1, y_max1, z_max1 = x1 + Size_x1 / 2, y1 + Size_y1 / 2, Size_z1 / 2
    x_min2, y_min2, z_min2 = x2 - Size_x2 / 2, y2 - Size_y2 / 2, -Size_z2 / 2
    x_max2, y_max2, z_max2 = x2 + Size_x2 / 2, y2 + Size_y2 / 2, Size_z2 / 2

    x_min_inter = torch.max(torch.tensor([x_min1, x_min2]))
    y_min_inter = torch.max(torch.tensor([y_min1, y_min2]))
    z_min_inter = torch.max(torch.tensor([z_min1, z_min2]))
    x_max_inter = torch.min(torch.tensor([x_max1, x_max2]))
    y_max_inter = torch.min(torch.tensor([y_max1, y_max2]))
    z_max_inter = torch.min(torch.tensor([z_max1, z_max2]))

    inter_volume = torch.max(torch.tensor(0.0), x_max_inter - x_min_inter) * \
                   torch.max(torch.tensor(0.0), y_max_inter - y_min_inter) * \
                   torch.max(torch.tensor(0.0), z_max_inter - z_min_inter)
    
    volume1 = (x_max1 - x_min1) * (y_max1 - y_min1) * (z_max1 - z_min1)
    volume2 = (x_max2 - x_min2) * (y_max2 - y_min2) * (z_max2 - z_min2)

    union_volume = volume1 + volume2 - inter_volume

    return inter_volume / union_volume if union_volume != 0 else torch.tensor(0.0)


def is_existing_object(new_bbox, existing_objects, distance_threshold, iou_threshold):
    new_center = new_bbox[:2]
    for idx, obj in enumerate(existing_objects):
        obj_center = obj[0][:2]
        if calculate_distance(new_center, obj_center) < distance_threshold:
            if calculate_iou(new_bbox, obj[0]) > iou_threshold:
                return idx, obj
    return None, None

coco_to_labels = {1:0, # bicycle
                  5:1, # bus
                  2:2, # car
                  0:3, # pedestrian
                  7:4} # truck
class phySense(torch.nn.Module):
    def __init__(self, *,
                 # For transformer
                 # num_emb, emb_dim, nhead, num_layers, dim_feedforward, max_seq_length, num_classes_bst, feature_dim,
                 # For LSTM+Attn
                 lstm_model_path, lstm_label_path, lstm_label_df,
                 # For bayesian prob model
                 file_path, num_size_x_bin, num_size_y_bin, num_size_z_bin, bayesian_model_path, label_encoder_path,
                 edge_buffer_ratio, num_regions, region_size_ratio, behavior_mask_path,
                 # For crf setting
                 num_states, num_actions, num_inter_actions, beam_size,
                 device,
                 use_precompute=False,
                 small_labelspace=False, trust_frame_num=3) -> None:
        super().__init__()
        self.behavior_masks = torch.load(behavior_mask_path).to(device)
        self.interaction_masks = torch.ones((num_states, num_states), dtype=torch.bool)
        if not small_labelspace:
            self.interaction_masks[0, :] = False
            self.interaction_masks[7, :] = False
            self.interaction_masks[:, 0] = False
            self.interaction_masks[:, 7] = False
        self.interaction_masks = self.interaction_masks.unsqueeze(-1)
        self.lstm = ATBiLSTM(9, 512, 3, 31)
        self.lstm.load_state_dict(torch.load(lstm_model_path, map_location=device))

        with open(lstm_label_path, 'rb') as f:
            self.lstm_label_to_int = pickle.load(f)
        self.lstm_label_df = lstm_label_df
        self.lstm_label_dict = {}
        for key, value in self.lstm_label_to_int.items():
            for i, label in enumerate(self.lstm_label_df.index):
                if label in key:
                    for j, action in enumerate(self.lstm_label_df.columns):
                        if action in key:
                            self.lstm_label_dict[value] = (i, j)

        self.source_indices = torch.tensor(list(self.lstm_label_dict.keys())).to(device)
        self.target_indices = torch.tensor(list(self.lstm_label_dict.values())).to(device)
        self.state_indices, self.action_indices = torch.tensor(list(zip(*self.lstm_label_dict.values()))).to(device)

        self.crf = phySenseCRF.phySenseCRF(
            num_states=num_states,
            num_actions=num_actions,
            num_inter_actions=num_inter_actions,
            beam_size=beam_size,
            learning=Learning.PERCEPTRON,
            inference=Inference.BELIEF_PROPAGATION,
        )
        self.bayesianUnary = BayesianUnary(file_path, num_size_x_bin, num_size_y_bin, num_size_z_bin,
                                           bayesian_model_path, label_encoder_path,
                                           edge_buffer_ratio, num_regions, region_size_ratio)
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_inter_actions = num_inter_actions
        self.bayesianUnary.create_bins()
        self.use_precompute = use_precompute
        self.trust_frame_num = trust_frame_num
        self.trustworthy_white_list = []
        self.trustworthy_monitor_list = []
        self.trustworthy_monitor_list_labels = []

    def clear_trustworthy_info(self):
        self.trustworthy_white_list = []
        self.trustworthy_monitor_list = []
        self.trustworthy_monitor_list_labels = []

    def forward(self, filename_list, size_x, size_y, size_z,  # [batch_size, num_nodes_max]
                behaviorSeq, lengths,
                # [batch_size, num_nodes_max, num_length_max, feature_dim], [batch_size, num_nodes_max]
                masks,  # [batch_size, num_nodes_max]
                bin_edges,  # [batch_size, n_edges, 2]
                interactions,  # [graph_num, n_bsz, n_nodes, n_nodes, n_states, n_states, n_inter_actions]
                graph_num,
                bin_masks,  # [batch_size, n_edges]
                targets,
                device,
                perception_results=None,  #for trustworthy verification: object has not been detected attack for trust_frame_num should be trustworthy
                trustworthy_iou_threshold=0.5):
        filename_list_flattened = [item for sublist in filename_list for item in sublist]
        images = [Image.open(path).convert('L') for path in filename_list_flattened]
        total_start_time = time.time()
        filtered_size_x = size_x[masks]
        filtered_size_y = size_y[masks]
        filtered_size_z = size_z[masks]
        if self.use_precompute:
            hasher = hashlib.sha256()
            hasher.update(str(filename_list).encode('utf-8'))
            unaries_hash = hasher.hexdigest()
            if os.path.isfile('./lbp_precompute/' + unaries_hash + '.pkl'):
                with open('./lbp_precompute/' + unaries_hash + '.pkl', 'rb') as file:
                    unaries = pickle.load(file)
            else:
                unaries = self.bayesianUnary.inference(filtered_size_x.flatten(), filtered_size_y.flatten(),
                                                       filtered_size_z.flatten(),
                                                       [item for sublist in filename_list for item in sublist])
                with open('./lbp_precompute/' + unaries_hash + '.pkl', 'wb') as file:
                    pickle.dump(unaries, file)
        else:
            unaries = self.bayesianUnary.inference(filtered_size_x.flatten(), filtered_size_y.flatten(),
                                                   filtered_size_z.flatten(),
                                                   images)
        unaries_torch = [torch.from_numpy(df.values) for df in unaries]

        # Find if there's any trustworthy items, make their unaries biggest
        if self.trust_frame_num > 0:
            trusted_objs = []
            trusted_idxs = []

            lengths_0 = lengths[0]
            behaviorSeq_0 = behaviorSeq[0]

            for i, length in enumerate(lengths_0):
                cur_box = behaviorSeq_0[i][length-1][:5]
                trust_idx, trust_obj = is_existing_object(cur_box, self.trustworthy_white_list, torch.min(cur_box[2:]), trustworthy_iou_threshold)
                if trust_idx is not None:
                    trusted_objs.append((i, trust_obj[1]))
                    trusted_idxs.append(trust_idx)
            self.trustworthy_white_list = [self.trustworthy_white_list[i] for i in trusted_idxs]
            for matrix in unaries_torch:
                for trusted_obj in trusted_objs:
                    matrix[trusted_obj[0]].zero_()
                    matrix[trusted_obj[0], trusted_obj[1]] = 1

        log_sum = torch.zeros_like(unaries_torch[0])
        for matrix in unaries_torch:
            log_sum += torch.log(matrix + 1e-10)  # [batch_size*num_nodes_max, num_class]

        batch_size, num_nodes_max = size_x.shape
        unaries_result_shape = (batch_size, num_nodes_max, self.num_states)

        log_sum_expanded = torch.zeros(unaries_result_shape).to(device)

        num_valid_nodes_per_batch = masks.sum(dim=1)

        log_sum_idx = 0
        for i in range(batch_size):
            num_valid_nodes = num_valid_nodes_per_batch[i].item()
            if num_valid_nodes > 0:
                log_sum_expanded[i, :num_valid_nodes] = log_sum[log_sum_idx:log_sum_idx + num_valid_nodes]
                log_sum_idx += num_valid_nodes

        # LSTM input: [batch_size*num_nodes_max, sequence_length, feature_dim], [batch_size]
        # output: [batch_size*num_nodes_max, num_actions]
        filtered_behaviorSeq = behaviorSeq[masks]
        filtered_lengths = lengths[masks]
        outputLSTM = self.lstm(filtered_behaviorSeq, filtered_lengths)
        outputLSTM = outputLSTM.detach()
        behavior_prob = F.softmax(outputLSTM, dim=1)
        behavior_result_shape = (batch_size, num_nodes_max, len(self.lstm_label_to_int))

        behavior_prob_expanded = torch.zeros(behavior_result_shape).to(device)

        behavior_prob_idx = 0
        for i in range(batch_size):
            num_valid_nodes = num_valid_nodes_per_batch[i].item()
            if num_valid_nodes > 0:
                behavior_prob_expanded[i, :num_valid_nodes] = behavior_prob[
                                                              behavior_prob_idx:behavior_prob_idx + num_valid_nodes]
                behavior_prob_idx += num_valid_nodes

        # expand behavior label space to interaction label-action label-space

        behavior_prob_expanded_final = torch.zeros((batch_size, num_nodes_max, self.num_states, self.num_actions)).to(
            device)
        
        batch_indices = torch.arange(batch_size).view(-1, 1, 1, 1)
        node_indices = torch.arange(num_nodes_max).view(1, -1, 1, 1)
        state_indices = self.state_indices.view(1, 1, -1, 1).to(device)
        action_indices = self.action_indices.view(1, 1, -1).to(device)

        behavior_prob_expanded = behavior_prob_expanded.unsqueeze(3)
        behavior_prob_expanded_final[
            batch_indices, node_indices, state_indices, action_indices] = behavior_prob_expanded

        if targets is not None:
            loss_total = None
            for i in range(graph_num):
                loss = self.crf(
                    unaries=log_sum_expanded,
                    behaviors=behavior_prob_expanded_final,
                    masks=masks,
                    interactions=interactions[i],
                    behavior_masks=self.behavior_masks,
                    interaction_masks=self.interaction_masks,
                    binary_edges=bin_edges,
                    binary_masks=bin_masks,
                    targets=targets,
                    device=device
                )
                if loss_total is None:
                    loss_total = loss
                else:
                    loss_total += loss
            return loss_total / graph_num
        else:
            results_idx = []
            results_energy = []
            for i in range(graph_num):
                result = self.crf(
                    unaries=log_sum_expanded,
                    behaviors=behavior_prob_expanded_final,
                    masks=masks,
                    interactions=interactions[i],
                    behavior_masks=self.behavior_masks,
                    interaction_masks=self.interaction_masks,
                    binary_edges=bin_edges,
                    binary_masks=bin_masks,
                    targets=targets,
                    device=device
                )
                results_idx.append(result[1])
                results_energy.append(result[2])
            scores_tensor = torch.tensor(results_energy)
            idx_tensor = torch.stack(results_idx)

            _, max_indices = torch.max(scores_tensor, dim=0)
            selected_idx = idx_tensor[max_indices, torch.arange(batch_size), :]

            if perception_results is not None:
                assert size_x.shape[0] == 1
                perception_results = [
                    coco_to_labels[result[0]] if result and result[0] in coco_to_labels else -1
                    for result in perception_results
                ]
            
            if self.trust_frame_num > 0 and perception_results is not None:
                # 1. update watchlist
                # Iterative through object list. if not appeared in old monitor list, add to new list. else if appeared, if the result is consistent, add frame cnt to new list.
                this_monitor_list = []
                this_monitor_list_labels = []
                if len(self.trustworthy_monitor_list) == 0:
                    for i, length in enumerate(lengths[0]):
                        if selected_idx[0][i] != perception_results[i]:
                            continue

                        cur_box = behaviorSeq[0][i][length-1][:5]
                        this_monitor_list.append((cur_box.clone().detach(), 1))
                        this_monitor_list_labels.append(selected_idx[0][i].clone().detach())
                else:
                    for i, length in enumerate(lengths[0]):
                        if selected_idx[0][i] != perception_results[i]:
                            continue

                        cur_box = behaviorSeq[0][i][length-1][:5]

                        trust_idx, trust_obj = is_existing_object(cur_box, self.trustworthy_monitor_list, torch.min(cur_box[2:]), trustworthy_iou_threshold)
                        
                        if trust_idx is not None and self.trustworthy_monitor_list_labels[trust_idx] == selected_idx[0][i]:
                            this_monitor_list.append((cur_box.clone().detach(), trust_obj[1]+1))
                            this_monitor_list_labels.append(selected_idx[0][i])
                        else:
                            this_monitor_list.append((cur_box.clone().detach(), 1))
                            this_monitor_list_labels.append(selected_idx[0][i].clone().detach())
                            
                self.trustworthy_monitor_list = this_monitor_list
                self.trustworthy_monitor_list_labels = this_monitor_list_labels
                                
                # 2. iterative through watchlist, if reach trustworthy frame cnt, move to whitelist
                self.trustworthy_white_list.extend(
                    (monitor[0], label)
                    for monitor, label in zip(self.trustworthy_monitor_list, self.trustworthy_monitor_list_labels)
                    if monitor[1] >= self.trust_frame_num
                )
            total_end_time = time.time()
            return None, selected_idx, total_end_time - total_start_time, unaries_torch[-1].detach(), behavior_prob_expanded_final.detach()

    def decode(self, filenames, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks, batch_edges,
               interactions, graph_num, batch_edge_masks, device, perception_results=None, trustworthy_iou_threshold=0.5):
        self.eval()
        with torch.no_grad():
            return self(filenames, size_xs, size_ys, size_zs, padded_beh_seqs, lengths, batch_masks, batch_edges,
                        interactions, graph_num, batch_edge_masks, None, device, perception_results, trustworthy_iou_threshold)[1:]
    
    def clear_white_list(self):
        self.trustworthy_white_list = []
        self.trustworthy_monitor_list = []


class GraphDataset(Dataset):
    def __init__(self, data_dir, label_encoder_path, cache_max=5, part_len=None):
        self.data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl') and len(f) <= 19], key=lambda path: int(path.split('_')[-2]))
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)
        
        self.part_lengths = []
        if part_len is None:
            for file_path in tqdm(self.data_files):
                with open(file_path, 'rb') as f:
                    data_list = pickle.load(f)
                    self.part_lengths.append(len(data_list))
        else:
            self.part_lengths = part_len
            
        self.index_ranges = np.cumsum([0] + self.part_lengths)
        
        self.cached_file_indice = collections.deque(maxlen=cache_max)
        self.cached_data_dict = collections.deque(maxlen=cache_max)

    def __len__(self):
        return self.index_ranges[-1]

    def __getitem__(self, idx):
        file_index = np.searchsorted(self.index_ranges, idx + 1) - 1
        part_index = idx - self.index_ranges[file_index]
        
        if file_index not in self.cached_file_indice:
            with open(self.data_files[file_index], 'rb') as f:
                self.cached_data_dict.append(pickle.load(f))
            self.cached_file_indice.append(file_index)
            
        data_point, labels = self.cached_data_dict[self.cached_file_indice.index(file_index)][part_index]
        
        beh_seq = data_point['beh_seq']
        interactions = data_point['interactions']
        graph_nums = data_point['graph_nums']
        filenames = data_point['filename']
        size_x = torch.tensor(data_point['size_x'], dtype=torch.float)
        size_y = torch.tensor(data_point['size_y'], dtype=torch.float)
        size_z = torch.tensor(data_point['size_z'], dtype=torch.float)
        edges = torch.tensor(data_point['edges'], dtype=torch.long)
        
        encoded_labels = torch.tensor(self.label_encoder.transform(labels), dtype=torch.long)

        return beh_seq, edges, size_x, size_y, size_z, interactions, graph_nums, encoded_labels, filenames


def custom_collate_fn(batch, n_states=10, n_inter_actions=1):
    beh_seqs_list, edges_list, size_xs_list, size_ys_list, size_zs_list, interactions_list, graph_nums, labels_list, filenames_list = zip(
        *batch)

    max_seq_length = max(max(len(seq) for seq in beh_seqs) for beh_seqs in beh_seqs_list)
    num_nodes = [len(beh_seqs) for beh_seqs in beh_seqs_list]
    max_nodes = max(num_nodes)
    
    sample_seq = next(seq for beh_seqs in beh_seqs_list for seq in beh_seqs if len(seq) > 0)
    feature_dim = len(sample_seq[0]) if sample_seq else 0

    padded_beh_seqs = torch.zeros((len(beh_seqs_list), max_nodes, max_seq_length, feature_dim))
    padded_size_xs = torch.zeros((len(size_xs_list), max_nodes))
    padded_size_ys = torch.zeros((len(size_ys_list), max_nodes))
    padded_size_zs = torch.zeros((len(size_zs_list), max_nodes))
    padded_labels = torch.zeros((len(beh_seqs_list), max_nodes), dtype=torch.long)
    batch_masks = torch.zeros((len(beh_seqs_list), max_nodes), dtype=torch.bool)
    lengths = torch.zeros((len(beh_seqs_list), max_nodes), dtype=torch.long)

    max_edges = max(len(edges) for edges in edges_list)
    batch_edges = torch.zeros((len(edges_list), max_edges, 2), dtype=torch.long)
    batch_edge_masks = torch.zeros((len(edges_list), max_edges), dtype=torch.bool)

    for i, (beh_seqs, edges, size_xs, size_ys, size_zs, lbls, filenames) in enumerate(
            zip(beh_seqs_list, edges_list, size_xs_list, size_ys_list, size_zs_list, labels_list, filenames_list)):
        n = num_nodes[i]
        for j, seq in enumerate(beh_seqs):
            seq_length = len(seq)
            if seq_length > 0:
                padded_beh_seqs[i, j, :seq_length] = torch.tensor(seq, dtype=torch.float)
                lengths[i, j] = seq_length
        padded_size_xs[i, :n] = size_xs.clone().detach()
        padded_size_ys[i, :n] = size_ys.clone().detach()
        padded_size_zs[i, :n] = size_zs.clone().detach()
        padded_labels[i, :n] = lbls
        batch_masks[i, :n] = 1

        n_edges = len(edges)
        if n_edges > 0:
            batch_edges[i, :n_edges, :] = edges
        batch_edge_masks[i, :n_edges] = 1

    padded_interactions = torch.zeros(
        (max(graph_nums), len(beh_seqs_list), max_nodes, max_nodes, n_states, n_states, n_inter_actions))
    for batch_idx, graphs in enumerate(interactions_list):
        num_nodes = len(beh_seqs_list[batch_idx])
        for graph_idx, graph_matrix in enumerate(graphs):
            padded_interactions[graph_idx, batch_idx, :num_nodes, :num_nodes, :, :, :] = torch.tensor(graph_matrix,
                                                                                                      dtype=torch.float)
    return (padded_beh_seqs, batch_edges, batch_edge_masks, padded_size_xs, padded_size_ys, padded_size_zs,
            padded_labels, batch_masks, lengths, padded_interactions, max(graph_nums), list(filenames_list))
