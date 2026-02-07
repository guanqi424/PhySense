import argparse
import pickle
import sys
from collections import Counter
import torch
from sklearn.utils import resample
from torch import nn
from torch.utils.data import random_split, DataLoader
from phySense_initial.phySenseLSTM import ATBiLSTM
import random
import os

LabelSpace = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    with open(args.dataset_file, 'rb') as f:
        final_dataset_nongt = pickle.load(f)

    final_dataset_nongt = {key: value for key, value in final_dataset_nongt.items() if any(sub in key for sub in LabelSpace)}

    final_items_nongt = list(final_dataset_nongt.items())
    random.shuffle(final_items_nongt)

    final_items_nongt = dict(final_items_nongt)

    behavior_dataset_nongt = []
    for frames in final_items_nongt.values():
        my_behavior = None
        start_i = 0
        end_i = 0
        for i in range(len(frames)):
            if 'behavior' in frames[i]:
                if my_behavior is not None and my_behavior != frames[i]['behavior']:
                    if end_i > start_i:
                        my_sequence = []
                        for j in range(end_i - start_i):
                            my_sequence.append([
                                frames[start_i + j]['timestamp'],
                                frames[start_i + j]['translation'][0] - frames[start_i]['translation'][0],
                                frames[start_i + j]['translation'][1] - frames[start_i]['translation'][1],
                                frames[start_i + j]['translation'][2] - frames[start_i]['translation'][2],
                                frames[start_i + j]['size'][0], frames[start_i + j]['size'][1],
                                frames[start_i + j]['size'][2],
                                frames[start_i + j]['velocity'][0], frames[start_i + j]['velocity'][1]
                            ])
                        behavior_dataset_nongt.append((my_sequence, my_behavior))
                    start_i = end_i = i
                    my_behavior = frames[i]['behavior']
                elif my_behavior is None:
                    start_i = end_i = i
                    my_behavior = frames[i]['behavior']
                else:
                    end_i = i
            else:
                if my_behavior is not None:
                    if end_i > start_i:
                        my_sequence = []
                        for j in range(end_i - start_i):
                            my_sequence.append([
                                frames[start_i + j]['timestamp'],
                                frames[start_i + j]['translation'][0] - frames[start_i]['translation'][0],
                                frames[start_i + j]['translation'][1] - frames[start_i]['translation'][1],
                                frames[start_i + j]['translation'][2] - frames[start_i]['translation'][2],
                                frames[start_i + j]['size'][0], frames[start_i + j]['size'][1],
                                frames[start_i + j]['size'][2],
                                frames[start_i + j]['velocity'][0], frames[start_i + j]['velocity'][1]
                            ])
                        behavior_dataset_nongt.append((my_sequence, my_behavior))
                    start_i = end_i = i
                    my_behavior = None

            if i == len(frames) - 1 and my_behavior is not None:
                if end_i > start_i:
                    my_sequence = []
                    for j in range(end_i - start_i):
                        my_sequence.append([
                            frames[start_i + j]['timestamp'],
                            frames[start_i + j]['translation'][0] - frames[start_i]['translation'][0],
                            frames[start_i + j]['translation'][1] - frames[start_i]['translation'][1],
                            frames[start_i + j]['translation'][2] - frames[start_i]['translation'][2],
                            frames[start_i + j]['size'][0], frames[start_i + j]['size'][1],
                            frames[start_i + j]['size'][2],
                            frames[start_i + j]['velocity'][0], frames[start_i + j]['velocity'][1]
                        ])
                    behavior_dataset_nongt.append((my_sequence, my_behavior))

    washed_transformed_seqs = ATBiLSTM.add_acc_gt(behavior_dataset_nongt)
    normalized_dataset = ATBiLSTM.normalize_vector_magnitude(washed_transformed_seqs)
    my_dataset = ATBiLSTM.CustomDataset(normalized_dataset)
    class_counts_nongt_all = Counter()
    inverse_label_dict_nongt = {v: k for k, v in my_dataset.label_to_int.items()}

    for _, label in my_dataset:
        class_counts_nongt_all[inverse_label_dict_nongt[int(label)]] += 1

    train_size = int(args.train_frac * len(my_dataset))
    val_size = int(args.val_frac * len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(args.seed))

    class_counts_nongt_train = Counter()
    for _, label in train_dataset:
        class_counts_nongt_train[inverse_label_dict_nongt[int(label)]] += 1
    max_samples = max(class_counts_nongt_train.values())

    samples_per_class = {label: [] for label in class_counts_nongt_train.keys()}
    for data, label in train_dataset:
        label_text = inverse_label_dict_nongt[int(label)]
        samples_per_class[label_text].append((data, label_text))

    balanced_samples = []
    for label, samples in samples_per_class.items():
        replace_flag = class_counts_nongt_train[label] < max_samples // 2
        oversampled_samples = resample(samples, replace=replace_flag, n_samples=max_samples // 2, random_state=args.seed)
        balanced_samples.extend(oversampled_samples)

    balanced_train_dataset = ATBiLSTM.CustomDataset(balanced_samples)
    balanced_train_dataset.label_to_int = my_dataset.label_to_int

    train_loader = DataLoader(balanced_train_dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=ATBiLSTM.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ATBiLSTM.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=ATBiLSTM.collate_fn)

    model_LSTM_nongt = ATBiLSTM.ATBiLSTM(args.input_size, args.hidden_size, args.num_layers, len(my_dataset.label_to_int))

    class_sample_counts_nongt = [0 for _ in range(len(class_counts_nongt_all))]
    for index in range(len(class_sample_counts_nongt)):
        label_text = inverse_label_dict_nongt[index]
        class_sample_counts_nongt[index] = class_counts_nongt_all[label_text]

    weights_nongt = 1. / torch.tensor(class_sample_counts_nongt, dtype=torch.float)
    weights_nongt = weights_nongt / weights_nongt.sum()
    weights_nongt = weights_nongt.to(device)

    criterion_nongt = nn.CrossEntropyLoss()
    optimizer_nongt = torch.optim.Adam(model_LSTM_nongt.parameters(), lr=args.learning_rate)

    print("Starting to train the AT-BiLSTM behavior model")
    model_LSTM_nongt.to(device)
    ATBiLSTM.train_model(model_LSTM_nongt, train_loader, val_loader, criterion_nongt, optimizer_nongt,
                            num_epochs=args.num_epochs,
                            device=device)

    torch.save(model_LSTM_nongt, args.model_save_path)
    with open(args.label_to_int_save_path, 'wb') as f:
        pickle.dump(my_dataset.label_to_int, f)

    ATBiLSTM.test_model(model_LSTM_nongt, test_loader, label_dict=my_dataset.label_to_int, device=device, small_labelspace=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for LSTM with Attention.')
    parser.add_argument('--cuda_device', type=str, default="0", help='CUDA device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dataset_file', type=str, default='./data/nusc/nuscenes_behavior_dataset.pkl', help='Dataset file path')
    parser.add_argument('--train_frac', type=float, default=0.7, help='Training data fraction')
    parser.add_argument('--val_frac', type=float, default=0.15, help='Validation data fraction')
    parser.add_argument('--test_frac', type=float, default=0.15, help='Test data fraction')
    parser.add_argument('--input_size', type=int, default=9, help='Input size for LSTM')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers in LSTM')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='trained_models/at_bilstm.pth', help='Path to save trained model')
    parser.add_argument('--label_to_int_save_path', type=str, default='trained_models/lstm_label_to_int.pkl', help='Path to save label to int mapping')
    
    args = parser.parse_args()
    os.makedirs('trained_models', exist_ok=True)
    main(args)
