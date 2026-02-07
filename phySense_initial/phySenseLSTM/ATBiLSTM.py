import pickle
import time
from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import resample
from tqdm import tqdm

import seaborn as sns


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.key_layer = nn.Linear(key_size, hidden_size)
        self.query_layer = nn.Linear(query_size, hidden_size)
        self.score_layer = nn.Linear(hidden_size, 1)

    def forward(self, query, values):
        # Transforming the query and keys
        query_transformed = self.query_layer(query)
        keys_transformed = self.key_layer(values)

        # Calculating scores
        scores = self.score_layer(torch.tanh(query_transformed.unsqueeze(2) + keys_transformed.unsqueeze(1)))
        scores = scores.squeeze(-1)

        # Applying softmax to get attention weights
        attn_weights = F.softmax(scores, dim=2)

        # Multiplying the weights with the values to get the context
        context = torch.bmm(attn_weights, values)
        return context, attn_weights


class ATBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dim_feedforward=1024):
        super(ATBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden_size * 2, hidden_size * 2, hidden_size * 2)
        self.fc1 = nn.Linear(hidden_size * 2, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, dim_feedforward // 2)
        self.fc3 = nn.Linear(dim_feedforward // 2, num_classes)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        x_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(x_packed, (h0, c0))

        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

        forward_out = lstm_out[torch.arange(len(lengths)), lengths - 1, :self.hidden_size]
        backward_out = lstm_out[:, 0, self.hidden_size:]
        last_outputs = torch.cat((forward_out, backward_out), dim=1)

        attn_out, attn_weights = self.attention(last_outputs.unsqueeze(1), lstm_out)  # (query, values)
        attn_out = attn_out.squeeze(1)

        output = self.leaky_relu(self.fc1(attn_out))
        output = self.leaky_relu(self.fc2(output))
        output = self.fc3(output)
        return output


def normalize_vector_magnitude(data):
    normalized_data = []
    for sequence, label in data:
        normalized_sequence = []
        for frame in sequence:
            xy = np.array(frame[1:3])
            velocity = np.array(frame[4:6])
            acc = np.array(frame[6:])
            normalized_sequence.append(np.concatenate([xy, velocity, acc]).tolist())

        normalized_data.append((normalized_sequence, label))

    return normalized_data


class CustomDataset(Dataset):
    def __init__(self, data_list, label_to_int=None):
        self.data_list = data_list
        if label_to_int is None:
            self.label_to_int = self.create_label_mapping()
        else:
            self.label_to_int = label_to_int

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, label = self.data_list[idx]

        data_tensor = torch.tensor(data, dtype=torch.float)
        label_int = self.label_to_int[label]
        label_tensor = torch.tensor(label_int, dtype=torch.long)

        return data_tensor, label_tensor

    def create_label_mapping(self):
        unique_labels = set(label for _, label in self.data_list)
        return {label: idx for idx, label in enumerate(unique_labels)}


def collate_fn(batch):
    data, labels = zip(*batch)
    lengths = [len(seq) for seq in data]
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)

    return data_padded, labels, lengths


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10,
                device=torch.device("cpu"), min_delta=1e-4, patience=15, save_path=None):
    model.train()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Create a tqdm progress bar
    progress_bar = tqdm(range(num_epochs), desc='Training Progress')

    for epoch in progress_bar:
        total_loss = 0
        for data_padded, labels, lengths in train_loader:
            data_padded = data_padded.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(data_padded, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        if (epoch + 1) % 5 == 0 and save_path is not None:
            torch.save(model, f'{save_path}/lstm_model_epoch_{epoch + 1}.pth')
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device=device)
        model.train()

        # Update the progress bar with additional information
        progress_bar.set_postfix({
            'Epoch': f'{epoch + 1}/{num_epochs}',
            'Train Loss': f'{avg_train_loss:.4f}',
            'Validation Loss': f'{val_loss:.4f}',
            'Validation Accuracy': f'{val_accuracy:.2f}%'
        })


def test_model(model, test_loader, label_dict, device=torch.device("cpu"), small_labelspace = False):
    model.eval()
    all_labels = []
    all_preds = []
    correct = 0
    total = 0
    total_time = 0

    with torch.no_grad():
        for data_padded, labels, lengths in test_loader:
            start_time = time.time()
            data_padded = data_padded.to(device)
            labels = labels.to(device)

            outputs = model(data_padded, lengths)
            _, predicted = torch.max(outputs.data, 1)
            end_time = time.time()
            total_time += end_time - start_time

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    inverse_label_dict = {v: k for k, v in label_dict.items()}
    inverse_label_dict_names = {k: v.split('#')[0] for k, v in inverse_label_dict.items()}
    unique_labels_in_test = sorted(set(all_labels + all_preds))
    target_names = [inverse_label_dict[label] for label in unique_labels_in_test]
    target_names_label = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck']
    target_names_label_to_int = {'barrier': 0,
                                 'bicycle': 1,
                                 'bus': 2,
                                 'car': 3,
                                 'construction_vehicle': 4,
                                 'motorcycle': 5,
                                 'pedestrian': 6,
                                 'traffic_cone': 7,
                                 'trailer': 8,
                                 'truck': 9}
    if small_labelspace:
        target_names_label = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']
        target_names_label_to_int = {'bicycle': 0,
                                     'bus': 1,
                                     'car': 2,
                                     'pedestrian': 3,
                                     'truck': 4}
    print(f'#{total} Inference time: {total_time} seconds, {total_time / total} s/it.')
    new_labels = []
    new_preds = []
    for i in range(len(all_labels)):
        my_class = target_names_label_to_int[inverse_label_dict_names[all_labels[i]]]
        new_labels.append(my_class)
    for i in range(len(all_preds)):
        my_class = target_names_label_to_int[inverse_label_dict_names[all_preds[i]]]
        new_preds.append(my_class)

    print()
    print('Classification Report (per label):')
    print(classification_report(new_labels, new_preds, labels=range(5), target_names=target_names_label, zero_division=0))

    # Calculate and print accuracy
    accuracy = accuracy_score(new_labels, new_preds)
    print(f'Accuracy: {accuracy:.2f}')


def evaluate_model(model, val_loader, criterion, device=torch.device("cpu")):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data_padded, labels, lengths in val_loader:
            data_padded = data_padded.to(device)
            labels = labels.to(device)
            outputs = model(data_padded, lengths)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def add_velocity_gt(washed_transformed_seqs):
    new_seqs = []
    for seq in washed_transformed_seqs:
        my_seq = ([], seq[1])
        for i in range(len(seq[0])):
            if i == 0 and len(seq[0]) > 2:
                if (seq[0][1][0] - seq[0][0][0]) != 0:
                    v1 = (np.array(seq[0][1])[1:3] - np.array(seq[0][0])[1:3]) / np.array(seq[0][1][0] - seq[0][0][0])
                else:
                    #raise ValueError('Same timestamp')
                    v1 = np.array(seq[0][1])[1:3] - np.array(seq[0][0])[1:3]
                if (seq[0][2][0] - seq[0][1][0]) != 0:
                    v2 = (np.array(seq[0][2])[1:3] - np.array(seq[0][1])[1:3]) / np.array(seq[0][2][0] - seq[0][1][0])
                else:
                    #raise ValueError('Same timestamp')
                    v2 = np.array(seq[0][2])[1:3] - np.array(seq[0][1])[1:3]
                velocity = 2 * v1 - v2
            elif i == 0 and len(seq[0]) > 1:
                if (seq[0][1][0] - seq[0][0][0]) != 0:
                    velocity = (np.array(seq[0][1])[1:3] - np.array(seq[0][0])[1:3]) / np.array(
                        seq[0][1][0] - seq[0][0][0])
                else:
                    #raise ValueError('Same timestamp')
                    velocity = np.array(seq[0][1])[1:3] - np.array(seq[0][0])[1:3]
            elif i == 0:
                velocity = np.array([0, 0])
            else:
                if (seq[0][i][0] - seq[0][i - 1][0]) == 0:
                    #raise ValueError('Same timestamp')
                    velocity = np.array(seq[0][i])[1:3] - np.array(seq[0][i - 1])[1:3]
                else:
                    velocity = (np.array(seq[0][i])[1:3] - np.array(seq[0][i - 1])[1:3]) / np.array(
                        seq[0][i][0] - seq[0][i - 1][0])
            my_seq[0].append(np.concatenate([np.array(seq[0][i]), velocity]).tolist())
        new_seqs.append(my_seq)
    return new_seqs


def add_acc_gt(washed_transformed_seqs):
    new_seqs = []
    for seq in washed_transformed_seqs:
        my_seq = ([], seq[1])
        for i in range(len(seq[0])):
            if i == len(seq[0]) - 1:
                if i == 0:
                    acc = np.array([0, 0])
                elif i == 1:
                    if (seq[0][i][0] - seq[0][i - 1][0]) != 0:
                        acc = (np.array(seq[0][i][-2:]) - np.array(seq[0][i - 1][-2:])) / np.array(
                            seq[0][i][0] - seq[0][i - 1][0])
                    else:
                        #raise ValueError('Same timestamp')
                        acc = (np.array(seq[0][i][-2:]) - np.array(seq[0][i - 1][-2:]))
                else:
                    if (seq[0][i - 1][0] - seq[0][i - 2][0]) != 0:
                        a1 = (np.array(seq[0][i - 1][-2:]) - np.array(seq[0][i - 2][-2:])) / np.array(
                            seq[0][i - 1][0] - seq[0][i - 2][0])
                    else:
                        #raise ValueError('Same timestamp')
                        a1 = (np.array(seq[0][i - 1][-2:]) - np.array(seq[0][i - 2][-2:]))
                    if (seq[0][i][0] - seq[0][i - 1][0]) != 0:
                        a2 = (np.array(seq[0][i][-2:]) - np.array(seq[0][i - 1][-2:])) / np.array(
                            seq[0][i][0] - seq[0][i - 1][0])
                    else:
                        #raise ValueError('Same timestamp')
                        a2 = (np.array(seq[0][i][-2:]) - np.array(seq[0][i - 1][-2:]))
                    acc = 2 * a2 - a1
            else:
                if (seq[0][i + 1][0] - seq[0][i][0]) != 0:
                    acc = (np.array(seq[0][i + 1][-2:]) - np.array(seq[0][i][-2:])) / (seq[0][i + 1][0] - seq[0][i][0])
                else:
                    #raise ValueError('Same timestamp')
                    acc = (np.array(seq[0][i + 1][-2:]) - np.array(seq[0][i][-2:]))
            my_seq[0].append(np.concatenate([np.array(seq[0][i]), acc]).tolist())
        new_seqs.append(my_seq)
    return new_seqs


def main():
    train_ratio = 0.7
    val_ratio = 0.15
    batch_size = 32

    input_size = 6
    # input_size = 4
    hidden_size = 512

    num_layers = 3
    num_classes = 12

    num_epochs = 100
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('./data_playground/washed_transformed_seqs_gt.pkl', 'rb') as file:
        washed_transformed_seqs = pickle.load(file)

    washed_transformed_seqs = add_velocity_gt(washed_transformed_seqs)

    washed_transformed_seqs = add_acc_gt(washed_transformed_seqs)

    normalized_dataset = normalize_vector_magnitude(washed_transformed_seqs)
    my_dataset = CustomDataset(normalized_dataset)

    class_counts = Counter()
    inverse_label_dict = {v: k for k, v in my_dataset.label_to_int.items()}

    for _, label in my_dataset:
        class_counts[inverse_label_dict[int(label)]] += 1
    print(class_counts)

    train_size = int(train_ratio * len(my_dataset))
    val_size = int(val_ratio * len(my_dataset))
    test_size = len(my_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size],
                                                            generator=torch.Generator().manual_seed(42))

    class_counts = Counter()
    inverse_label_dict = {v: k for k, v in my_dataset.label_to_int.items()}

    for _, label in train_dataset:
        class_counts[inverse_label_dict[int(label)]] += 1
    print(class_counts)
    max_samples = max(class_counts.values())

    samples_per_class = {label: [] for label in class_counts.keys()}
    for data, label in train_dataset:
        label_text = inverse_label_dict[int(label)]
        samples_per_class[label_text].append((data, label_text))

    balanced_samples = []
    for label, samples in samples_per_class.items():
        if label == 'stationary':
            oversampled_samples = resample(samples, replace=False, n_samples=max_samples // 3, random_state=42)
        else:
            oversampled_samples = resample(samples, replace=True, n_samples=max_samples // 3, random_state=42)
        balanced_samples.extend(oversampled_samples)

    balanced_train_dataset = CustomDataset(balanced_samples)
    balanced_train_dataset.label_to_int = my_dataset.label_to_int

    class_counts = Counter()
    inverse_label_dict = {v: k for k, v in balanced_train_dataset.label_to_int.items()}

    for _, label in balanced_train_dataset:
        class_counts[inverse_label_dict[int(label)]] += 1
    print(class_counts)

    class_counts = Counter()
    inverse_label_dict = {v: k for k, v in balanced_train_dataset.label_to_int.items()}

    for _, label in val_dataset:
        class_counts[inverse_label_dict[int(label)]] += 1
    print(class_counts)

    class_counts = Counter()
    inverse_label_dict = {v: k for k, v in balanced_train_dataset.label_to_int.items()}

    for _, label in test_dataset:
        class_counts[inverse_label_dict[int(label)]] += 1
    print(class_counts)

    train_loader = DataLoader(balanced_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = ATBiLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

    torch.save(model, 'lstm_attn_model.pth')

    test_model(model, test_loader, label_dict=my_dataset.label_to_int, device=device)


if __name__ == '__main__':
    main()
