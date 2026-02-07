import argparse
import pickle
import random
from phySense_initial.phySenseLSTM import ATBiLSTM


LabelSpace = ['bicycle', 'bus', 'car', 'pedestrian', 'truck']

def main(args):
    with open(args.behavior_anno_path, 'rb') as f:
        final_dataset_nongt_150 = pickle.load(f)
    totalbeh = 0
    beh_labels = {}
    for key, value in final_dataset_nongt_150.items():
        for item in value:
            if 'behavior' in item:
                totalbeh += 1
                if item['behavior'] not in beh_labels:
                    beh_labels[item['behavior']] = 0
                beh_labels[item['behavior']] += 1

    print("Total annotated behavior datapoint:", totalbeh)
    for key, value in beh_labels.items():
        print(key, ": ", value)

    final_dataset_nongt = {key: value for key, value in final_dataset_nongt_150.items() if any(sub in key for sub in LabelSpace)}

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
    print()
    print("As an example, the first line of the behavior dataset is:")
    print()
    print("Data  :", normalized_dataset[0][0])
    print()
    print("Label :", normalized_dataset[0][1])
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Show Behavior Label Samples")
    parser.add_argument('--behavior_anno_path', type=str, default='./data/nusc/nuscenes_behavior_dataset.pkl', help='Path to the annotated behavior dataset.')

    args = parser.parse_args()
    main(args)
