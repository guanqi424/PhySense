import argparse
import pickle
import torch
from tqdm import tqdm
from phySense_initial import phySense_initial
from scripts.interactions_construct_nusc import main as interaction_mean
from scripts.interactions_construct_nusc import parse_arguments as interaction_parse_arguments
from torch.utils.data import DataLoader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Runtime evaluation script')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for computation (default: auto-detect)')
    
    # File paths
    parser.add_argument('--lstm_label_df_path', type=str, default='weights/lstm_label_df.pkl',
                        help='Path to LSTM label DataFrame pickle file')
    parser.add_argument('--lstm_model_path', type=str, default='weights/at_bilstm.pth',
                        help='Path to LSTM model file')
    parser.add_argument('--lstm_label_path', type=str, default='weights/lstm_label_to_int.pkl',
                        help='Path to LSTM label-to-int mapping file')
    parser.add_argument('--behavior_mask_path', type=str, default='weights/tensor_behavior_mask.pt',
                        help='')
    parser.add_argument('--bayesian_dataset_path', type=str, default='weights/bayesian_dataset_nusc.json',
                        help='Path to Bayesian dataset file')
    parser.add_argument('--bayesian_model_path', type=str, default='weights/xgb_lpb_model.json',
                        help='Path to Bayesian model file')
    parser.add_argument('--label_encoder_path', type=str, default='weights/label_encoder_classes.npy',
                        help='Path to label encoder file')
    parser.add_argument('--crf_model_path', type=str, default='weights/crf_model_best.pth',
                        help='Path to CRF model file')
    parser.add_argument('--test_dataset_path', type=str, default='data/nusc/crf_dataset_qd3dt/test',
                        help='Path to test dataset folder')
    parser.add_argument('--label_encoder_classes_path', type=str, default='weights/label_encoder_classes.npy',
                        help='Path to label encoder classes file')

    # Model configuration
    parser.add_argument('--num_size_x_bin', type=int, default=20, help='Number of bins for size X')
    parser.add_argument('--num_size_y_bin', type=int, default=20, help='Number of bins for size Y')
    parser.add_argument('--num_size_z_bin', type=int, default=20, help='Number of bins for size Z')
    parser.add_argument('--edge_buffer_ratio', type=tuple, default=(0.2, 0.2), help='Edge buffer ratio')
    parser.add_argument('--num_regions', type=int, default=8, help='Number of regions')
    parser.add_argument('--region_size_ratio', type=tuple, default=(0.3, 0.3), help='Region size ratio')
    parser.add_argument('--num_states', type=int, default=5, help='Number of states')
    parser.add_argument('--num_actions', type=int, default=15, help='Number of actions')
    parser.add_argument('--num_inter_actions', type=int, default=1, help='Number of inter actions')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')

    # Flags
    parser.add_argument('--small_labelspace', action='store_true', default=True, help='Use small label space')
    parser.add_argument('--use_precompute', action='store_true', default=False, help='Use precomputed values')

    return parser.parse_args()

def main(smoketest=False):
    args = parse_arguments()

    device = torch.device(args.device)

    # Load LSTM label DataFrame
    with open(args.lstm_label_df_path, 'rb') as file:
        lstm_label_df = pickle.load(file)

    # Initialize phySense_initial
    myGuard_runtime_eval = phySense_initial.phySense(
        lstm_model_path=args.lstm_model_path,
        lstm_label_path=args.lstm_label_path,
        file_path=args.bayesian_dataset_path,
        num_size_x_bin=args.num_size_x_bin, num_size_y_bin=args.num_size_y_bin, num_size_z_bin=args.num_size_z_bin,
        bayesian_model_path=args.bayesian_model_path,
        label_encoder_path=args.label_encoder_path,
        behavior_mask_path=args.behavior_mask_path,
        edge_buffer_ratio=args.edge_buffer_ratio, num_regions=args.num_regions,
        region_size_ratio=args.region_size_ratio,
        num_states=args.num_states, num_actions=args.num_actions, num_inter_actions=args.num_inter_actions,
        beam_size=args.beam_size, device=device, lstm_label_df=lstm_label_df, 
        small_labelspace=args.small_labelspace, use_precompute=args.use_precompute
    )

    # Load CRF model state
    crf_state_dict = torch.load(args.crf_model_path)
    myGuard_runtime_eval.crf.load_state_dict(crf_state_dict)
    myGuard_runtime_eval.to(device)
    myGuard_runtime_eval.eval()

    def wrapper_collate_fn(batch):
        return phySense_initial.custom_collate_fn(batch, n_states=args.num_states)

    test_dataset_nongt = phySense_initial.GraphDataset(args.test_dataset_path, args.label_encoder_classes_path)
    test_loader_nongt = DataLoader(test_dataset_nongt, batch_size=args.batch_size, shuffle=False, collate_fn=wrapper_collate_fn)

    total_size_time = 0
    total_lbp_time = 0
    total_lstm_time = 0
    total_crf_time = 0

    with torch.no_grad():
        counter = 0
        for batch_data in tqdm(test_loader_nongt):
            if smoketest and counter == 10:
                break
            (padded_beh_seqs, batch_edges, batch_edge_masks, size_xs, size_ys, size_zs,
             padded_labels, batch_masks, lengths, interactions, graph_num, filenames) = batch_data

            # Move data to device
            padded_beh_seqs, batch_edges, batch_edge_masks, size_xs, size_ys, size_zs, padded_labels, batch_masks, interactions = \
                padded_beh_seqs.to(device), batch_edges.to(device), batch_edge_masks.to(device), \
                size_xs.to(device), size_ys.to(device), size_zs.to(device), padded_labels.to(device), \
                batch_masks.to(device), interactions.to(device)
            new_filenames = []
            for i in filenames:
                new_filenames.append([])
                for j in i:
                    new_filenames[-1].append('./data/nusc/bbox_dataset/b'+j[8:])

            _, runtime_breakdown = myGuard_runtime_eval.decode(
                new_filenames, size_xs, size_ys, size_zs, padded_beh_seqs, lengths,
                batch_masks, batch_edges, interactions, graph_num, batch_edge_masks, device, runtime_breakdown=True
            )
            total_size_time += runtime_breakdown[0]
            total_lbp_time += runtime_breakdown[1]
            total_lstm_time += runtime_breakdown[2]
            total_crf_time += runtime_breakdown[3]
            counter += 1
    myGuard_runtime_eval.bayesianUnary.texture_predictor.shutdown()

    if smoketest:
        return
    # Interaction construction and runtime test
    interaction_args = interaction_parse_arguments()
    interaction_args.runtime_test = True
    int_inference_time, graph_construction_time = interaction_mean(interaction_args)

    # Print runtime breakdown
    print('Initial Runtime Breakdown')
    print('3D Size: ', total_size_time / len(test_loader_nongt))
    print('LBP: ', total_lbp_time / len(test_loader_nongt))
    print('AT-BiLSTM : ', total_lstm_time / len(test_loader_nongt))
    print('Rule-based Interaction inference: ', int_inference_time)
    print('Build Graph: ', graph_construction_time)
    print('CRF Inference: ', total_crf_time / len(test_loader_nongt))

if __name__ == '__main__':
    main()
