import os
from scripts import physical_attack_kitti, physical_attack_nusc
import argparse
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(description='Attack Training Script')

    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for computation (e.g., 'cuda:0', 'cpu')")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--target_label', type=int, default=2, help="Target label for attack")
    parser.add_argument('--target_name', type=str, default='car', help="Target name for filtering images")
    parser.add_argument('--mask_path', type=str, default='./data/masks/mask_1024x1024_large.png', help="Path to the mask image")
    parser.add_argument('--output_path', type=str, default='./attacker/', help="Path to save attack results")
    
    # Adam optimizer parameters
    parser.add_argument('--adam_lr', type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.999), help="Betas for Adam optimizer")
    parser.add_argument('--adam_eps', type=float, default=1e-08, help="Epsilon for Adam optimizer")


    # Flags for selecting dataset
    # Add --patch_size argument with choices 'large' and 'small'
    parser.add_argument(
        '--patch_size',
        choices=['large', 'small'],
        required=True,
        help='Specify the patch size: large or small.'
    )

    # Add --dataset argument with choices 'nuscene' and 'kitti'
    parser.add_argument(
        '--dataset',
        choices=['nuscenes', 'kitti'],
        required=True,
        help='Specify the dataset: nuscene or kitti.'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    sys.argv = [sys.argv[0]]
    if args.patch_size == 'large':
        args.mask_path = './data/masks/mask_1024x1024_large.png'
    else:
        args.mask_path = './data/masks/mask_1024x1024_small.png'
    if args.dataset == 'kitti':
        args.output_path = args.output_path + 'kitti/'
    else:
        args.output_path = args.output_path + 'nusc/'
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.dataset == 'kitti':
        print("Run Attack on KITTI")
        physical_attack_kitti.main(args.device, args.num_epochs, args.batch_size, args.seed, 
                                   args.target_label, args.target_name, args.mask_path, args.output_path, 
                                   args.adam_lr, args.adam_betas, args.adam_eps)
    if args.dataset == 'nuscenes':
        print("Run Attack on nuScenes")
        physical_attack_nusc.main(args.device, args.num_epochs, args.batch_size, args.seed, 
                                   args.target_label, args.target_name, args.mask_path, args.output_path, 
                                   args.adam_lr, args.adam_betas, args.adam_eps)


if __name__ == "__main__":
    main()
