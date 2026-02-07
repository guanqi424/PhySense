import os
import pickle
import argparse
import torch
from PIL import Image
from matplotlib import pyplot as plt
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
import cv2
from util import yolov3_pytorch, sticker_apply


def parse_arguments():
    parser = argparse.ArgumentParser(description="Adversarial Attack Script")

    parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for computation (e.g., 'cuda:0', 'cpu')")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--target_label', type=int, default=2, help="Target label for attack")
    parser.add_argument('--target_name', type=str, default='car', help="Target name for filtering images")
    parser.add_argument('--mask_path', type=str, default='./data/masks/mask_1024x1024_large.png', help="Path to the mask image")
    parser.add_argument('--dataset_root', type=str, default='./data/nusc/bbox_dataset', help="Root path to the dataset")
    parser.add_argument('--output_path', type=str, default='./attacker/nusc/', help="Path to save attack results")
    
    # Adam optimizer parameters
    parser.add_argument('--adam_lr', type=float, default=0.01, help="Learning rate for Adam optimizer")
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.999), help="Betas for Adam optimizer")
    parser.add_argument('--adam_eps', type=float, default=1e-08, help="Epsilon for Adam optimizer")

    return parser.parse_args()


class CustomImageDataset(Dataset):
    def __init__(self, all_imgs_path, dataset_root, transform=None):
        self.all_imgs = all_imgs_path
        self.dataset_root = dataset_root
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset_root, 'b' + self.all_imgs[idx][1:].lstrip('/')[6:])
        image = Image.open(img_path).convert("RGB")
        image = TF.to_tensor(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.all_imgs[idx]


def collate_fn(batch):
    imgs, labels = zip(*batch)
    return imgs, labels


def custom_loss(detections, target_class):
    target_confidences = []
    non_target_confidences = []

    if detections is not None:
        for detection in detections:
            confidence = detection[-2]
            class_pred = detection[-1]

            is_target_class = (int(class_pred) == target_class)
            if is_target_class:
                target_confidences.append(confidence * is_target_class)
            else:
                non_target_confidences.append((1.0 - confidence) * (1 - is_target_class))

    if target_confidences:
        target_confidences = torch.stack(target_confidences)
        target_loss = torch.mean(target_confidences)
    else:
        target_loss = torch.tensor(0.0, device=detections.device)

    if non_target_confidences:
        non_target_confidences = torch.stack(non_target_confidences)
        non_target_loss = torch.mean(non_target_confidences)
    else:
        non_target_loss = torch.tensor(0.0, device=detections.device)
    loss = target_loss + 1 - non_target_loss
    return loss


def tensor2cv2(input_tensor: torch.Tensor):
    my_input_tensor = input_tensor.detach().to(torch.device('cpu')).numpy()
    in_arr = np.transpose(my_input_tensor, (1, 2, 0))
    cv2img = cv2.cvtColor(np.uint8(in_arr * 255), cv2.COLOR_RGB2BGR)
    return cv2img


def cv2_to_tensor(image, device):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_transposed = image_rgb.transpose((2, 0, 1))
    image_normalized = image_transposed.astype('float32') / 255.0
    tensor_image = torch.from_numpy(image_normalized).to(device)
    return tensor_image


def showtensor(img_tensor):
    img_tensor_display = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_tensor_display)
    plt.show()


def untarget_attack(model, device, X, filenames, ori_sticker_tensor, project_point_dict, target_label, optimizer_params, lastNoiseTensor=None):
    if lastNoiseTensor is None:
        noise = torch.rand_like(ori_sticker_tensor).float() * ori_sticker_tensor
        noiseVar = noise.clamp(0, 1).to(device)
    else:
        noiseVar = lastNoiseTensor.to(device)

    noiseVar = autograd.Variable(noiseVar, requires_grad=True)
    optimizer = optim.Adam([noiseVar], lr=optimizer_params['lr'], betas=optimizer_params['betas'], eps=optimizer_params['eps'])
    global optimizer_state
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    optimizer.zero_grad()

    transformed_masks_tensor, flags = sticker_apply.project_mask_pytorch_batch(
        X, filenames, noiseVar, False, project_point_dict, model.img_size)
    stickers_transformed_tensor, _ = sticker_apply.project_mask_pytorch_batch(
        X, filenames, ori_sticker_tensor, True, project_point_dict, model.img_size)

    pert_imgs = (
        sticker_apply.resize_and_pad_tensors(X, model.img_size).to(device) * (1 - stickers_transformed_tensor) +
        transformed_masks_tensor * stickers_transformed_tensor)

    predictions = model.predict(pert_imgs)
    loss_total = None
    valid_num = 0
    for i, predict in enumerate(predictions):
        if not flags[i]:
            if predict is None or predict.nelement() == 0:
                continue
            if loss_total is None:
                loss_total = custom_loss(predict, target_label)
            else:
                loss_total += custom_loss(predict, target_label)
            valid_num += 1

    if valid_num > 0:
        loss_total = loss_total / valid_num
        loss_total.backward(retain_graph=True)
        optimizer.step()

    optimizer_state = optimizer.state_dict()
    noiseVar = noiseVar * ori_sticker_tensor
    noiseVar = noiseVar.clamp(0, 1)

    if loss_total is None:
        return noiseVar.detach(), 0
    return noiseVar.detach(), loss_total.item()


def find_in_dict(a, key, value):
    for list_key, list_of_dicts in a.items():
        for index, dict_item in enumerate(list_of_dicts):
            if dict_item.get(key) == value:
                return list_key, index
    return None, None


def evaluate(test_loader, result, project_point_dict, model, mask, data_this_test, root_path, target_label, device):
    this_attacked_instances = {}
    total = 0
    correct = 0
    correct_pert = 0
    attacked_num = 0
    with torch.no_grad():
        for images, file_names in tqdm(test_loader):
            for image, file_name in zip(images, file_names):
                transformed_masks_tensor, transformed_masks_tensor_unpadded, flags = sticker_apply.project_mask_pytorch_batch_worigin(
                    image.unsqueeze(0), [file_name], result, False, project_point_dict, model.img_size)
                if not flags[0]:
                    class_pred = None
                    class_pred_pert = None
                    total += 1
                    model_input = sticker_apply.resize_and_pad(image.unsqueeze(0), model.img_size)
                    outputs = model.predict(model_input)
                    if len(outputs) > 0:
                        predictions = outputs[0]
                        if predictions is not None and predictions.nelement() != 0:
                            for prediction in predictions:
                                class_pred = int(prediction[-1])
                                if class_pred == target_label:
                                    found_key, found_index = find_in_dict(data_this_test, 'path', file_name)
                                    if found_key is None:
                                        raise ValueError("Clean instance not found")
                                    if found_key not in this_attacked_instances:
                                        this_attacked_instances[found_key] = data_this_test[found_key]
                                        for index in range(len(this_attacked_instances[found_key])):
                                            this_attacked_instances[found_key][index]['Clean_Recognized'] = False
                                            this_attacked_instances[found_key][index]['attacked'] = False
                                    this_attacked_instances[found_key][found_index]['Clean_Recognized'] = True
                                    correct += 1
                                    break
                    stickers_transformed_tensor, stickers_transformed_tensor_unpadded, _ = sticker_apply.project_mask_pytorch_batch_worigin(
                        image.unsqueeze(0), [file_name], mask, True, project_point_dict, model.img_size)

                    pert_imgs = (sticker_apply.resize_and_pad_tensors(image.unsqueeze(0), model.img_size).to(device) * (
                            1 - stickers_transformed_tensor) +
                                 transformed_masks_tensor * stickers_transformed_tensor)

                    outputs = model.predict(pert_imgs)
                    if len(outputs) > 0:
                        predictions = outputs[0]
                        if predictions is not None and predictions.nelement() != 0:
                            for prediction in predictions:
                                class_pred_pert = int(prediction[-1])
                                if class_pred_pert == target_label:
                                    correct_pert += 1
                                    break
                            if class_pred_pert != target_label:
                                attacked_num += 1
                                pert_img_origin = (image.to(device) * (1 - stickers_transformed_tensor_unpadded[0]) +
                                                   transformed_masks_tensor_unpadded[0] *
                                                   stickers_transformed_tensor_unpadded[0])
                                cv2.imwrite(root_path + os.path.basename(file_name),
                                           tensor2cv2(pert_img_origin))
                                found_key, found_index = find_in_dict(data_this_test, 'path', file_name)
                                if found_key is None:
                                    raise ValueError("Attacked instance not found")
                                if found_key not in this_attacked_instances:
                                    this_attacked_instances[found_key] = data_this_test[found_key]
                                    for index in range(len(this_attacked_instances[found_key])):
                                        this_attacked_instances[found_key][index]['Clean_Recognized'] = False
                                        this_attacked_instances[found_key][index]['attacked'] = False
                                this_attacked_instances[found_key][found_index]['attacked'] = True
                        else:
                            attacked_num += 1
                            pert_img_origin = (image.to(device) * (1 - stickers_transformed_tensor_unpadded[0]) +
                                               transformed_masks_tensor_unpadded[0] *
                                               stickers_transformed_tensor_unpadded[0])
                            cv2.imwrite(root_path + os.path.basename(file_name),
                                       tensor2cv2(pert_img_origin))
                            found_key, found_index = find_in_dict(data_this_test, 'path', file_name)
                            if found_key is None:
                                raise ValueError("Attacked instance not found")
                            if found_key not in this_attacked_instances:
                                this_attacked_instances[found_key] = data_this_test[found_key]
                                for index in range(len(this_attacked_instances[found_key])):
                                    this_attacked_instances[found_key][index]['Clean_Recognized'] = False
                                    this_attacked_instances[found_key][index]['attacked'] = False
                            this_attacked_instances[found_key][found_index]['attacked'] = True
                    else:
                        raise RuntimeError('yolo return nothing')

    print(f"model acc : {correct / total}")
    print(f"attacked model acc : {correct_pert / total}")
    print(f"asr: {attacked_num / total}")
    return this_attacked_instances


def main(device, num_epochs, batch_size, seed, 
         target_label, target_name, mask_path, output_path, 
         adam_lr, adam_betas, adam_eps):
    args = parse_arguments()
    args.num_epochs = num_epochs
    args.output_path = output_path
    args.mask_path = mask_path
    args.batch_size = batch_size
    os.makedirs(args.output_path, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    global optimizer_state
    optimizer_state = None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    target_label = target_label
    target_name = target_name

    optimizer_params = {
        'lr': adam_lr,
        'betas': adam_betas,
        'eps': adam_eps
    }
    
    with open(os.path.join(args.dataset_root, 'train_data_nongt_new.pkl'), 'rb') as f:
        train_data_nongt_new = pickle.load(f)
    with open(os.path.join(args.dataset_root, 'val_data_nongt_new.pkl'), 'rb') as f:
        val_data_nongt_new = pickle.load(f)
    with open(os.path.join(args.dataset_root, 'test_data_nongt_new.pkl'), 'rb') as f:
        test_data_nongt_new = pickle.load(f)

    data_nongt_train = {k: train_data_nongt_new[k] for k in sorted(train_data_nongt_new)}
    data_nongt_test = {**val_data_nongt_new, **test_data_nongt_new}
    data_nongt_test = {k: data_nongt_test[k] for k in sorted(data_nongt_test)}

    with open(os.path.join(args.dataset_root, 'project_point_dict_nongt.pkl'), 'rb') as f:
        project_point_dict_nongt = pickle.load(f)

    print("Start attack")
    model = yolov3_pytorch.yolov3(device)
    nongt_train_imgs = []
    nongt_test_imgs = []
    for key, value in data_nongt_train.items():
        if target_name not in key:
            continue
        for frame in value:
            nongt_train_imgs.append(frame['path'])
    for key, value in data_nongt_test.items():
        if target_name not in key:
            continue
        for frame in value:
            nongt_test_imgs.append(frame['path'])

    train_dataset = CustomImageDataset(nongt_train_imgs + nongt_test_imgs, dataset_root=args.dataset_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    test_dataset = CustomImageDataset(nongt_train_imgs + nongt_test_imgs, dataset_root=args.dataset_root)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

    mask = Image.open(args.mask_path)
    mask = TF.to_tensor(mask).to(device)
    result = None
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:
        total_loss = 0
        for i, (images, file_names) in enumerate(train_loader):
            result, loss = untarget_attack(model, device, images, file_names, mask, project_point_dict_nongt, target_label, optimizer_params, result)
            pbar.set_description(f"Epoch {epoch} Batch {i + 1}/{len(train_loader)}, Loss {loss}")
            total_loss += loss

        cv2.imwrite(os.path.join(args.output_path, f'{target_name}_epoch{epoch}_loss{total_loss / len(train_loader)}.png'),
                    tensor2cv2(result))
        print(f'Epoch {epoch} Loss {total_loss / len(train_loader)}')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch}')
            root_path = os.path.join(args.output_path, f'{target_name}/epoch{epoch}/')
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            nongt_attacked_instances = evaluate(test_loader, result, project_point_dict_nongt, model, mask,
                                                {**data_nongt_train, **data_nongt_test},
                                                root_path, target_label, device)
            for key, value in nongt_attacked_instances.items():
                attacked_frame = 0
                for frame in value:
                    if frame['attacked']:
                        attacked_frame += 1
                nongt_attacked_instances[key] = (attacked_frame / len(value), nongt_attacked_instances[key])

            with open(os.path.join(args.output_path, f'attacked_instances_{target_name}_epoch_{epoch}.pkl'), 'wb') as f:
                pickle.dump(nongt_attacked_instances, f)

    print('Done')


if __name__ == "__main__":
    main()
