import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as F
import numpy as np
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = Image.open(self.file_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        image = torch.squeeze(torch.sum(image, dim=2, keepdim=True))
        return image, label


def data_set_split(image_path):

    # 读取所有图像文件路径和标签
    file_paths = []
    labels = []
    labels__ = []
    label_map = {}
    label_index = 0

    for label_folder in sorted(os.listdir(image_path)):
        labels__.append(label_folder)
        label_folder_path = os.path.join(image_path, label_folder)
        if os.path.isdir(label_folder_path):
            for file_name in os.listdir(label_folder_path):
                file_paths.append(os.path.join(label_folder_path, file_name))
                labels.append(label_folder)
            label_map[label_folder] = label_index
            label_index += 1

    # 标签转换为索�?
    labels = [label_map[label] for label in labels]

    # 自定�? 6:2:2 比例划分
    train_ratio = 0.6
    val_ratio = 0.2
    test_ratio = 0.2

    # 先划分训练集和剩余集
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(file_paths, labels, test_size=(1-train_ratio), random_state=42)

    # 再划分验证集和测试集
    val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42)

    # 创建数据变换
    data_transform = {# �ж�ʧ��Ϣ�Ŀ��ܣ������Ż�
        "train": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda x: torch.squeeze(torch.sum(x, dim=2, keepdim=True)))  # image = torch.squeeze(torch.sum(image, dim=2, keepdim=True))由此句修改
        ]),
        "val": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda x: torch.squeeze(torch.sum(x, dim=2, keepdim=True)))  # ????
        ]),
        "test": transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            # transforms.Lambda(lambda x: torch.squeeze(torch.sum(x, dim=2, keepdim=True)))  # ????
        ])
    }

    # 创建数据�?
    train_dataset = CustomDataset(train_paths, train_labels, transform=data_transform["train"])
    validate_dataset = CustomDataset(val_paths, val_labels, transform=data_transform["val"])
    test_dataset = CustomDataset(test_paths, test_labels, transform=data_transform["test"])

    # print(f"训练集大�?: {len(train_dataset)}")
    # print(f"验证集大�?: {len(validate_dataset)}")
    # print(f"测试集大�?: {len(test_dataset)}")

    return train_dataset, validate_dataset, test_dataset, labels__
