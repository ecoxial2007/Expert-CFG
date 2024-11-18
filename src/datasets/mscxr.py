import os
import json
from PIL import Image
from torch.utils.data import Dataset

class MSCXDataset(Dataset):
    def __init__(self, json_path, img_root, transform=None):
        """
        初始化数据集类。

        参数:
            json_path (str): JSON文件的路径，用于读取标注信息。
            img_root (str): 图像根目录的路径。
            transform (callable, optional): 图像转换方法，应用在每个PIL图像上。
        """
        # 1. 读取并解析JSON文件
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # 2. 提取图像和注释信息
        self.images = data["images"]
        self.annotations = data["annotations"]
        self.img_root = img_root
        self.transform = transform

        # 3. 创建图像ID和注释的映射关系
        self.img_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[img_id] = []
            self.img_id_to_annotations[img_id].append(ann)

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        根据索引返回一个样本，包括图像和相关的注释信息。
        """
        # 4. 获取图像信息
        img_info = self.images[index]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_root, img_info["path"])
        
        # 5. 加载图像
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 6. 获取该图像的所有注释
        annotations = self.img_id_to_annotations.get(img_id, [])
        bboxes = []
        labels = []
        
        # 7. 处理每个注释
        for ann in annotations:
            x1, y1 = ann["bbox"][:2]
            x2, y2 = ann["bbox"][2:]
            bboxes.append([x1, y1, x2, y2])
            labels.append(ann["label_text"])

        return {
            "image": image,
            "bboxes": bboxes,
            "labels": labels,
            "image_id": img_id
        }
