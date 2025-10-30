from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import random
import os
from utils.dataset_utils import class_to_image_path, extract_class
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, data_dir, extract_class_fn, class_to_image_path_fn, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.extract_class_fn = extract_class_fn
        self.class_to_image_path_fn = class_to_image_path_fn
        
        self.classes = list(set(self.extract_class_fn(i) for i in os.listdir(data_dir)))
        self.class_to_images = self.class_to_image_path_fn(self.classes, self.data_dir)
        self.cached_images = {}
        for cls, paths in self.class_to_images.items():
            self.cached_images[cls] = [transforms.Resize((224,224))(Image.open(p)) for p in paths]
    def __len__(self):
        return sum(len(images) for images in self.class_to_images.values())

    def __getitem__(self, idx):
        anchor_class = random.choice(self.classes)
        positive_class = anchor_class
        negative_class = random.choice([c for c in self.classes if c != anchor_class])

        anchor_img = random.choice(self.cached_images[anchor_class])
        positive_img = random.choice(self.cached_images[positive_class])
        negative_img = random.choice(self.cached_images[negative_class])

        # anchor_img = Image.open(anchor_img_path)
        # positive_img = Image.open(positive_img_path)
        # negative_img = Image.open(negative_img_path)

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img


class DatasetClassification(Dataset):
    def __init__(self, data_dir, extract_class_fn, class_to_image_path_fn, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.extract_class_fn = extract_class_fn
        self.class_to_image_path_fn = class_to_image_path_fn
        
        self.classes = list(set(self.extract_class_fn(i) for i in os.listdir(data_dir)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.class_to_images = self.class_to_image_path_fn(self.classes, self.data_dir)
        self.cached_images = []
        for cls, paths in self.class_to_images.items():
            for path in paths:
                self.cached_images.append({cls: transforms.Resize((224,224))(Image.open(path))}) 
                
    def __len__(self):
        return sum(len(images) for images in self.class_to_images.values())

    def __getitem__(self, idx):
        image = list(self.cached_images[idx].values())[0]
        label_string = list(self.cached_images[idx].keys())[0]
        label = self.class_to_idx[label_string]
        # anchor_class = random.choice(self.classes)
        # positive_class = anchor_class
        # negative_class = random.choice([c for c in self.classes if c != anchor_class])

        # anchor_img = random.choice(self.cached_images[anchor_class])
        # positive_img = random.choice(self.cached_images[positive_class])
        # negative_img = random.choice(self.cached_images[negative_class])

        # # anchor_img = Image.open(anchor_img_path)
        # # positive_img = Image.open(positive_img_path)
        # # negative_img = Image.open(negative_img_path)

        if self.transform:
            image = self.transform(image)
            
        return image, label

