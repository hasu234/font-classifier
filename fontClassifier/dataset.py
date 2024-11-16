import os
from PIL import Image
from torch.utils.data import Dataset

class FontDataset(Dataset):
    def __init__(self, root_dir, preprocess_func, transform=None):
        """
        Args:
        - root_dir (str): Path to the dataset folder.
        - preprocess_func (callable): Function to preprocess images.
        - transform (callable, optional): Transformations to apply after preprocessing.
        """
        self.root_dir = root_dir
        self.preprocess_func = preprocess_func
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        # Load all image paths and labels
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.endswith(('.png', '.jpg', '.jpeg')):
                        self.data.append(os.path.join(class_dir, img_file))
                        self.labels.append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if not already

        # Apply preprocessing
        image = self.preprocess_func(image)

        # Apply additional transformations
        if self.transform:
            image = self.transform(image)

        return image, label