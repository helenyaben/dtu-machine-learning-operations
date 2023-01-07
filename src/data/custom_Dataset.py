from torch.utils.data import Dataset
import torch
from typing import Tuple
import os

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, train:bool=True):
        
        # 3. Create class attributes
        self.classes = list(range(0, 10))
        self.class_to_idx = {str(idx):idx for idx in self.classes}

        # Import preprocessed data
        train_test_data = torch.load(os.path.join(targ_dir, 'train_test_processed.pt'))
        if train:
            self.images = train_test_data['train_data']
            self.labels = train_test_data['train_labels']
        else:
            self.images = train_test_data['test_data']
            self.labels = train_test_data['test_labels']

    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:

        return self.images.shape[0]
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.images[index]
        class_name  = str(int(self.labels[index]))
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        return img, class_idx # return data, label (X, y)