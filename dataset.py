import os
from os.path import join

import imageio
from jittor import transform
from jittor.dataset import Dataset


class ColorSymbol(Dataset):
    def __init__(self, path, split):
        super().__init__()
        self.path = path
        self.files = []
        self.img_size = (128, 128)
        self.transform = transform.Compose([
            transform.RandomHorizontalFlip(),
            transform.Resize(self.img_size),
            transform.ToTensor(),
            transform.ImageNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        for file_name in sorted(os.listdir(self.path)):
            self.files.append(file_name)
        self.split = split

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image = imageio.imread(join(self.path, self.files[index]))
        if self.split == 'train':
            image = self.transform(image)
        return image
