import os
import os.path
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image


class DataSource(data.Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = os.path.expanduser(root)
        self.transforms = transforms
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            if not train:
                self.transforms = T.Compose(
                    [T.Resize(224),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     normalize]
                )
            else:
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.RandomCrop(224),
                     T.ToTensor(),
                     normalize]
                )

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, img_pose

    def __len__(self):
        return len(self.images_path)