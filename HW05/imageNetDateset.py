import torchvision.datasets as datasets
import random

class ImageNetDataset(datasets.ImageFolder):

    def __init__(self, root, annotations, class_to_idx, transform=None):
        super(ImageNetDataset, self).__init__(root, transform)
        self.file_to_label = self.generate_labels(root, annotations, class_to_idx)
        self.samples = self.generate_samples(self.imgs, self.file_to_label)

    
    def generate_labels(root, annotations, class_to_idx):
        file_to_label = dict()
        with open(root+"/"+annotations, "r") as fin:
            for line in fin.readlines():
                sp = line.split("\t")
                file_to_label[sp[0]] = class_to_idx[sp[1]]
        return file_to_label
   
    def generate_samples(_imgs, _file_to_label):
        new_imgs = list()
        for img_path, _ in _imgs:
            sp = img_path.split("/")
            new_imgs.append((img_path, _file_to_label[sp[-1]]))
        return new_imgs

    def __getitem__(self, index):
        def transform(img_path):
            img = self.loader(img_path)
            return self.transform(img)

        img_path, label = self.samples[index]
        img = transform(img_path)
        return img, label, img_path

    def __len__(self):
        return len(self.samples)
