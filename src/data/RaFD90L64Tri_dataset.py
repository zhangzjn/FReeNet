import os
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import collections
from torch.utils.data import sampler
import torch
from torch.utils.data import dataloader
from torch.utils.data.sampler import  WeightedRandomSampler
import cv2
from ULC.net_ULC import *
import numpy as np


class RandomSampler(sampler.Sampler):
    def __init__(self, data_source, batch_id, batch_image):
        super(RandomSampler, self).__init__(data_source)

        self.data_source = data_source
        self.batch_id = batch_id
        self.batch_image = batch_image

        self._id2index = collections.defaultdict(list)
        for idx, path in enumerate(data_source.labels):
            _id = data_source.id(os.path.basename(path[0]))
            self._id2index[_id].append(idx)

        self.id_valid = len(self._id2index) // self.batch_image * self.batch_image

    def __iter__(self):
        unique_ids = self.data_source.unique_ids
        random.shuffle(unique_ids)

        imgs = []
        for i in range(self.id_valid):
            imgs.extend(self._sample(self._id2index[unique_ids[i]], self.batch_image * 16))
        return iter(imgs)

    def __len__(self):
        return self.id_valid * self.batch_image * 16

    @staticmethod
    def _sample(population, k):
        if len(population) < k:
            population = population * k
        return random.sample(population, k)


class RaFD90L64TriDataset(BaseDataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        root = '{}/RaFD'.format(opt.dataroot)
        self.id2calm = dict()
        self.shapes = list()
        self.name2shape = dict()
        self.id2names = dict()

        data_names = ['RaFD45', 'RaFD90', 'RaFD135']
        for data_name in data_names:
            data_dir = '{}/{}'.format(root, data_name)
            landmark_path = '{}/landmark_crop.txt'.format(data_dir)
            image_dir = '{}/image_crop'.format(data_dir)
            label_dir = '{}/landmark_crop'.format(data_dir)
            labs = open(landmark_path, 'r').readlines()
            for i, lab in enumerate(labs):
                name, shape = self.parse_lab(lab)
                path = os.path.join(image_dir, name)
                path_lab = os.path.join(label_dir, name)
                id = self.id(name)
                if name.find('neutral_frontal') != -1 and name.find('Rafd090') != -1:
                    self.id2calm[id] = [path, shape, id]
                self.shapes.append([path, shape, id])
                if id not in self.id2names.keys():
                    self.id2names[id] = [[path, shape, id]]
                else:
                    self.id2names[id].append([path, shape, id])
                # self.id2names.get(id, []).append([[name, shape, id]])
                self.name2shape[name] = shape
        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        self._emotion2label = {_emotion: idx for idx, _emotion in enumerate(self.unique_emotions)}
        self.uids = self.unique_ids
        # LSNet
        self.net = ULC()
        self.net.load_state_dict(torch.load('ULC/checkpoints/RaFD/best.pth', {'cuda:0': 'cuda:{}'.format(opt.gpu_ids[0])})['net_G'])
        self.net.eval()
        # self.net.apply(weight_init)


        # transform
        # add transforms.RandomHorizontalFlip() later
        self.transforms_image = transforms.Compose([transforms.Resize([256, 256], Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transforms_label = transforms.Compose([transforms.Resize([64, 64], Image.BICUBIC),

                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.img_size = 512
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.shapes)

    def __getitem__(self, index):
        A1_path, A1_shape, A_id = self.shapes[index]
        names = self.id2names[A_id]
        A2_path, A2_shape, A_id = random.sample(names, 1)[0]
        diff = set(self.uids) - set([A_id])
        B_id = random.sample(diff, 1)[0]
        B1_path, B1_shape, B_id = random.sample(self.id2names[B_id], 1)[0]

        A_shape_calm = self.id2calm[A_id][1]
        B_shape_calm = self.id2calm[B_id][1]

        A1_shape = torch.Tensor(A1_shape)
        A2_shape = torch.Tensor(A2_shape)
        B1_shape = torch.Tensor(B1_shape)
        A_shape_calm = torch.Tensor(A_shape_calm)
        B_shape_calm = torch.Tensor(B_shape_calm)

        input1 = torch.cat([B1_shape.unsqueeze(0), A2_shape.unsqueeze(0)], dim=0)
        input2 = torch.cat([A_shape_calm.unsqueeze(0), B_shape_calm.unsqueeze(0)], dim=0)
        output = self.net(input1, input2)

        A1A2_l = A2_shape
        B1A1_l = output[0][0]
        A2B1_l = output[0][1]

        lab_template = np.zeros((self.img_size, self.img_size, 3))
        A1A2_lab = self.drawCircle(lab_template.copy(), A1A2_l.data, radius=1, color=(255, 255, 255), thickness=8)
        B1A1_lab = self.drawCircle(lab_template.copy(), B1A1_l.data, radius=1, color=(255, 255, 255), thickness=8)
        A2B1_lab = self.drawCircle(lab_template.copy(), A2B1_l.data, radius=1, color=(255, 255, 255), thickness=8)
        A1_lab = self.drawCircle(lab_template.copy(), A1_shape.data, radius=1, color=(255, 255, 255), thickness=8)
        A2_lab = self.drawCircle(lab_template.copy(), A2_shape.data, radius=1, color=(255, 255, 255), thickness=8)
        B1_lab = self.drawCircle(lab_template.copy(), B1_shape.data, radius=1, color=(255, 255, 255), thickness=8)
        A1A2_lab = Image.fromarray(np.uint8(A1A2_lab)).convert('RGB')
        B1A1_lab = Image.fromarray(np.uint8(B1A1_lab)).convert('RGB')
        A2B1_lab = Image.fromarray(np.uint8(A2B1_lab)).convert('RGB')
        A1_lab = Image.fromarray(np.uint8(A1_lab)).convert('RGB')
        A2_lab = Image.fromarray(np.uint8(A2_lab)).convert('RGB')
        B1_lab = Image.fromarray(np.uint8(B1_lab)).convert('RGB')

        img_A1 = Image.open(A1_path).convert('RGB')
        img_A2 = Image.open(A2_path).convert('RGB')
        img_B1 = Image.open(B1_path).convert('RGB')

        img_A1 = self.transforms_image(img_A1)
        img_A2 = self.transforms_image(img_A2)
        img_B1 = self.transforms_image(img_B1)
        lab_A1A2 = self.transforms_label(A1A2_lab)
        lab_B1A1 = self.transforms_label(B1A1_lab)
        lab_A2B1 = self.transforms_label(A2B1_lab)
        A1_lab = self.transforms_label(A1_lab)
        A2_lab = self.transforms_label(A2_lab)
        B1_lab = self.transforms_label(B1_lab)

        # black label
        black_template = np.zeros((self.img_size, self.img_size, 3))
        lab_black = Image.fromarray(np.uint8(black_template.copy())).convert('RGB')
        lab_black = self.transforms_label(lab_black)
        # black img
        img_black = Image.fromarray(np.uint8(black_template.copy())).convert('RGB')
        img_black = self.transforms_image(img_black)

        # img_id = self._id2label[A_id]
        # img_id = torch.Tensor([img_id]).long()
        label_id = [self.id(os.path.basename(A1_path)), self.id(os.path.basename(A2_path)), self.id(os.path.basename(B1_path))]
        label_id = [torch.Tensor([self._id2label[_]]) for _ in label_id]
        label_emotion = [self.emotion(os.path.basename(A1_path)), self.emotion(os.path.basename(A2_path)), self.emotion(os.path.basename(B1_path))]
        label_emotion = [torch.Tensor([self._emotion2label[_]]) for _ in label_emotion]

        return {'A1': img_A1, 'A2': img_A2, 'B1': img_B1, 'l_A1A2': lab_A1A2, 'l_B1A1': lab_B1A1, 'l_A2B1': lab_A2B1,
                'l_A1': A1_lab, 'l_A2': A2_lab, 'l_B1': B1_lab, 'label_id': label_id, 'label_emotion': label_emotion,
                'lab_black': lab_black, 'img_black': img_black}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.shapes)

    @staticmethod
    def emotion(img_path):
        return img_path.split('_')[-2]

    @property
    def emotions(self):
        return [self.emotion(os.path.basename(shape[0])) for shape in self.shapes]

    @property
    def unique_emotions(self):
        return sorted(set(self.emotions))

    @staticmethod
    def id(img_path):
        return int(img_path.split('_')[1])

    @property
    def ids(self):
        return [self.id(os.path.basename(shape[0])) for shape in self.shapes]

    @property
    def unique_ids(self):
        return sorted(set(self.ids))

    def parse_lab(self, lab):
        l = lab.strip().split()
        name = l[0]
        w_ori, h_ori = [int(_) for _ in l[1].split('-')]
        shape = []
        for l_ in l[2:]:
            w, h = [float(_) for _ in l_.split('-')]
            shape.extend([w, h])
        return name, shape

    def drawCircle(self, img, shape, radius=1, color=(255, 255, 255), thickness=1):
        for i in range(len(shape)//2):
            img = cv2.circle(img, (int(shape[2*i]), int(shape[2*i+1])), radius, color, thickness)
        return img

if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    train_dataset = RaFD90L64TriDataset(opt)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=4*4, num_workers=1)
    dataset_size = len(train_dataset)
    print(dataset_size)
    for i, data in enumerate(train_loader):
        print(i)
