import os
from options.test_options import TestOptions
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import sys
from util.util import *
import random
from data.RaFD90L64Tri_dataset import *
from models.landmark_L64_Tri_model import *
from ULC.net_ULC import *

def parse_lab(lab):
    l = lab.strip().split()
    name = l[0]
    w_ori, h_ori = [int(_) for _ in l[1].split('-')]
    shape = []
    for l_ in l[2:]:
        w, h = [float(_) for _ in l_.split('-')]
        shape.extend([w, h])
    return name, shape


def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
    for i in range(len(shape) // 2):
        img = cv2.circle(img, (int(shape[2 * i]), int(shape[2 * i + 1])), radius, color, thickness)
    return img


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.isTrain = False
    # opt.name = 'RaFD-08-16'
    # opt.model = 'landmark_L64_Tri'
    # opt.netG = 'resnet_9blocks_cat'
    # opt.dataset_mode = 'RaFD90L64Tri'
    # opt.gpu_ids = [1]
    model = LandmarkL64TriModel(opt)
    model.setup(opt)
    model.eval()

    transforms_image = transforms.Compose([transforms.Resize([256, 256], Image.BICUBIC),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transforms_label = transforms.Compose([transforms.Resize([64, 64], Image.BICUBIC),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # load ULC
    net = ULC()
    net.load_state_dict(torch.load('ULC/checkpoints/RaFD/best.pth', {'cuda:0': 'cuda:{}'.format(opt.gpu_ids[0])})['net_G'])
    net.eval()

    root = 'evaluation/RaFD90'
    t_persons = ['calm1', 'calm2']
    s_persons = ['target1', 'target2']

    save_dir = '{}/results'.format(root)
    os.makedirs(save_dir, exist_ok=True)
    i = 0
    for t_person in t_persons:
        for s_person in s_persons:
            # image
            img_t_path = '{}/{}/1.jpg'.format(root, t_person)
            img_s_path = '{}/{}/1.jpg'.format(root, s_person)
            lab_t_path = '{}/{}/landmark_calm.txt'.format(root, t_person)
            lab_s_path = '{}/{}/landmark.txt'.format(root, s_person)
            lab_t = open(lab_t_path, 'r').readlines()[0]
            lab_s = open(lab_s_path, 'r').readlines()[0]
            name_t, shape_t = parse_lab(lab_t)
            name_s, shape_s = parse_lab(lab_s)
            shape_t = torch.Tensor(shape_t)
            shape_s = torch.Tensor(shape_s)

            input_s = shape_s.unsqueeze(0)
            input_t = shape_t.unsqueeze(0)
            output = net(input_s, input_t)

            shape_st = output[0][0]

            img_size = 512
            lab_template = np.zeros((img_size, img_size, 3))
            lab_st = drawCircle(lab_template.copy(), shape_st.data, radius=1, color=(255, 255, 255), thickness=8)
            lab_st = Image.fromarray(np.uint8(lab_st)).convert('RGB')
            lab_st = transforms_label(lab_st).unsqueeze(0)
            img_t = Image.open(img_t_path).convert('RGB')
            img_t = transforms_image(img_t).unsqueeze(0)

            input_data = {'A1': img_t, 'l_A1A2': lab_st}
            model.set_input(input_data)
            model.test()
            img_trans = tensor2im(model.A1A2)

            cv2.imwrite('{}/{}-{}.jpg'.format(save_dir, s_person, t_person), cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB))
            i += 1
            print('{}/{}'.format(i, len(t_persons)*len(s_persons)))
