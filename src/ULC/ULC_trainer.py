import torch
import torch.nn as nn
from net_ULC import *
from utils import *
from loss import *
import cv2
import numpy as np


class ULC_trainer():

    def __init__(self, opt):

        self.gpus = opt.gpus[0]
        self.isTrain = opt.isTrain
        self.lr = opt.lr
        self.every = opt.every
        self.save_every = opt.save_every
        self.checkpoints = opt.checkpoints
        self.img_size = opt.img_size
        self.logdir = os.path.join(self.checkpoints, opt.name)
        os.makedirs(self.logdir, exist_ok=True)

        self.epoch = 0

        self.netG = ULC()
        self.netG.apply(weight_init)
        if opt.resume:
            checkpoint = torch.load('{}/{}.pth'.format(self.logdir, opt.resume_epoch if opt.resume_epoch else 'best'))
            self.netG.load_state_dict(checkpoint['net_G'])
            self.epoch = checkpoint['epoch']
        self.netG.cuda()

        self.best_loss = 100

        if self.isTrain:
            self.netD = DiscriminatorDiv()
            self.netDr = DiscriminatorReal()
            self.netD.apply(weight_init)
            self.netDr.apply(weight_init)
            if opt.resume:
                self.netD.load_state_dict(checkpoint['net_D'])
                self.netDr.load_state_dict(checkpoint['net_Dr'])
            self.netD.cuda()
            self.netDr.cuda()

        if self.isTrain:
            self.criterionGAN = GANLoss(gan_mode='mse').cuda()
            self.criterionCycle = nn.L1Loss()
            self.criterionIdt = nn.L1Loss()
            self.criterionL1 = nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.99, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.99, 0.999))
            self.optimizer_Dr = torch.optim.Adam(self.netDr.parameters(), lr=self.lr, betas=(0.99, 0.999))

    def train(self):
        self.isTrain = True

    def eval(self):
        self.isTrain = False

    def reset(self):
        self.loss_log_G_A = 0
        self.loss_log_G_A_Dr = 0
        self.loss_log_cycle_A = 0
        self.loss_log_idt_A = 0
        self.loss_log_D_real = 0
        self.loss_log_D_fake = 0
        self.loss_log_Dr_real = 0
        self.loss_log_Dr_fake = 0
        self.loss_log_L1 = 0

    def test_draw(self, dataloader):
        def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
            for i in range(len(shape) // 2):
                img = cv2.circle(img, (int(shape[2 * i]), int(shape[2 * i + 1])), radius, color, thickness)
            return img

        def drawArrow(img, shape1, shape2, ):
            for i in range(len(shape1) // 2):
                point1 = (int(shape1[2 * i]), int(shape1[2 * i + 1]))
                point2 = (int(shape1[2 * i] + shape2[2 * i]), int(shape1[2 * i + 1] + shape2[2 * i + 1]))
                img = cv2.circle(img, point2, radius=6, color=(0, 0, 255), thickness=2)
                img = cv2.line(img, point1, point2, (255, 255, 255), thickness=2)
            return img

        test_one = False
        if test_one:
            def parse_lab(lab):
                l = lab.strip().split()
                name = l[0]
                w_ori, h_ori = [int(_) for _ in l[1].split('-')]
                shape = []
                for l_ in l[2:108]:
                    w, h = [float(_) for _ in l_.split('-')]
                    shape.extend([w, h])
                ori_gaze = []
                for l_ in l[108:]:
                    ori_gaze.append(float(l_))

                return name, shape, ori_gaze

            root = self.logdir
            s_path = 'dataset/test/results'
            os.makedirs(root, exist_ok=True)
            os.makedirs(s_path, exist_ok=True)
            with torch.no_grad():
                lab_s = open('dataset/test/source/landmark_crop.txt', 'r').readlines()[0]
                lab_t = open('dataset/test/target/landmark_crop.txt', 'r').readlines()[0]
                name_A, shape_A, ori_gaze_A = parse_lab(lab_s)
                name_B, shape_B, ori_gaze_B = parse_lab(lab_t)
                shape_A = torch.Tensor(shape_A).unsqueeze(0)
                shape_B = torch.Tensor(shape_B).unsqueeze(0)
                shape = [shape_A, shape_B, shape_B, shape_A, shape_B]
                self.set_input(shape)
                self.forward()
                img_template = np.zeros((self.img_size, self.img_size, 3))
                img_A = drawCircle(img_template.copy(), self.A.squeeze(0).data, radius=1, color=(255, 255, 255), thickness=2)
                img_B_pre = drawCircle(img_template.copy(), self.fake_B.squeeze(0).data, radius=1, color=(255, 255, 255), thickness=2)
                img_B = drawCircle(img_template.copy(), self.B.squeeze(0).data, radius=1, color=(255, 255, 255), thickness=2)
                img_flow = drawArrow(img_template.copy(), self.B.squeeze(0).data, self.fusion_flow_AB.squeeze(0).data)

                img_compare1 = np.concatenate([img_A[:, :, 0][:, :, np.newaxis], img_B[:, :, 0][:, :, np.newaxis],
                                               img_B_pre[:, :, 0][:, :, np.newaxis]], axis=2)
                img_compare2 = img_flow
                img_compare3 = np.concatenate([img_A[:, :, 0][:, :, np.newaxis], img_B[:, :, 0][:, :, np.newaxis],
                                               img_flow[:, :, 2][:, :, np.newaxis]], axis=2)

                cv2.imwrite('{}/{}.jpg'.format(s_path, 1), img_compare1)
                cv2.imwrite('{}/{}.jpg'.format(s_path, 2), img_compare2)
                cv2.imwrite('{}/{}.jpg'.format(s_path, 3), img_compare3)

        else:
            root = self.logdir
            s_path1 = '{}/result1'.format(root)
            s_path2 = '{}/result2'.format(root)
            s_path3 = '{}/result3'.format(root)
            os.makedirs(s_path1, exist_ok=True)
            os.makedirs(s_path2, exist_ok=True)
            os.makedirs(s_path3, exist_ok=True)
            with torch.no_grad():
                for batch_idx, shape in enumerate(dataloader):
                    self.set_input(shape)
                    self.forward()
                    img_template = np.zeros((self.img_size, self.img_size, 3))
                    img_A = drawCircle(img_template.copy(), self.A.squeeze(0).data, radius=1, color=(255, 255, 255), thickness=2)
                    img_B_pre = drawCircle(img_template.copy(), self.fake_B.squeeze(0).data, radius=1, color=(255, 255, 255), thickness=2)
                    img_B = drawCircle(img_template.copy(), self.B.squeeze(0).data, radius=1, color=(255, 255, 255), thickness=2)
                    img_flow = drawArrow(img_template.copy(), self.B.squeeze(0).data, self.fusion_flow_AB.squeeze(0).data)

                    img_compare1 = np.concatenate([img_A[:, :, 0][:, :, np.newaxis], img_B[:, :, 0][:, :, np.newaxis],
                                                   img_B_pre[:, :, 0][:, :, np.newaxis]], axis=2)
                    img_compare2 = img_flow
                    img_compare3 = np.concatenate([img_A[:, :, 0][:, :, np.newaxis], img_B[:, :, 0][:, :, np.newaxis],
                                                   img_flow[:, :, 2][:, :, np.newaxis]], axis=2)
                    cv2.imwrite('{}/{}.jpg'.format(s_path1, batch_idx), img_compare1)
                    cv2.imwrite('{}/{}.jpg'.format(s_path2, batch_idx), img_compare2)
                    cv2.imwrite('{}/{}.jpg'.format(s_path3, batch_idx), img_compare3)
                    print('\r{}/{}'.format(batch_idx + 1, len(dataloader)), end='')

    def run(self, dataloader, epoch=None):
        if not self.isTrain:
            self.test_draw(dataloader)
        else:
            self.epoch += 1
            if epoch:
                self.epoch = epoch
            self.reset()
            adjust_learning_rate(self.optimizer_G, self.lr, self.epoch, every=self.every)
            adjust_learning_rate(self.optimizer_D, self.lr, self.epoch, every=self.every)
            adjust_learning_rate(self.optimizer_Dr, self.lr, self.epoch, every=self.every)
            for batch_idx, shape in enumerate(dataloader):
                self.batch_idx = batch_idx + 1
                self.set_input(shape)
                self.optimize_parameters()
                log_string = '\r'
                log_string += 'epoch {:>5} '.format(self.epoch)
                log_string += 'batch {:>3}/{} '.format(batch_idx + 1, len(dataloader))
                log_string += ' | L_G_A {:>6.3f}'.format(self.loss_log_G_A / (batch_idx + 1))
                log_string += ' | L_D_real {:>6.3f}'.format(self.loss_log_D_real / (batch_idx + 1))
                log_string += ' | L_D_fake {:>6.3f}'.format(self.loss_log_D_fake / (batch_idx + 1))

                log_string += ' | L_G_A_Dr {:>6.3f}'.format(self.loss_log_G_A_Dr / (batch_idx + 1))
                log_string += ' | L_Dr_real {:>6.3f}'.format(self.loss_log_Dr_real / (batch_idx + 1))
                log_string += ' | L_Dr_fake {:>6.3f}'.format(self.loss_log_Dr_fake / (batch_idx + 1))

                log_string += ' | L_L1 {:>6.3f}'.format(self.loss_log_L1 / (batch_idx + 1))
                log_string += ' | L_cycle_A {:>6.3f}'.format(self.loss_log_cycle_A / (batch_idx + 1))
                log_string += ' | L_idt_A {:>6.3f}'.format(self.loss_log_idt_A / (batch_idx + 1))

                print(log_string, end='')
            if self.loss_log_L1 / (batch_idx + 1) < self.best_loss:
                self.best_loss = self.loss_log_L1 / (batch_idx + 1)
                print('  ==> best_L1_loss {:.5f}'.format(self.best_loss))
                self.save(mark='best')
            if self.epoch % self.save_every == 0:
                self.save()

    def set_input(self, shape):
        self.A, self.B, self.B1, self.A_calm, self.B_calm = \
            shape[0].to(self.gpus), shape[1].to(self.gpus), shape[2].to(self.gpus), shape[3].to(self.gpus), shape[4].to(self.gpus)

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD, self.netDr], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        if self.batch_idx % 1 == 0:
            self.set_requires_grad([self.netD, self.netDr], True)
            self.optimizer_D.zero_grad()
            self.optimizer_Dr.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            self.optimizer_Dr.step()

    def forward(self):
        self.fake_B, self.fusion_flow_AB = self.netG(self.A, self.B_calm)  # G_A(A)
        self.rec_A, self.fusion_flow_BA = self.netG(self.fake_B, self.A_calm)   # G_B(G_A(A))

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_G(self):
        lambda_gan = 0.1
        lambda_gan_Dr = 0.1
        lambda_idt = 1
        lambda_cycle = 10
        lambda_L1 = 100

        # Identity loss
        if lambda_idt > 0:
            self.idt_A, self.fusion_flow = self.netG(self.A, self.A_calm)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.A)
        else:
            self.loss_idt_A = 0

        # GAN loss D_A(G_A(A))
        # fake_AB = torch.cat([self.fake_B, self.B], dim=1)
        self.loss_G_A = self.criterionGAN(self.netD(self.fake_B, self.B1), True)
        self.loss_G_A_Dr = self.criterionGAN(self.netDr(self.fake_B), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.A)
        # L1
        self.loss_L1 = self.criterionL1(self.fake_B, self.B)
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A * lambda_gan + self.loss_G_A_Dr * lambda_gan_Dr + \
                      self.loss_idt_A * lambda_cycle * lambda_idt + self.loss_cycle_A * lambda_cycle + self.loss_L1 * lambda_L1

        self.loss_G.backward()
        # log
        self.loss_log_G_A += self.loss_G_A.item()
        self.loss_log_G_A_Dr += self.loss_G_A_Dr.item()
        self.loss_log_cycle_A += self.loss_cycle_A.item()
        self.loss_log_idt_A += self.loss_idt_A.item() if lambda_idt > 0 else 0
        self.loss_log_L1 += self.loss_L1.item()

    def backward_D(self):
        """Calculate GAN loss for discriminator D_A"""
        lambda_D = 0.1
        lambda_Dr = 1
        pred_real = self.netD(self.B.detach(), self.B1.detach())
        loss_D_real = self.criterionGAN(pred_real, True)
        pred_fake = self.netD(self.fake_B.detach(), self.B1.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_Dr_real = self.criterionGAN(self.netDr(self.B.detach()), True)
        loss_Dr_fake = self.criterionGAN(self.netDr(self.fake_B.detach()), False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5 * lambda_D + (loss_Dr_real + loss_Dr_fake) * 0.5 * lambda_Dr
        loss_D.backward()
        self.loss_log_D_real += loss_D_real.item()
        self.loss_log_D_fake += loss_D_fake.item()
        self.loss_log_Dr_real += loss_Dr_real.item()
        self.loss_log_Dr_fake += loss_Dr_fake.item()

    def save(self, mark=None):
        state = {
            'net_G': self.netG.state_dict(),
            'net_D': self.netD.state_dict(),
            'net_Dr': self.netDr.state_dict(),
            'epoch': self.epoch,
        }
        if mark is not None:
            torch.save(state, '{}/{}.pth'.format(self.logdir, mark))
        else:
            torch.save(state, '{}/{}.pth'.format(self.logdir, self.epoch))
