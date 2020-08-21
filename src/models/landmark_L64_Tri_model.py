import torch
import torch.nn as nn
import random
from .base_model import BaseModel
from . import networks
from . import vgg as per


class LandmarkL64TriModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='resnet_9blocks_cat', dataset_mode='RaFD90L64Tri')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_GAN', 'G_L1', 'G_Tri', 'D_real', 'D_fake']
        self.visual_names = ['A1', 'A2', 'B1', 'A1A2', 's_A1A2', 's_B1A1', 's_A2B1', 's_All']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        self.per = per.vgg16(pretrained=True)
        self.per.to(opt.gpu_ids[0])
        self.set_requires_grad(self.per, False)

        if self.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = nn.L1Loss()
            self.criterionPerceptual = nn.MSELoss()
            self.criterionClassification = nn.CrossEntropyLoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam([{'params': self.netG.module.model1.parameters()},
            #                                      {'params': self.netG.module.model2.parameters()},
            #                                      {'params': self.netG.module.model2_AdaIN.parameters()},
            #                                      {'params': self.netG.module.model3.parameters()},
            #                                      {'params': self.netG.module.model_l.parameters(), 'lr': opt.lr * 0.1}],
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.A1 = input['A1'].to(self.device)
        if self.isTrain:
            self.A2 = input['A2'].to(self.device)
            self.B1 = input['B1'].to(self.device)

        self.l_A1A2 = input['l_A1A2'].to(self.device)
        if self.isTrain:
            self.l_B1A1 = input['l_B1A1'].to(self.device)
            self.l_A2B1 = input['l_A2B1'].to(self.device)
            self.l_A1 = input['l_A1'].to(self.device)
            self.l_A2 = input['l_A2'].to(self.device)
            self.l_B1 = input['l_B1'].to(self.device)

            self.label_id = input['label_id']
            self.label_id = [id.squeeze(1).long().to(self.device) for id in self.label_id]
            self.label_emotion = input['label_emotion']
            self.label_emotion = [emotion.squeeze(1).long().to(self.device) for emotion in self.label_emotion]

            self.lab_black = input['lab_black'].to(self.device)
            self.img_black = input['img_black'].to(self.device)

            # for show
            self.s_A1A2 = torch.nn.Upsample(scale_factor=4)(self.l_A1A2)
            self.s_B1A1 = torch.nn.Upsample(scale_factor=4)(self.l_B1A1)
            self.s_A2B1 = torch.nn.Upsample(scale_factor=4)(self.l_A2B1)
            self.s_All = torch.cat([self.l_A1A2[:, 0, :, :].unsqueeze(1), self.l_B1A1[:, 0, :, :].unsqueeze(1), self.l_A2B1[:, 0, :, :].unsqueeze(1)], dim=1)
            self.s_All = torch.nn.Upsample(scale_factor=4)(self.s_All)

    def forward(self):

        self.A1A2 = self.netG(self.A1, self.l_A1A2)  # G(A)

        if self.isTrain:
            self.A2B1 = self.netG(self.B1, self.l_A2B1)  # G(A)
            self.B1A1 = self.netG(self.A1, self.l_B1A1)  # G(A)
            # perceptual
            self.f_A1A2 = self.per(self.A1A2)
            self.f_A2B1 = self.per(self.A2B1)
            self.f_B1A1 = self.per(self.B1A1)

            # idt_black
            # self.A1_black = self.netG(self.img_black, self.l_A1A2)

            # idt_weak
            # r_num = random.randint(1, 3)
            # self.A1_iw = self.netG(self.A1, self.lab_black)

            # idt
            # self.A1_fake = self.netG(self.A1, self.l_A1)
            # self.A2_fake = self.netG(self.A2, self.l_A2)
            # self.B1_fake = self.netG(self.B1, self.l_B1)

    def backward_D(self):
        lambda_D = 1
        # fake
        fake_A1A2 = torch.cat((self.A1, self.A1A2), 1)
        pred_fake = self.netD(fake_A1A2.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_A1A2 = torch.cat((self.A1, self.A2), 1)
        pred_real = self.netD(real_A1A2)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * lambda_D
        self.loss_D.backward()

    def backward_triplet_perceptual(self):
        loss_P = 0
        for i in range(0, len(self.f_A1A2)-1):
            # loss_1 = self.criterionPerceptual(self.f_A1A2[i], self.f_A1B1[i])
            # loss_2 = -self.criterionPerceptual(self.f_A1A2[i], self.f_B1A2[i]) * 0.5
            # loss_3 = -self.criterionPerceptual(self.f_A1B1[i], self.f_B1A2[i]) * 0.5
            # loss_P += (loss_1 + loss_2 + loss_3)
            loss_1 = self.criterionPerceptual(self.f_B1A1[i], self.f_A1A2[i])
            loss_2 = -self.criterionPerceptual(self.f_A2B1[i], self.f_A1A2[i])
            if float(loss_1 + loss_2 + 0.3) > 0:
                loss_P += (loss_1 + loss_2)
        return loss_P

    def backward_idt(self):
        loss_1 = self.criterionL1(self.A1, self.A1_fake)
        loss_2 = self.criterionL1(self.A2, self.A2_fake)
        loss_3 = self.criterionL1(self.B1, self.B1_fake)
        loss_I = (loss_1 + loss_2 + loss_3) / 3

        return loss_I

    def backward_classification(self):
        A1A2_id, A1A2_emotion = self.CNet(self.A1A2)
        A1B1_id, A1B1_emotion = self.CNet(self.A1B1)
        B1A2_id, B1A2_emotion = self.CNet(self.B1A2)
        # id
        loss_id_1 = self.criterionClassification(A1A2_id, self.label_id[0])
        loss_id_2 = self.criterionClassification(A1B1_id, self.label_id[0])
        loss_id_3 = self.criterionClassification(B1A2_id, self.label_id[2])
        loss_id = (loss_id_1 + loss_id_2 + loss_id_3) / 3
        # emotion
        loss_emotion_1 = self.criterionClassification(A1A2_emotion, self.label_emotion[1])
        loss_emotion_2 = self.criterionClassification(A1B1_emotion, self.label_emotion[2])
        loss_emotion_3 = self.criterionClassification(B1A2_emotion, self.label_emotion[1])
        loss_emotion = (loss_emotion_1 + loss_emotion_2 + loss_emotion_3) / 3
        # all
        loss_C = (loss_id + loss_emotion) / 2

        return loss_C

    def backward_G(self):
        lambda_GAN = 0.1
        lambda_L1 = 100
        lambda_Tri = 0.1
        lambda_idt = 1
        lambda_cls = 0.01
        lambda_black_iw = 100

        fake_A1A2 = torch.cat((self.A1, self.A1A2), 1)
        pred_fake = self.netD(fake_A1A2)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.A1A2, self.A2)
        self.loss_G_Tri = self.backward_triplet_perceptual()
        # self.loss_I = self.backward_idt()
        # self.loss_C = self.backward_classification() * weight_C

        # idt_black & idt_weak
        # self.loss_B = self.criterionL1(self.A1_black, self.img_black)
        # self.loss_W = self.criterionL1(self.A1_iw, self.A1)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN * lambda_GAN + self.loss_G_L1 * lambda_L1 + self.loss_G_Tri * lambda_Tri
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
