import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.parameter import Parameter


class ULC(nn.Module):
    def __init__(self):
        super(ULC, self).__init__()
        emotion = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512)]

        calm = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                nn.Linear(512, 512)]
        
        fusion_flow = [nn.Linear(1024, 1024), nn.LeakyReLU(0.2, True),
                       nn.Linear(1024, 512), nn.LeakyReLU(0.2, True),
                       nn.Linear(512, 106 * 2)]

        self.emotion = nn.Sequential(*emotion)
        self.calm = nn.Sequential(*calm)
        self.fusion_flow = nn.Sequential(*fusion_flow)

    def forward(self, s, t):
        s_hid = self.emotion(s)
        c_hid = self.calm(t)
        fusion = torch.cat([s_hid, c_hid], dim=1)
        fusion_flow = self.fusion_flow(fusion)
        out = t + fusion_flow
        return out, fusion_flow


class DiscriminatorReal(nn.Module):
    def __init__(self):
        super(DiscriminatorReal, self).__init__()
        layers = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 64), nn.LeakyReLU(0.2, True),
                  nn.Linear(64, 1)]

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)


class DiscriminatorDiv(nn.Module):
    def __init__(self):
        super(DiscriminatorDiv, self).__init__()
        layers1 = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 64)]

        layers2 = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 64)]

        layers3 = [nn.Linear(128, 128), nn.LeakyReLU(0.2, True),
                   nn.Linear(128, 32), nn.LeakyReLU(0.2, True),
                   nn.Linear(32, 1)]

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)

    def forward(self, input1, input2):
        x1 = self.layers1(input1)
        x2 = self.layers2(input2)
        x_cat = torch.cat([x1, x2], dim=1)
        out = self.layers3(x_cat)
        return out


if __name__ == "__main__":
    from thop import profile
    torch.cuda.set_device(0)
    input = torch.randn(1, 212).cuda()
    net = ULC().cuda()
    flops, params = profile(net, inputs=(input, input))
    print('flops: {}   params: {}'.format(flops, params))
    # out = net(input)
