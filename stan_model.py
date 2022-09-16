import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchviz import make_dot

class encoder_block(nn.Module):
    def __init__(self, input_channels_1, input_channels_2, output_channels):
        super(encoder_block, self).__init__()
        self.conv5 = nn.Conv2d(in_channels=input_channels_2, out_channels=output_channels, kernel_size=5, padding='same')
        self.conv1 = nn.Conv2d(in_channels=input_channels_2, out_channels=output_channels, kernel_size=1, padding='same')
        self.conv3_1 = nn.Conv2d(in_channels=input_channels_1, out_channels=output_channels, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding='same')
        self.maxp = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x_1, x_2 = x
        # print("x_1 : ", x_1.shape)
        # print("x_2 : ", x_2.shape)
        x_2_5 = self.conv5(x_2)
        x_2_5 = self.relu(x_2_5)
        x_2_1 = self.conv1(x_2)
        x_2_1 = self.relu(x_2_1)
        x_1_3 = self.conv3_1(x_1)
        x_1_3 = self.relu(x_1_3)
        # print("x_2_5 : ", x_2_5.shape)
        # print("x_2_1 : ", x_2_1.shape)
        # print("x_1_3 : ", x_1_3.shape)
        x_2_5_3 = self.conv3_2(x_2_5)
        x_2_5_3 = self.relu(x_2_5_3)
        x_2_1_3 = self.conv3_2(x_2_1)
        x_2_1_3 = self.relu(x_2_1_3)
        x_1_3_3 = self.conv3_2(x_1_3)
        x_1_3_3 = self.relu(x_1_3_3)
        # print("x_2_5_3 : ", x_2_5_3.shape)
        # print("x_2_1_3 : ", x_2_1_3.shape)
        # print("x_1_3_3 : ", x_1_3_3.shape)
        concat = torch.cat((x_2_5_3, x_2_1_3), dim=1)
        # print("concat : ", concat.shape)
        concat_pool = self.maxp(concat)
        x_1_3_3_pool = self.maxp(x_1_3_3)
        skip1 = x_1_3_3
        skip2 = torch.cat((concat, x_1_3), dim=1)
        # print("out1 : ", x_1_3_3_pool.shape)
        # print("out2 : ", concat_pool.shape)
        # print("skip1 : ", skip1.shape)
        # print("skip2 : ", skip2.shape)

        return x_1_3_3_pool, concat_pool, skip1, skip2

class middle_block(nn.Module):
    def __init__(self, input_channels_1, input_channels_2, output_channels):
        super(middle_block, self).__init__()
        self.conv5 = nn.Conv2d(in_channels=input_channels_2, out_channels=output_channels, kernel_size=5, padding='same')
        self.conv1 = nn.Conv2d(in_channels=input_channels_2, out_channels=output_channels, kernel_size=1, padding='same')
        self.conv3_1 = nn.Conv2d(in_channels=input_channels_1, out_channels=output_channels, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding='same')
        self.maxp = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_1, x_2 = x
        # print("x_1 : ", x_1.shape)
        # print("x_2 : ", x_2.shape)
        x_2_5 = self.conv5(x_2)
        x_2_5 = self.relu(x_2_5)
        x_2_1 = self.conv1(x_2)
        x_2_1 = self.relu(x_2_1)
        x_1_3 = self.conv3_1(x_1)
        x_1_3 = self.relu(x_1_3)
        # print("x_2_5 : ", x_2_5.shape)
        # print("x_2_1 : ", x_2_1.shape)
        # print("x_1_3 : ", x_1_3.shape)
        x_2_5_3 = self.conv3_2(x_2_5)
        x_2_5_3 = self.relu(x_2_5_3)
        x_2_1_3 = self.conv3_2(x_2_1)
        x_2_1_3 = self.relu(x_2_1_3)
        x_1_3_3 = self.conv3_2(x_1_3)
        x_1_3_3 = self.relu(x_1_3_3)
        # print("x_2_5_3 : ", x_2_5_3.shape)
        # print("x_2_1_3 : ", x_2_1_3.shape)
        # print("x_1_3_3 : ", x_1_3_3.shape)
        concat = torch.cat((x_2_5_3, x_2_1_3, x_1_3_3), dim=1)
        # print("concat : ", concat.shape)
        return concat

class decoder_block(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(decoder_block, self).__init__()
        self.deconv3 = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=output_channels*2, out_channels=output_channels, kernel_size=3, padding='same')
        self.conv3_2 = nn.Conv2d(in_channels=output_channels*4, out_channels=output_channels, kernel_size=3, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        x_t, skip1, skip2 = x
        # print("x_t : ", x_t.shape)
        # print("skip1 : ", skip1.shape)
        # print("skip2 : ", skip2.shape)
        x_t_d = self.deconv3(x_t)
        # print("x_t_d : ", x_t_d.shape)
        x_t_d = torch.cat((x_t_d, skip1), dim=1)
        # print("x_t_d : ", x_t_d.shape)
        x_t_d_c = self.conv3_1(x_t_d)
        x_t_d_c = self.relu(x_t_d_c)
        # print("x_t_d_c : ", x_t_d_c.shape)
        x_t_d_c = torch.cat((x_t_d_c, skip2), dim=1)
        # print("x_t_d_c : ", x_t_d_c.shape)
        x_t_d_c_c = self.conv3_2(x_t_d_c)
        x_t_d_c_c = self.relu(x_t_d_c_c)
        # print("x_t_d_c_c : ", x_t_d_c_c.shape)

        return x_t_d_c_c

class stan_architecture(nn.Module):
    def __init__(self, initial_channels=3, filters=[32, 64, 128, 256, 512], final_channels=1):
        super(stan_architecture, self).__init__()
        self.enc1 = encoder_block(initial_channels, initial_channels, filters[0])
        self.enc2 = encoder_block(filters[0], filters[1], filters[1])
        self.enc3 = encoder_block(filters[1], filters[2], filters[2])
        self.enc4 = encoder_block(filters[2], filters[3], filters[3])
        self.mid = middle_block(filters[3], filters[4], filters[4])
        self.dec4 = decoder_block(filters[4]*3, filters[3])
        self.dec3 = decoder_block(filters[3], filters[2])
        self.dec2 = decoder_block(filters[2], filters[1])
        self.dec1 = decoder_block(filters[1], filters[0])
        self.conv = nn.Conv2d(in_channels=filters[0], out_channels=final_channels, kernel_size=3, padding='same')
        self.relu = nn.ReLU()
    
    def forward(self, input_image):
        x = input_image
        # print("input: ", x.shape)
        # print("********** Encoder 1 **********")
        e1_out1, e1_out2, e1_skip1, e1_skip2 = self.enc1((x, x))
        # print("e1_out1", e1_out1.shape)
        # print("e1_out2", e1_out2.shape)
        # print("e1_skip1", e1_skip1.shape)
        # print("e1_skip2", e1_skip2.shape)
        # print("********** Encoder 2 **********")
        e2_out1, e2_out2, e2_skip1, e2_skip2 = self.enc2((e1_out1, e1_out2))
        # print(e2_out1.shape)
        # print(e2_out2.shape)
        # print(e2_skip1.shape)
        # print(e2_skip2.shape)
        # print("********** Encoder 3 **********")
        e3_out1, e3_out2, e3_skip1, e3_skip2 = self.enc3((e2_out1, e2_out2))
        # print(e3_out1.shape)
        # print(e3_out2.shape)
        # print(e3_skip1.shape)
        # print(e3_skip2.shape)
        # print("********** Encoder 4 **********")
        e4_out1, e4_out2, e4_skip1, e4_skip2 = self.enc4((e3_out1, e3_out2))
        # print(e4_out1.shape)
        # print(e4_out2.shape)
        # print(e4_skip1.shape)
        # print(e4_skip2.shape)
        # print("********** Middle **********")
        mid_out = self.mid((e4_out1, e4_out2))
        # print(mid_out.shape)
        # print("********** Decoder 4 **********")
        d4_out = self.dec4((mid_out, e4_skip1, e4_skip2))
        # print("********** Decoder 3 **********")
        d3_out = self.dec3((d4_out, e3_skip1, e3_skip2))
        # print("********** Decoder 2 **********")
        d2_out = self.dec2((d3_out, e2_skip1, e2_skip2))
        # print("********** Decoder 1 **********")
        d1_out = self.dec1((d2_out, e1_skip1, e1_skip2))
        # print("********** Final **********")
        out = self.conv(d1_out)
        out = self.relu(out)
        # print(d4_out.shape)
        # print(d3_out.shape)
        # print(d2_out.shape)
        # print(d1_out.shape)
        # print("output : ", out.shape)
        return out

# model = stan_architecture(3, [32, 64, 128, 256, 512], 1).cuda()
# x = torch.zeros(1, 3, 256, 256, dtype=torch.float, requires_grad=False).cuda()
# print(model)
# # print(summary(model, (3, 256, 256), batch_size=16))
# x = model(x)
# dot = make_dot(x, params=dict(list(model.named_parameters())))
# dot.render("stan_dot.png")