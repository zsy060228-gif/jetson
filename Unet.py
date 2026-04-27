# -*- coding:utf-8 _*-
from collections import OrderedDict
import torch.nn as nn
import torch
from torch import nn
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=9, init_features=8):
        super(UNet, self).__init__()
        features = init_features
        # 编码
        self.encoder1 = UNet._block3(in_channels, features, name="enc1")

        self.encoder2 = UNet._block3(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block3(features * 2, features * 2, name="enc3")
        self.dop3 = nn.Dropout(0.2)

        self.encoder4 = UNet._block3(features * 2, features * 4, name="enc4")
        self.dop4 = nn.Dropout(0.2)

        self.encoder5 = UNet._block3(features * 4, features * 4, name="enc5")
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dop5 = nn.Dropout(0.2)

        self.encoder6 = UNet._block3(features * 4, features * 8, name="enc6")
        self.dop6 = nn.Dropout(0.2)

        self.encoder7 = UNet._block3(features * 8, features * 8, name="enc7")
        self.dop7 = nn.Dropout(0.2)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码
        self.decoder10 = UNet._Tblock2(features * 8, features * 8, name="dec10")

        self.decoder9 = UNet._Tblock3(features * 8, features * 8, name="dec9")
        self.tdop9 = nn.Dropout(0.2)

        self.decoder8 = UNet._Tblock3(features * 8, features * 8, name="dec8")
        self.tdop8 = nn.Dropout(0.2)

        self.decoder7 = UNet._Tblock2(features * 8, features * 8, name="dec7")

        self.decoder6 = UNet._Tblock3(features * 8, features * 4, name="dec6")
        self.tdop6 = nn.Dropout(0.2)

        self.decoder5 = UNet._Tblock3(features * 4, features * 4, name="dec5")
        self.tdop5 = nn.Dropout(0.2)

        self.decoder4 = UNet._Tblock3(features * 4, features * 2, name="dec4")
        self.tdop4 = nn.Dropout(0.2)

        self.decoder3 = UNet._Tblock2(features * 2, features * 2, name="dec3")

        self.decoder2 = UNet._block3(features * 2, features, name="dec2")

        self.decoder1 = UNet._block3(features, out_channels, name="dec1")

    def forward(self, x):
        # x=torch.randn(1,3,256,256)
        enc1 = self.encoder1(x)
       
        enc2 = self.encoder2(enc1)
       
        enc3 = self.dop3(self.encoder3(self.pool2(enc2)))
        enc4 = self.dop4(self.encoder4(enc3))
        enc5 = self.dop5(self.encoder5(enc4))
        enc6 = self.dop6(self.encoder6(self.pool5(enc5)))
        enc7 = self.dop7(self.encoder7(enc6))

        enc = self.pool7(enc7)
       

        dec9 = self.tdop9(self.decoder10(enc))
        
        dec8 = self.tdop8(self.decoder9(dec9))
        
        dec7 = self.decoder8(dec8)
        dec6 = self.tdop6(self.decoder7(dec7))
        dec5 = self.tdop5(self.decoder6(dec6))
        dec4 = self.tdop4(self.decoder5(dec5))
        dec3 = self.decoder4(dec4)
        dec2 = self.decoder3(dec3)
        dec1 = self.decoder2(dec2)
        out = self.decoder1(dec1)
        # print("11111111111111111111111111111")
        # print(dec9.shape)
        
        return out

    @staticmethod
    def _block3(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (
                        name + "conv3",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )

    @staticmethod
    def _Tblock3(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (
                        name + "Tconv3",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )

    @staticmethod
    def _Tblock2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(  # 用字典的形式进行网络定义，字典key即为网络每一层的名称
                [
                    (name + "up",
                     nn.Upsample(
                         scale_factor=2,
                         mode='bilinear',
                         align_corners=True)
                     ),
                    (
                        name + "Tconv2",
                        nn.ConvTranspose2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            # padding_mode='reflect',
                            bias=False,
                        )
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
    # def detect_image(self, image):
    #         # ---------------------------------------------------------#
    #         #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    #         #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    #         # ---------------------------------------------------------#
    #         image = cvtColor(image)
    #         # ---------------------------------------------------#
    #         #   对输入图像进行一个备份，后面用于绘图
    #         # ---------------------------------------------------#
    #         old_img = copy.deepcopy(image)
    #         orininal_h = np.array(image).shape[0]
    #         orininal_w = np.array(image).shape[1]
    #         # ---------------------------------------------------------#
    #         #   给图像增加灰条，实现不失真的resize
    #         #   也可以直接resize进行识别
    #         # ---------------------------------------------------------#
    #         image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
    #         # ---------------------------------------------------------#
    #         #   添加上batch_size维度
    #         # ---------------------------------------------------------#
    #         image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    #         with torch.no_grad():
    #             images = torch.from_numpy(image_data)
    #             if self.cuda:
    #                 images = images.cuda()

    #             # ---------------------------------------------------#
    #             #   图片传入网络进行预测
    #             # ---------------------------------------------------#
    #             pr = self.net(images)[0]
    #             # ---------------------------------------------------#
    #             #   取出每一个像素点的种类
    #             # ---------------------------------------------------#
    #             pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
    #             # --------------------------------------#
    #             #   将灰条部分截取掉
    #             # --------------------------------------#
    #             pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
    #                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
    #             # ---------------------------------------------------#
    #             #   进行图片的resize
    #             # ---------------------------------------------------#
    #             pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
    #             # ---------------------------------------------------#
    #             #   取出每一个像素点的种类
    #             # ---------------------------------------------------#
    #             pr = pr.argmax(axis=-1)

    #         if self.mix_type == 0:
    #             # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    #             # for c in range(self.num_classes):
    #             #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
    #             #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
    #             #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
    #             seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    #             # ------------------------------------------------#
    #             #   将新图片转换成Image的形式
    #             # ------------------------------------------------#
    #             image = Image.fromarray(np.uint8(seg_img))
    #             # ------------------------------------------------#
    #             #   将新图与原图及进行混合
    #             # ------------------------------------------------#
    #             image = Image.blend(old_img, image, 0.7)

    #         elif self.mix_type == 1:
    #             # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
    #             # for c in range(self.num_classes):
    #             #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
    #             #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
    #             #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
    #             seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
    #             # ------------------------------------------------#
    #             #   将新图片转换成Image的形式
    #             # ------------------------------------------------#
    #             image = Image.fromarray(np.uint8(seg_img))

    #         elif self.mix_type == 2:
    #             seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
    #             # ------------------------------------------------#
    #             #   将新图片转换成Image的形式
    #             # ------------------------------------------------#
    #             image = Image.fromarray(np.uint8(seg_img))

    #         return image
if __name__ == '__main__':
    x=torch.randn(1,3,256,256)
    net=UNet()
    net(x)
    print("--------")
    print(net(x).shape)
    print("--------")
