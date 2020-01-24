# ROADNet full model definition for Pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

MAX_H = 512

# This module combines road segmentation module and road defect segmentation module
class ROADNet(nn.Module):

    def __init__(self, base_model,
                 fine_grained_model):  #use encoder to pass pretrained encoder
        super().__init__()
        self.base_model = base_model
        self.fine_grained_model = fine_grained_model

    def forward(self, input):
        x = self.base_model(input)
        x = x.max(1)[1]
        x[x != 0] = 255
        x[x == 0] = 1
        x[x == 255] = 0
        road_mask = x
        #print("road_mask shape =", road_mask.shape)

        x = torch.stack((x, x, x), 1).float()

        #print("input shape =", input.shape)
        inputs = (input.squeeze(0) * x.squeeze(0))
        #print("fine_grained_model(inputs) 1 =", inputs.shape)
        if len(inputs) == 3:
            inputs = inputs.unsqueeze(0)
        #print("fine_grained_model(inputs) 2 =", inputs.shape)
        x = self.fine_grained_model(inputs)
        return x, road_mask        
        '''pred_img_gpu = x[0].max(0)[1].unsqueeze(0)
        pred_img_gpu[0][road_mask[0] == 0] = 255
        return pred_img_gpu'''
        

#ROADNet
class RoadSegNet(nn.Module):
    def __init__(self,
                 num_classes,
                 encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()
        print("model init calld")
        if (encoder == None):
            self.encoder = RoadSegNetEncoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = RoadSegNetDecoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input,
                                  predict=False)  #predict=False by default
            return self.decoder.forward(output)

class RoadDefectNet(nn.Module):

    def __init__(self,
                 num_classes,
                 encoder=None):  #use encoder to pass pretrained encoder
        super().__init__()
        print("model road_defectnet called")
        if (encoder == None):
            self.encoder = RoadDefectNetEncoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = RoadDefectNetDecoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input,
                                  predict=False)  #predict=False by default
            return self.decoder.forward(output)

# This Module is used to segment out the road defects
class RoadDefectSegNet(nn.Module):

    def __init__(self, num_classes,
                 model):  #use encoder to pass pretrained encoder
        super().__init__()

        if 'DataParallel' in str(type(model)):
            self.encoder = list(model.children())[0].encoder
           
            self.decoder = nn.Sequential(
                *list(list(model.children())[0].decoder.layers.children()))
            #print(self.decoder)
            self.decoder_last_child = list(
                list(model.children())[0].decoder.children())[-1]
        else:
            self.encoder = (model).encoder
            self.decoder = nn.Sequential(
                *list(list(model.decoder.layers)))

            #self.decoder_last_child = list(model.decoder)[-1]
        
        self.output_conv = nn.ConvTranspose2d(16,
                                              10,
                                              2,
                                              stride=2,
                                              padding=0,
                                              output_padding=0,
                                              bias=True)

    def forward(self, input):
        x = self.encoder(input)
        x = self.decoder(x)
        x2 = self.output_conv(x)
        return x2


class RoadSegNetDecoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16,
                                              num_classes,
                                              2,
                                              stride=2,
                                              padding=0,
                                              output_padding=0,
                                              bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

class RoadSegNetEncoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  #5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        #self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        #if predict:
        #    output = self.output_conv(output)

        return output

class DownsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput,
                              noutput - ninput, (3, 3),
                              stride=2,
                              padding=1,
                              bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        #print("DownsamplerBlock inp shape =", input.shape)
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class non_bottleneck_1d(nn.Module):

    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann,
                                   chann, (3, 1),
                                   stride=1,
                                   padding=(1, 0),
                                   bias=True)

        self.conv1x3_1 = nn.Conv2d(chann,
                                   chann, (1, 3),
                                   stride=1,
                                   padding=(0, 1),
                                   bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann,
                                   chann, (3, 1),
                                   stride=1,
                                   padding=(1 * dilated, 0),
                                   bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann,
                                   chann, (1, 3),
                                   stride=1,
                                   padding=(0, 1 * dilated),
                                   bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  #+input = identity (residual connection)


class RoadDefectNetEncoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):  #5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):  #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128,
                                     num_classes,
                                     1,
                                     stride=1,
                                     padding=0,
                                     bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class sfpBlock(nn.Module):

    def __init__(self, ninput):
        super().__init__()
        self.avg_pool1 = nn.AvgPool2d(MAX_H//8)
        #self.avg_pool2 = nn.AvgPool2d(32)
        self.avg_pool3 = nn.AvgPool2d(16)
        #self.avg_pool4 = nn.AvgPool2d(8)
        #self.avg_pool5 = nn.AvgPool2d(128)
        self.conv1 = nn.Conv2d(ninput,
                               ninput,
                               1,
                               stride=1,
                               padding=0,
                               bias=False)
        #self.conv2 = nn.Conv2d(ninput, ninput, 1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(ninput,
                               ninput,
                               1,
                               stride=1,
                               padding=0,
                               bias=False)
        #self.conv4 = nn.Conv2d(ninput, ninput, 1, stride=1, padding=0, bias=False)
        #self.conv5 = nn.Conv2d(ninput, ninput, 1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ninput, momentum=0.95)
        #self.bn2 = nn.BatchNorm2d(ninput, momentum =0.95)
        self.bn3 = nn.BatchNorm2d(ninput, momentum=0.95)
        #self.bn4 = nn.BatchNorm2d(ninput, momentum =0.95)
        self.bn5 = nn.BatchNorm2d(ninput, momentum=0.95)
        #self.upsample = F.upsample_bilinear(size=(128,64))
        self.finalconv = nn.Conv2d(384,
                                   ninput,
                                   1,
                                   stride=1,
                                   padding=0,
                                   bias=False)
        #self.dropout = F.dropout2d(p=0.9)

    def forward(self, input):

        avg1 = F.upsample_bilinear(F.relu(
            self.bn1(self.conv1(self.avg_pool1(input)))),
                                   size=(MAX_H//8, MAX_H//4))
        #avg2 = F.upsample_bilinear(F.relu(self.bn2(self.conv2(self.avg_pool2(input)))),size(128,64))
        avg3 = F.upsample_bilinear(F.relu(
            self.bn3(self.conv3(self.avg_pool3(input)))),
                                   size=(MAX_H//8, MAX_H//4))
        #avg4 = F.upsample_bilinear(F.relu(self.bn4(self.conv4(self.avg_pool4(input)))),size(128,64))
        #avg5 = F.upsample_bilinear(F.relu(self.bn5(self.conv5(self.avg_pool5(input)))),size(128,64))

        return F.dropout(F.relu(
            self.bn5(self.finalconv(torch.cat((avg1, avg3, input), dim=1)))),
                         p=0.9)

class UpsamplerBlock(nn.Module):

    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput,
                                       noutput,
                                       3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1,
                                       bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class RoadDefectNetDecoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(sfpBlock(128))
        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(16,
                                              num_classes,
                                              2,
                                              stride=2,
                                              padding=0,
                                              output_padding=0,
                                              bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output
