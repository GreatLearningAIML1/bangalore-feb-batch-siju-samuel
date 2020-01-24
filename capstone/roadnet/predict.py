import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=UserWarning)
import os
import time
import numpy as np
import torch
import math
import torchvision

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

from utilities.dataset import save_output, load_my_state_dict, cityscapes, save_one_output
from utilities.transform import TransformImages
from utilities.parallel import ModelDataParallel, CriterionDataParallel
from torchsummary import summary

import importlib

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from roadnet import ROADNet, RoadSegNet, RoadDefectNet, RoadDefectSegNet

#np.set_printoptions(threshold='nan')

NUM_CLASSES = 10

best_acc = 0
save_val = False

def test(args, model, enc):
    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)
    assert os.path.exists(
        args.datadir), "Error: datadir (dataset directory) could not be loaded"

    #Loading the dataset
    test_transform = TransformImages(False, augment=False,
                                       height=args.height)
    test_dataset = cityscapes(args.datadir, test_transform, 'test')
    test_loader = DataLoader(test_dataset,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False)

    start_epoch = 1

    save_pred_img = True
    val_ct = 0
    for epoch in range(start_epoch, args.num_epochs + 1):
        #Validate on val images after each epoch of training
        #print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        for step, (images, _, _, filename, _) in enumerate(test_loader):
            if step == 1:
                break
                #pass
            #print("Processing ", filename, end="...")
            if args.cuda:
                images = images.cuda()

            inputs = Variable(images)
            '''outputs, road_mask = model(inputs)
            print("Outputs [0] =", outputs.flatten()[0:5])
            print("road_mask [1] =", road_mask.flatten()[0:5], " Sum=", road_mask.sum())
            #exit(0)
            if (save_pred_img):
                save_output(outputs, road_mask, images, filename, val_ct)
            '''
            road_mask = model(inputs)
            save_one_output(road_mask, images, filename, val_ct)
            print(filename, " Done.")

    return (model)  #return model (convenience for encoder-decoder training)


def main(args):
    print(args)
    exit(0)
    #Load Model
    model = RoadDefectNet(20)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if args.state:
        model = load_my_state_dict(model, torch.load(args.state))

    #print("========== MODEL SUMMARY ===========")
    #print(summary(model, (3, 512, 1024)))
    if (not args.state):
        pretrainedEnc = next(model.children()).encoder
        defectNetModel = RoadDefectNet(20, encoder=pretrainedEnc)  #Add decoder to encoder
        defectNetModel = torch.nn.DataParallel(defectNetModel).cuda()

        defSegNet = RoadDefectSegNet(20, defectNetModel)
        defSegNet = torch.nn.DataParallel(defSegNet).cuda()
        #print(summary(defSegNet, (3, 256, 512)))
        roadSeg_model = RoadSegNet(20)
        roadSeg_model = torch.nn.DataParallel(roadSeg_model).cuda()
        #print(summary(roadSeg_model, (3, 256, 512)))

        roadNet_model = ROADNet(roadSeg_model, defSegNet)
        roadNet_model = torch.nn.DataParallel(roadNet_model).cuda()
        roadNet_model = load_my_state_dict(roadNet_model,
                                           torch.load('./model_best.pth'))
        print("========== roadNet_model MODEL SUMMARY ===========")
        #print(summary(roadNet_model, (3, 256, 512)))
        #print(roadNet_model)
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = test(args, roadNet_model, False)  #Train decoder
    #print("========== TESTING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir',
                        default="/media/siju/DATA/work/capstone/anue/")
    parser.add_argument('--height', type=int, default=512)  #default 512
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)  #default=6
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument(
        '--epochs-save', type=int,
        default=0)  #You can use this value to save model every X epochs
    parser.add_argument('--savedir', default="savedir", required=False)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--eval', action='store_true', default=False)
    parser.add_argument(
        '--pretrainedEncoder'
    )  #, default="../trained_models/roadnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')

    parser.add_argument(
        '--iouTrain', action='store_true',
        default=False)
    parser.add_argument(
        '--iouVal', action='store_true', default=False
    )
    parser.add_argument('--resume', action='store_true'
                       )  #Use this flag to load last checkpoint for training

    main(parser.parse_args())
