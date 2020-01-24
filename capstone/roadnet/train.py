import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    warnings.filterwarnings("ignore",category=UserWarning)
import os
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage

from utilities.dataset import save_output, load_my_state_dict, cityscapes, save_one_output
from utilities.losses import CrossEntropyLoss2dv2
from utilities.transform import TransformImages
from utilities.parallel import ModelDataParallel, CriterionDataParallel
from torchsummary import summary

import importlib
import checkIoU
import utilities.transform

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from roadnet import ROADNet, RoadSegNet, RoadDefectNet, RoadDefectSegNet

#np.set_printoptions(threshold='nan')

NUM_CHANNELS = 3
NUM_CLASSES = 10

best_acc = 0
save_val = False


def train(args, model, enc):
    global best_acc

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing

    weight = torch.ones(NUM_CLASSES)

    assert os.path.exists(
        args.datadir), "Error: datadir (dataset directory) could not be loaded"

    #Loading the dataset
    tt = TransformImages(False, augment=True,
                                   height=args.height)
    tv = TransformImages(False, augment=False,
                                       height=args.height)

    x_train = cityscapes(args.datadir, tt, 'train')
    x_test = cityscapes(args.datadir, tv, 'test')

    loader = DataLoader(x_train,
                        num_workers=args.num_workers,
                        batch_size=args.batch_size,
                        shuffle=True)
    loader_val = DataLoader(x_test,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=False)

    losses = CrossEntropyLoss2dv2(weight.cuda())

    savedir = './save/' + args.savedir

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)
       ):  #dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write(
                "Epoch\t\rain-loss\t\est-loss\t\rain-IoU\t\est-IoU\t\tlearningRate"
            )

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))

    # We use Adam optimizer with lr of 5e-4
    optimizer = Adam([
        {
            'params': model.parameters()
        },
    ],
                     5e-4, (0.9, 0.999),
                     eps=1e-08,
                     weight_decay=1e-4)

    start_epoch = 1

    lambda1 = lambda epoch: pow(
        (1 - ((epoch - 1) / args.num_epochs)), 0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer,
                                      lr_lambda=lambda1)  ## scheduler 2

    cont_train_loss = []
    cont_val_loss = []
    for epoch in range(start_epoch, args.num_epochs + 1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)  ## scheduler 2

        epoch_loss = []
        time_train = []

        #TODO: remake the checkIoU.py code to avoid using "checkIoU.args"
        confMatrix = checkIoU.generateMatrixTrainId(checkIoU.args)
        perImageStats = {}
        nbPixels = 0

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()

        for step, (images, oldimages, labels, filename,
                   filenameGt) in enumerate(loader):
            start_time = time.time()
            break

            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs, road_mask = model(inputs)

            optimizer.zero_grad()
            loss = losses(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            time_train.append(time.time() - start_time)

        if not args.eval:
            average_epoch_loss_train = 0  #sum(epoch_loss) / len(epoch_loss)
        else:
            average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        #checkIoU.printConfMatrix(confMatrix, checkIoU.args)

        #Validate on val images after each epoch of training
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        #model = pretrained_model
        epoch_loss_val = []
        time_val = []

        #New confusion matrix data
        confMatrix = checkIoU.generateMatrixTrainId(checkIoU.args)
        perImageStats = {}
        nbPixels = 0
        val_ct = 0
        for step, (images, oldimages, labels, filename,
                   filenameGt) in enumerate(loader_val):
            start_time = time.time()
            #break
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs, road_mask = model(inputs)

            loss = losses(outputs, targets[:, 0])
            #print("loss.shape =", loss.shape)

            #epoch_loss_val.append(loss.data[0])
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)
        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

        iouVal = 0 #dummy, add later
        confMatrix = confMatrix[:12, :12]
        current_acc = iouVal
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        filenameCheckpoint = savedir + '/checkpoint.pth.tar'
        filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        filename = savedir + '/model-' + str(epoch) + '}.pth'
        filenamebest = savedir + '/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print('save: {' + filename + '} (epoch: {' + str(epoch) + '})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print('save: {' + filenamebest + '} (epoch: {' + str(epoch) + '})')

            with open(savedir + "/best_encoder.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" %
                             (epoch, iouVal))

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" %
                         (epoch, average_epoch_loss_train,
                          average_epoch_loss_val, False, False, usedLr))

    return (model)  #return model (convenience for encoder-decoder training)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


def evaluate(args, model):
    model.eval()

    image = input_transform(Image.open(args.image))
    label = model(Variable(image, volatile=True).unsqueeze(0))
    #label = color_transform(label[0].data.max(0)[1])

    image_transform(label).save(args.label)


def main(args):
    savedir = './save/' + args.savedir
    print(args)
    exit(0)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    model = RoadDefectNet(20)
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
        #model = ModelDataParallel(model).cuda()

    if args.state:
        model = load_my_state_dict(model, torch.load(args.state))

    #train(args, model)
    print("========== TRAINING ===========")
    if (not args.state):
        pretrainedEnc = next(model.children()).encoder
        defectNetModel = RoadDefectNet(20, encoder=pretrainedEnc)  #Add decoder to encoder
        defectNetModel = torch.nn.DataParallel(defectNetModel).cuda()

        defSegNet = RoadDefectSegNet(20, defectNetModel)
        defSegNet = torch.nn.DataParallel(defSegNet).cuda()

        roadSeg_model = RoadSegNet(20)
        roadSeg_model = torch.nn.DataParallel(roadSeg_model).cuda()

        roadNet_model = ROADNet(roadSeg_model, defSegNet)
        roadNet_model = torch.nn.DataParallel(roadNet_model).cuda()
        roadNet_model = load_my_state_dict(roadNet_model,
                                           torch.load('./model_best.pth'))
        #print("========== roadNet_model MODEL SUMMARY ===========")
        #print(summary(roadNet_model, (3, 512, 1024)))

        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, roadNet_model, False)  #Train decoder
    print("========== TRAINING FINISHED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir',
                        default="/media/siju/DATA/work/capstone/anue/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
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
    )  #, default="./trained_models/roadnet_encoder_pretrained.pth.tar")
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
