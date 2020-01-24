import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

from torchvision.transforms import ToTensor, ToPILImage
from utilities.transform import Relabel, ToLabel, Colorize, TransformImages

EXTENSIONS = ['.jpg', '.png']

def tensor2im(image_tensor, imtype=np.uint8):
    image_tensor = image_tensor.detach()
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    # image_numpy = image_numpy.astype(imtype)
    return image_numpy

def save_one_output(pred_img_gpu, images, filename, val_ct):
    #print("pred_img_gpu sum [0] =", pred_img_gpu.flatten().sum())

    col_img = Colorize()(pred_img_gpu)
    predictionClr = ToPILImage()(col_img.cpu().byte())

    filenameSave = "./predicts/" + str(val_ct).zfill(3) + '.png'
    filename_break = str(filename[0]).split('/')
    filename_path = '/'.join(filename_break[-3:])
    filenameSave = "./predicts/" + str(filename_path)
    os.makedirs(os.path.dirname(filenameSave), exist_ok=True)

    ## SAve transparent color
    orig_img = Image.fromarray(
        tensor2im(images).astype(np.uint8))
    orig_file_save = filenameSave + 'orig.png'

    background = orig_img.convert("RGBA")
    overlay = predictionClr.convert("RGBA")
    new_img = Image.blend(background, overlay, 0.3)
    overlay_file_save = filenameSave + 'overlay.png'
    predictionClr.save(filenameSave)
    orig_img.save(orig_file_save)
    new_img.save(overlay_file_save)

def save_output(outputs, road_mask, images, filename, val_ct):
    #compatibility with criterion dataparallel
    if isinstance(outputs, list):  #merge gpu tensors
        outputs_cpu = outputs[0].cpu()
        for i in range(1, len(outputs)):
            outputs_cpu = torch.cat((outputs_cpu, outputs[i].cpu()),
                                    0)
    else:
        outputs_cpu = outputs.cpu()

    outputs_gpu = outputs
    pred_img_gpu = outputs_gpu[0].max(0)[1].unsqueeze(0)
    roadMask_gpu = road_mask[0]
    pred_img_gpu[0][roadMask_gpu == 0] = 255
    print("pred_img_gpu [0] =", pred_img_gpu.flatten().unique())

    print("outputs_cpu = ", outputs_cpu.shape)
    print("pred_img_gpu = ", pred_img_gpu.shape)

    col_img_gpu = Colorize()(pred_img_gpu)
    print("col_img = ", col_img_gpu.flatten()[0:5])

    for i in range(0, outputs_cpu.size(0)):  #args.batch_size
        val_ct += 1
        pred_img = outputs_cpu[i].max(0)[1].data.unsqueeze(0)
        #print(type(pred_img), pred_img.shape)
        roadMask = road_mask[i].data.cpu()

        pred_img[0][roadMask == 0] = 255
        col_img = Colorize()(pred_img.byte())
        print("col_img = ", col_img.flatten()[0:5])

        predictionClr = ToPILImage()(col_img)
        #prediction = ToPILImage()(pred_img.byte())

        filenameSave = "./predicts/" + str(val_ct).zfill(3) + '.png'
        filename_break = str(filename[0]).split('/')
        filename_path = '/'.join(filename_break[-3:])
        filenameSave = "./predicts/" + str(filename_path)
        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)

        ## SAve transparent color
        orig_img = Image.fromarray(
            tensor2im(images).astype(np.uint8))
        orig_file_save = filenameSave + 'orig.png'

        background = orig_img.convert("RGBA")
        overlay = predictionClr.convert("RGBA")
        new_img = Image.blend(background, overlay, 0.3)
        overlay_file_save = filenameSave + 'overlay.png'
        #predictionClr.save(filenameSave)
        orig_img.save(orig_file_save)
        new_img.save(overlay_file_save)

class cityscapes(Dataset):

    def __init__(self, root, co_transform=None, subset='train'):
        self.images_root = os.path.join(root, 'leftImg8bit/')
        self.labels_root = os.path.join(root, 'gtFine/')

        self.images_root += subset
        self.labels_root += subset

        self.filenames = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(self.images_root))
            for f in fn
            if is_image(f)
        ]
        self.filenames.sort()

        self.filenamesGt = [
            os.path.join(dp, f)
            for dp, dn, fn in os.walk(os.path.expanduser(self.labels_root))
            for f in fn
            if is_label(f)
        ]
        self.filenamesGt.sort()

        self.co_transform = co_transform

    def __getitem__(self, index):

        filename = self.filenames[index]
        filenameGt = self.filenamesGt[index]

        with open(image_path_city(self.images_root, filename), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path_city(self.labels_root, filenameGt), 'rb') as f:
            label = load_image(f).convert('P')

        oldimage = image
        if self.co_transform is not None:
            image, label = self.co_transform(image, label)

        return image, ToTensor()(oldimage), label, filename, filenameGt

    def __len__(self):
        return len(self.filenames)

def load_image(file):
    return Image.open(file)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)


def is_label(filename):
    return filename.endswith("_labelTrainIds.png")


def image_path(root, basename, extension):
    return os.path.join(root, basename + extension)


def image_path_city(root, name):
    return os.path.join(root, name)


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def remove_all(substr, str):
    index = 0
    length = len(substr)
    while str.find(substr) != -1:
        index = str.find(substr)
        str = str[0:index] + str[index+length:]
    return str

def load_my_state_dict(model, state_dict, cuda=True):  #custom function to load model when not all dict keys are there
    own_state = model.state_dict()
    #for name, param in own_state.items():
    #    print("MODEL loading weight for ", name)
    
    for name, param in state_dict.items():
        #print("State Dict loading weight for ", name)
        name = remove_all("module.", name) if not cuda else name
        if name not in own_state:
            print("NOT loaded weight for ", name)
            continue
        own_state[name].copy_(param)
        #print("***********************loaded weight for ", name)
    print("All weights loaded")
    return model
