import os
import random
import numpy
import numpy as np
from scipy import ndimage
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
#         DataAugment
def random_rot_flip(image, label):
    k = numpy.random.randint(0, 4)
    image = numpy.rot90(image, k)
    label = numpy.rot90(label, k)
    axis = numpy.random.randint(0, 2)
    image = numpy.flip(image, axis=axis).copy()
    label = numpy.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = numpy.random.randint(-40, 40)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def randomGaussian(image, label, mean=0.2, sigma=0.3):
    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        Gaussian noise processing of images
        :param im: Single-channel images
        :param mean: Offset
        :param sigma: Standard deviation
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

     # Converting images into arrays
    img = numpy.asarray(image)
    img.flags.writeable = True  # Changing arrays to read and write mode
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return numpy.uint8(img), label

class RandomGenerator(object):
    def __init__(self, output_size,p):
        self.output_size = output_size
        self.p = p  
    def __call__(self, image, label):
        # image, label = sample['image'], sample['label']

        if random.random() > 1-self.p:
            image, label = random_rot_flip(image, label)
            if random.random() > 1-self.p:
                image, label = randomGaussian(image, label)
        elif random.random() > 1-self.p:
            image, label = random_rotate(image, label)
            if random.random() > 1-self.p:
                image, label = randomGaussian(image, label)
        elif random.random() >1-self.p:
            image, label = randomGaussian(image, label)


        return image, label

Background = [0,0,0]
Human = [0,0,1]
Plant= [0,1,0]
Wreck = [0,1,1]
Robot = [1,0,0]
Reef = [1,0,1]
Fish = [1,1,0]
Rocks = [1,1,1]

def rgb_to_2D_label(label):
    """
    Supply our label masks as input in RGB format.
    Replace pixels with specific RGB values ...
    """
    label_seg = np.zeros(label.shape)
    label_seg[np.all(label == Background, axis=-1)] = 0
    label_seg[np.all(label == Human, axis=-1)] = 1
    label_seg[np.all(label == Plant, axis=-1)] = 2
    label_seg[np.all(label == Wreck, axis=-1)] = 3
    label_seg[np.all(label == Robot, axis=-1)] = 4
    label_seg[np.all(label == Reef, axis=-1)] = 5
    label_seg[np.all(label == Fish, axis=-1)] = 6
    label_seg[np.all(label == Rocks, axis=-1)] = 7

    label_seg = label_seg[:,:,0]  # Just take the first channel, no need for all 3 channels

    return label_seg

class SUIM(Dataset):
    def __init__(self, img_path, label_path, image_transform, RandomGenerator=None):
        self.img_path = img_path
        self.label_path = label_path
        self.label_data = os.listdir(self.label_path)
        self.image_transform = image_transform # Data enhancement
        self.RandomGenerator = RandomGenerator
        # self.label_transform = label_transform # Data enhancement
        self.resize = transforms.Resize((256, 256)) # Trimming of data
    def __len__(self):
        return len(self.label_data)  # Number of data returned

    def __getitem__(self, item):
        img_name = os.path.join(self.img_path, self.label_data[item]) # 'dataset/training/picture' +‘00001_matte.png’
        img_name = os.path.split(img_name) # 'dataset/training/picture/00001_matte.png' -> ('dataset/training/picture','00001_matte.png')
        img_name = img_name[-1] #'00001_matte.png'
        img_name = img_name.split('.')
        img_name = img_name[0] + '.jpg'   #  '00001' + ‘.png’ is the file name of the data
        img_data = os.path.join(self.img_path, img_name) # 'dataset/training/picture/00001.png'
        label_data =os.path.join(self.label_path, self.label_data[item])
        img = Image.open(img_data)
        label = Image.open(label_data)
        label_0 = label.copy()
        img = self.resize(img)
        label = self.resize(label)
        img = numpy.array(img)
        label = numpy.array(label)
        label=label/255.
        label=rgb_to_2D_label(np.array(label))
        sample = {'image': img, 'label': label}
        
        if self.RandomGenerator:
            img,label = self.RandomGenerator(img, label)
        if self.image_transform:
            img = self.image_transform(img)
        return img, label


'''The dataloader is loaded, and when it is loaded, it is processed in the trainloader.（img，label）'''
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
])

label_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Make sure this matches the image resizing
    # Possibly other augmentations that are safe for label maps
    transforms.ToTensor(),
])

def get_dataloaders(data_dir, batch_size, p=0.5):
    train_dataset = SUIM(data_dir+'train_val/images',
                       data_dir+'train_val/train_label'
                       ,
                       image_transform=image_transform,
                       RandomGenerator=RandomGenerator(output_size=(256, 256),p=p)
                       )

    test_data = SUIM(data_dir+'test/images', data_dir+'test/test_label', image_transform=image_transform)

    train_data, val_data = torch.utils.data.random_split(train_dataset, [1220, 305], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_data, batch_size=batch_size,
                              shuffle=True, num_workers=2,
                              drop_last=True)

    val_loader = DataLoader(val_data, batch_size=batch_size,
                              shuffle=False, num_workers=2,
                              drop_last=True)

    test_loader = DataLoader(test_data, batch_size=1,
                              shuffle=False, num_workers=0,
                              drop_last=True)
    return train_loader, val_loader, test_loader