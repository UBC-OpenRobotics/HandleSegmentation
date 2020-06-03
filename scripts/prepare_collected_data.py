#!/usr/bin/python
import os
import numpy as np
import cv2
import json
import argparse

#Simple augmentation, just rotates and flips images
class Augmentor():
    
    def __init__(self, dataset, angle_max=15, rotate_percent=0.50):
        self.dataset = dataset
        self.angle_max = angle_max
        self.rotate_percent = rotate_percent
    
    def rotate_data(self,imgs, angles):
        ret = []
        for i,image in enumerate(imgs):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angles[i], 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            ret.append(result)
        return np.asarray(ret)
    
    def flip_data(self,imgs, rotate_idx):
        for idx in rotate_idx:
            imgs[idx] = cv2.flip(imgs[idx],1)
        return imgs
    
    def generate(self, n=10):
        sample = self.dataset[np.random.choice(self.dataset.shape[0], size=n)]
        angles = np.random.randint(-self.angle_max, high=self.angle_max, size=n) 
        rotate_idx = np.random.choice(imgs.shape[0],n)
        aug = self.rotate_data(sample, angles)
        aug = self.flip_data(aug, rotate_idx)
        return aug

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-r','--root_path', type=str, required=True, help='path to HandleSegmentation root folder')
    ap.add_argument('-a','--angle_max', type=int, required=False, default=15, help='maximum angle to rotate images in augmentation, default = 15')
    ap.add_argument('-rp','--rotate_percent', type=float, required=False,default=0.15, help='percentage of generated samples to flip, default = 0.15')
    args = vars(ap.parse_args())

    base_path = args['root_path']
    #Read configuration file
    with open(os.path.join(base_path,'config.json')) as f:
        config = json.load(f)

    dataset_path = config['dataset_path']
    w = config['input_w']
    h = config['input_h']
    frame_skip = config['collection_frame_skip']
    dev_split = config['dev_split']
    train_img_dir = config['train_img_dir']
    dev_img_dir = config['dev_img_dir']

    #Get absolute paths
    dataset_path = os.path.join(base_path, dataset_path)
    train_img_dir = os.path.join(base_path, train_img_dir)
    dev_img_dir = os.path.join(base_path, dev_img_dir)

    #Open dataset_path, which contains pictures and videos. save a resized version of each image 
    #and read the videos, saving each frame_skip number of frames. 
    imgs = []

    assert os.listdir(dataset_path), "No files found in %s" %(dataset_path)

    for file_name in os.listdir(dataset_path):
        ext = file_name.split('.')[-1]
        if ext in ['jpg', 'png']:
            img_path = os.path.join(dataset_path, file_name)
            img = cv2.imread(img_path, 0)
            img = cv2.resize(img, (w,h))
            imgs.append(img)
        elif ext in ['avi','mp4','wmv']:
            vid_path = os.path.join(dataset_path, file_name)
            vid = cv2.VideoCapture(vid_path)
            
            ret, frame  = vid.read()
            frame_counter = 0
            c=0
            while ret:
                if frame_counter % frame_skip == 0:
                    img = cv2.resize(frame, (w,h))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    imgs.append(img)
                    c+=1
                frame_counter+=1
                ret, frame=vid.read()
            
            print('Saved %i frames' %(c))

    imgs = np.asarray(imgs)
    print('Dataset Shape: ' + str(imgs.shape))

    aug_bool = raw_input('\n\nDo you want to Augment the Dataset? [y/n]\n')
    while aug_bool not in ['y','Y','yes','n','N','no']:
        aug_bool = raw_input('\n\nDo you want to Augment the Dataset? [y/n]\n')
    
    if aug_bool in ['y','Y','yes']:
        try:
            angle_max = int(args['angle_max'])
            rotate_percent = float(args['rotate_percent'])
            aug_size = int(raw_input('\n\nHow many samples do you wish to generate?\n'))
            
            aug = Augmentor(imgs, angle_max=angle_max, rotate_percent=rotate_percent)
            imgs = np.concatenate([imgs, aug.generate(aug_size)])

            print('\nNew Dataset Shape: ' + str(imgs.shape))
        except Exception as e:
            print(e)
            exit()


    #Shuffle, Take %15 and save to dev set, save the rest to test set
    np.random.shuffle(imgs)
    num_split = int(imgs.shape[0]*dev_split)

    dev = imgs[-num_split:]
    train = imgs[:-num_split]

    print('Train Set has %i elements\nDev Set has %i elements' %(train.shape[0],dev.shape[0]))

    #Save Train Set
    resp = raw_input('You are about to save images to %s and %s. You may overwrite existing data, Do you wish to proceed? y/n\n' %(train_img_dir, dev_img_dir))
    while resp not in ['y','n']:
        resp = raw_input('You are about to save images to %s and %s. You may overwrite existing data, Do you wish to proceed? y/n\n' %(train_img_dir, dev_img_dir))

    if resp == 'y':
        for i, img in enumerate(train):
            img_dir = os.path.join(train_img_dir, 'train_%i.jpg'%(i+1))
            cv2.imwrite(img_dir, img)

        #Save Dev Set
        for i, img in enumerate(dev):
            img_dir = os.path.join(dev_img_dir, 'dev_%i.jpg'%(i+1))
            cv2.imwrite(img_dir, img)

        print('Saved Train and Dev sets')
    if resp == 'n':
        print('Saving Aborted')