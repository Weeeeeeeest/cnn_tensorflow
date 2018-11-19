import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

#from create_seg import bbox2seg
import config as cf

Dataset_Debug_Display = False

class DataLoader():
    def __init__(self, phase='Train', shuffle=False):
        self.datas = []
        self.last_mb = 0
        self.phase = phase
        self.gt_count = [0 for _ in range(cf.Class_num)]
        self.prepare_datas(shuffle=shuffle)
        if phase == 'Train':
            self.mb = cf.Minibatch
        elif phase == 'Test':
            self.mb = 1
                
    def prepare_datas(self, shuffle=True):
        if self.phase == 'Train':
            dir_paths = cf.Train_dirs
        elif self.phase == 'Test':
            dir_paths = cf.Test_dirs
        
        print('------------\nData Load (phase: {})'.format(self.phase))
        
        for dir_path in dir_paths:
            files = []
            for ext in cf.File_extensions:
                files += glob.glob(dir_path + '/*' + ext)
            files.sort()
            
            load_count = 0
            for img_path in files:
                if cv2.imread(img_path) is None:
                    continue
                gt = get_gt(img_path)
                
                data = {'img_path': img_path, 'gt_path': gt,
                        'h_flip': False, 'v_flip': False, 'rotate': False}
                
                self.datas.append(data)
                self.gt_count[gt] += 1
                load_count += 1
            print(' - {} - {} datas -> loaded {}'.format(dir_path, len(files), load_count))

        self.display_gt_statistic()
                
        if self.phase == 'Train':
            self.data_augmentation()
            self.display_gt_statistic()

        self.set_index(shuffle=shuffle)
                        
    def display_gt_statistic(self):
        print(' -*- Training label  -*-')
        print('   Total data: {}'.format(len(self.datas)))
        for i, gt in enumerate(self.gt_count):
            print('  - {} : {}'.format(cf.Class_label[i], gt))

    def get_data_num(self):
        return self.data_n
    
    def set_index(self, shuffle=True):
        self.data_n = len(self.datas)
        self.indices = np.arange(self.data_n)
        if shuffle:
            np.random.seed(cf.Random_seed)
            np.random.shuffle(self.indices)
        

    def get_minibatch_index(self, shuffle=False):
        _last = self.last_mb + self.mb
        if _last >= self.data_n:
            mb_inds = self.indices[self.last_mb:]
            self.last_mb = _last - self.data_n
            if shuffle:
                np.random.seed(cf.Random_seed)
                np.random.shuffle(self.indices)
            _mb_inds = self.indices[:self.last_mb]
            mb_inds = np.hstack((mb_inds, _mb_inds))

        else:
            mb_inds = self.indices[self.last_mb : self.last_mb+self.mb]
            self.last_mb += self.mb

        self.mb_inds = mb_inds


    def get_minibatch(self, shuffle=True):
        self.get_minibatch_index(shuffle=shuffle)

        imgs = np.zeros((self.mb, cf.Height, cf.Width, cf.Channel), dtype=np.float32)
        gts = np.zeros((self.mb, cf.Class_num), dtype=np.float32)
        
        for i, ind in enumerate(self.mb_inds):
            data = self.datas[ind]
            img = load_image(data['img_path'])
            img = image_dataAugment(img, data)
            if cf.Channel == 1:
                img = cv2.cvtColor(cv2.GRAY_SCALE)
                img = expand_imds(img, axis=-1)
            
            gt = get_gt(data['img_path'])
            
            imgs[i] = img
            gts[i, gt] = 1

        if cf.Input_type == 'channels_first':
            imgs = imgs.transpose(0, 3, 1, 2)
        return imgs, gts
        
    
    
    def data_augmentation(self):
        print('   ||   -*- Data Augmentation -*-')
        if cf.Horizontal_flip:
            self.add_horizontal_flip()
            print('   ||    - Added horizontal flip')
        if cf.Vertical_flip:
            self.add_vertical_flip()
            print('   ||    - Added vertival flip')
        if cf.Rotate_ccw90:
            self.add_rotate_ccw90()
            print('   ||    - Added Rotate ccw90')
        print('  \  /')
        print('   \/')
    
    def add_horizontal_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['h_flip'] = True
            new_data.append(_data)
            gt = get_gt(data['img_path'])
            self.gt_count[gt] += 1
        self.datas.extend(new_data)

    def add_vertical_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['v_flip'] = True
            new_data.append(_data)
            gt = get_gt(data['img_path'])
            self.gt_count[gt] += 1
        self.datas.extend(new_data)

    def add_rotate_ccw90(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['rotate'] = True
            new_data.append(_data)
            gt = get_gt(data['img_path'])
            self.gt_count[gt] += 1
        self.datas.extend(new_data)


def get_gt(img_name):
    for ind, cls in enumerate(cf.Class_label):
        if cls in img_name:
            return ind
    raise Exception("Class label Error {}".format(img_name))
    
## Below functions are for data augmentation
def load_image(img_name):
    img = cv2.imread(img_name)
    if img is None:
        raise Exception('file not found: {}'.format(img_name))

    if cf.Variable_input:
        longer_side = np.max(img.shape[:2])
        scaled_ratio = 1. * cf.Max_side / longer_side
        scaled_height = np.min([img.shape[0] * scaled_ratio, cf.Max_side]).astype(np.int)
        scaled_width = np.min([img.shape[1] * scaled_ratio, cf.Max_side]).astype(np.int)
        img = cv2.resize(img, (scaled_width, scaled_height))
    else:
        scaled_height = cf.Height
        scaled_width = cf.Width
        img = cv2.resize(img, (scaled_width, scaled_height))

    img = img[:, :, (2,1,0)]
    img = img / 255.
    return img



def image_dataAugment(image, data):
    h, w = image.shape[:2]
    if data['h_flip']:
        image = image[:, ::-1]
    if data['v_flip']:
        image = image[::-1, :]
    if data['rotate']:
        max_side = max(h, w)
        if len(image.shape) == 3: 
            frame = np.zeros((max_side, max_side, 3), dtype=np.float32)
        elif len(image.shape) == 2:
            frame = np.zeros((max_side, max_side), dtype=np.float32)
        tx = int((max_side-w)/2)
        ty = int((max_side-h)/2)
        frame[ty:ty+h, tx:tx+w] = image
        M = cv2.getRotationMatrix2D((max_side/2, max_side/2), 90, 1)
        rot = cv2.warpAffine(frame, M, (max_side, max_side))
        image = rot[tx:tx+w, ty:ty+h]
        temp = h
        h = w
        w = temp
    return image
