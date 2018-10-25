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
        
        
    def prepare_datas(self, shuffle=True):
        if self.phase == 'Train':
            dir_paths = cf.Train_dirs
        elif self.phase == 'Test':
            dir_paths = cf.Test_dirs
        
        print('------------\nData Load (phase: {})'.format(self.phase))
        
        for dir_path in dir_paths:
            files1 = glob.glob(dir_path + '/*.jpg')
            files2 = glob.glob(dir_path + '/*.png')
            files = files1 + files2
            
            load_count = 0
            for img_path in files:
                if cv2.imread(img_path) is None:
                    continue
                gt = self.get_gt(img_path)
                #if self.gt_count[gt] >= 10000:
                #    continue
                
                data = {'img_path': img_path,
                        'gt_path': gt,
                        'h_flip': False,
                        'v_flip': False,
                        'rotate': False
                }
                
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
        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = 1
        _last = self.last_mb + mb
        if _last >= self.data_n:
            mb_inds = self.indices[self.last_mb:]
            self.last_mb = _last - self.data_n
            if shuffle:
                np.random.seed(cf.Random_seed)
                np.random.shuffle(self.indices)
            _mb_inds = self.indices[:self.last_mb]
            mb_inds = np.hstack((mb_inds, _mb_inds))

        else:
            mb_inds = self.indices[self.last_mb : self.last_mb+mb]
            self.last_mb += mb

        self.mb_inds = mb_inds


    def get_minibatch(self, shuffle=True):
        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = 1        

        self.get_minibatch_index(shuffle=shuffle)

        if cf.Variable_input:
            imgs = np.zeros((mb, cf.Max_side, cf.Max_side, 3), dtype=np.float32)
        else:
            imgs = np.zeros((mb, cf.Height, cf.Width, 3), dtype=np.float32)
            
        gts = np.zeros((mb, cf.Class_num), dtype=np.float32)
        max_height, max_width = 0, 0
        
        for i, ind in enumerate(self.mb_inds):
            data = self.datas[ind]
            img, orig_h, orig_w = self.load_image(data['img_path'])
            
            height, width = img.shape[:2]
            resized_ratio = 1. * height / orig_h

            img, height, width = self.image_dataAugment(img, data)

            if False:# and 'MSRA' in data['img_path']:
                print(data['img_path'])
                print('v_flip:{}, h_flip:{}, rotate:{}'.format(
                    data['v_flip'], data['h_flip'], data['rotate']))
                plt.imshow(img)
                plt.subplots()
                plt.imshow(gt, cmap='gray')
                plt.show()
            
            max_height = max(max_height, height)
            max_width = max(max_width, width)
            
            gt = self.get_gt(data['img_path'])
            
            imgs[i, :height, :width] = img
            gts[i, gt] = 1
            #print(data['img_path'], gt)
            if Dataset_Debug_Display:
                print(data['img_path'])
                print()
                plt.imshow(imgs[i].transpose(1,2,0))
                plt.subplots()
                plt.imshow(gts[i,0])
                plt.show()

        if cf.Input_type == 'channels_first':
            imgs = imgs.transpose(0, 3, 1, 2)
        
        return imgs, gts

    
    def get_gt(self, img_name):
        for ind, cls in enumerate(cf.Class_label):
            if cls in img_name:
                return ind
        raise Exception("Class label Error {}".format(img_name))

    
    def get_gt_image_path(self, img_name):
        file_name = os.path.basename(img_name).split('.')[0]
        if self.phase == 'Train':
            gt_dirs = cf.GT_dirs
        elif self.phase == 'Test':
            gt_dirs = cf.Test_GT_dirs

        for gt_dir in gt_dirs:
            gt_path = os.path.join(gt_dir, file_name) + '.txt'
            if os.path.exists(gt_path):
                return gt_path

        raise Exception('file not found ->', gt_path)
    
    
    ## Below functions are for data augmentation
    def load_image(self, img_name):
        img = cv2.imread(img_name)
        
        if img is None:
            raise Exception('file not found: {}'.format(img_name))
        
        orig_h, orig_w = img.shape[:2]

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

        return img, orig_h, orig_w


    def load_gt_image(self, img_name, h_flip=False, v_flip=False):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        orig_h, orig_w = img.shape[:2]
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

        #img = img[:, :, (2,1,0)]
        #img = img / 255.

        if len(img.shape) < 3:
            img = img[:, :, np.newaxis]
        
        return img, orig_h, orig_w


    def image_dataAugment(self, image, data):
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
        return image, h, w
        

    def image_downSampling(self, image, scale=1.0):
        mb, c, h, w = image.shape
        resized_h = int(h * scale)
        resized_w = int(w * scale)
        img = chainer.functions.resize_images(image, (resized_h, resized_w)).data.astype(np.int)
        #img = self.resize_images(image, resized_h, resized_w)
        img = self.image_forSoftmax_trans(img, cf.Class_num)
        #img = np.reshape(img, [-1, cf.Class_num])
        #img = tf.reshape(img, [-1, cf.Class_num]).eval()
        return img


    def image_forSoftmax(self, img, cls):
        if cf.Input_type == 'channels_first':
            mb, c, h, w = img.shape
        elif cf.Input_type == 'channels_last':
            mb, h, w, c = img.shape
        out = np.zeros((mb, cls, h, w), dtype=np.int)
        for i, _img in enumerate(img):
            _img = _img[0]
            out[i] = _img
        print(img.shape)
        #for cls_i in range(cls):
        #    map_ind = np.where(img.astype(np.int) == cls_i)[0]
        #    out[map_ind[0], cls_i, map_ind[2], map_ind[3]] = 1
        #print(out.shape)
        return out
    
    
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
            _data = data.copy
            _data['h_flip'] = True
            new_data.append(_data)
            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1
        self.datas.extend(new_data)

    def add_vertical_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['v_flip'] = True
            new_data.append(_data)
            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1
        self.datas.extend(new_data)

    def add_rotate_ccw90(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['rotate'] = True
            new_data.append(_data)
            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1
        self.datas.extend(new_data)


def resize_images(img, scale=1.0):
    if cf.Input_type == 'channels_first':
        mb, c, h, w = img.shape
        rh = int(h * scale)
        rw = int(w * scale)
        out = np.zeros((mb, c, rh, rw), dtype=np.float32)
    elif cf.Input_type == 'channels_last':
        mb, h, w, c = img.shape
        rh = int(h * scale)
        rw = int(w * scale)
        out = np.zeros((mb, rh, rw, c), dtype=np.float32)
        
    for i, _img in enumerate(img):
        for _c in range(c):
            if cf.Input_type == 'channels_first':
                _img_c = _img[_c]
            elif cf.Input_type == 'channels_last':
                _img_c = _img[:, :, _c]
                
            _img_c = cv2.resize(_img_c, (rw, rh))
            
            if cf.Input_type == 'channels_first':
                out[i, _c] = _img_c
            elif cf.Input_type == 'channels_last':
                out[i, :, :, _c] = _img_c

    return out

def resize_images_forClass(img, scale=1.0):
    if cf.Input_type == 'channels_first':
        mb, c, h, w = img.shape
        rh = int(h * scale)
        rw = int(w * scale)
        out = np.zeros((mb, c, rh, rw), dtype=np.float32)
    elif cf.Input_type == 'channels_last':
        mb, h, w, c = img.shape
        rh = int(h * scale)
        rw = int(w * scale)
        out = np.zeros((mb, rh, rw, c), dtype=np.float32)
        
    for i, _img in enumerate(img):
        if cf.Input_type == 'channels_first':
            _img = _img.transpose(1, 2, 0)
        _img = cv2.resize(_img, (rw, rh))
        if cf.Input_type == 'channels_first':
            _img = _img[None, ...]
        elif cf.Input_type == 'channels_last':
            _img = _img[..., None]
        out[i] = _img

    return out


def resize_images_forPosi(img, scale=1.0):
    if cf.Input_type == 'channels_first':
        mb, c, h, w = img.shape
        rh = int(h * scale)
        rw = int(w * scale)
        out = np.zeros((mb, c, rh, rw), dtype=np.float32)
    elif cf.Input_type == 'channels_last':
        mb, h, w, c = img.shape
        rh = int(h * scale)
        rw = int(w * scale)
        out = np.zeros((mb, rh, rw, c), dtype=np.float32)
        
    for i, _img in enumerate(img):
        if cf.Input_type == 'channels_last':
            _img = _img.transpose(1, 2, 0)
        _img = cv2.resize(_img, (rw, rh))
        if cf.Input_type == 'channels_first':
            _img = _img[None, ...]
            #_img = _img.transpose(2, 0, 1)
        out[i] = _img
        #out[i] = _img[..., None]

    return out


# Convert ground-truth shape
# from [mb, c, h, w] to [mb, h, w]

def trans_forSoftmax(img):
    if cf.Input_type == 'channels_first':
        mb, c, h, w = img.shape
    elif cf.Input_type == 'channles_last':
        mb, h, w, c = img.shape

    out = np.zeros((mb, h, w), dtype=np.float32)

    for i, _img in enumerate(img):
        if cf.Input_type == 'channels_first':
            _img = _img[0, ...]
        elif cf.Input_type == 'channels_last':
            _img = _img[..., 0]
        out[i] = _img

    return out
