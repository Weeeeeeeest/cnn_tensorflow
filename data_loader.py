import os
import glob
import cv2
import numpy as np

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
            
        print()
        print('------------')
        print('Data Load (phase: {})'.format(self.phase))
        
        for dir_path in dir_paths:
            
            files = glob.glob(dir_path + '/*')

            load_count = 0
            for img_path in files:
                #img_path = os.path.join(dir_path, img)

                gt = self.get_gt(img_path)
                if self.gt_count[gt] >= 10000:
                    continue

                img = self.load_image(img_path)
                gt_path = 1
                
                data = {'img_path': img_path,
                        'gt_path': gt_path,
                        'h_flip': False,
                        'v_flip': False
                }
                
                self.datas.append(data)

                self.gt_count[gt] += 1
                load_count += 1

            print(' - {} - {} datas -> loaded {}'.format(dir_path, len(files), load_count))

        self.display_gt_statistic()
                
        if self.phase == 'Train':
            self.data_augmentation(h_flip=cf.Horizontal_flip, v_flip=cf.Vertical_flip)
            
            self.display_gt_statistic()

        
        self.set_index(shuffle=shuffle)
                

        
    def display_data_total(self):

        print('   Total data: {}'.format(len(self.datas)))

        
    def display_gt_statistic(self):

        print()
        print('  -*- Training label  -*-')
        self.display_data_total()
        
        for i, gt in enumerate(self.gt_count):
            print('   - {} : {}'.format(cf.Class_label[i], gt))

            
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
            #if cf.Variable_input:
            #    mb = 1
            #else:
            #    mb = cf.Test_Minibatch if cf.Test_Minibatch is not None else self.data_n

                
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
            #mb = cf.Minibatch
            mb = 1
            #if cf.Variable_input:
            #    mb = 1
            #else:
            #    mb = cf.Test_Minibatch if cf.Test_Minibatch is not None else self.data_n
        

        self.get_minibatch_index(shuffle=shuffle)

        if cf.Variable_input:
            imgs = np.zeros((mb, cf.Max_side, cf.Max_side, 3), dtype=np.float32)
        else:
            imgs = np.zeros((mb, cf.Height, cf.Width, 3), dtype=np.float32)
            
        gts = np.zeros((mb, cf.Class_num), dtype=np.float32)
        
        for i, ind in enumerate(self.mb_inds):
            data = self.datas[ind]
            img, height, width = self.load_image(data['img_path'], h_flip=data['h_flip'])
            #gt = self.load_image(data['gt_path'])

            gt = self.get_gt(data['img_path'])
                
            imgs[i, :height, :width, :] = img
            gts[i, gt] = 1

            if Dataset_Debug_Display:
                import matplotlib.pyplot as plt
                plt.imshow(imgs[i])
                plt.show()

        if self.phase == 'Test' and cf.Variable_input:
            imgs = imgs[:, :height, :width, :]

        return imgs, gts


    
    def get_gt(self, img_name):

        for ind, cls in enumerate(cf.Class_label):
            if cls in img_name:
                return ind

        raise Exception("Class label Error {}".format(img_name))
                        

    ## Below functions are for data augmentation

    def load_image(self, img_name, h_flip=False, v_flip=False):

        ## Image load
        
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

        ## Horizontal flip
        if h_flip:
            img = img[:, ::-1, :]

        ## Vertical flip
        if v_flip:
            img = img[::-1, :, :]

        return img, scaled_height, scaled_width


    
    def data_augmentation(self, h_flip=False, v_flip=False):

        print()
        print('   ||   -*- Data Augmentation -*-')
        if h_flip:
            self.add_horizontal_flip()
            print('   ||    - Added horizontal flip')
        if v_flip:
            self.add_vertical_flip()
            print('   ||    - Added vertival flip')
        print('  \  /')
        print('   \/')
        
    

    def add_horizontal_flip(self):

        ## Add Horizontal flipped image data
        
        new_data = []
        
        for data in self.datas:
            _data = {'img_path': data['img_path'],
                     'gt_path': data['gt_path'],
                     'h_flip': True,
                     'v_flip': data['v_flip']
            }

            new_data.append(_data)

            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1
            
        self.datas.extend(new_data)


        
    def add_vertical_flip(self):

        ## Add Horizontal flipped image data
        
        new_data = []
        
        for data in self.datas:
            _data = {'img_path': data['img_path'],
                     'gt_path': data['gt_path'],
                     'h_flip': data['h_flip'],
                     'v_flip': True
            }

            new_data.append(_data)

            gt = self.get_gt(data['img_path'])
            self.gt_count[gt] += 1
            
        self.datas.extend(new_data)
