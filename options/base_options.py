import argparse
import os
from utilss import util
import torch

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        # dataset path
        # train 디렉토리 설정: PIPAL | LIVE | CSIQ | TID | KADID| 중에 고르도록 디렉토리 준비하기
        # 주의할 점은 AHIQ는 validation dataset을 tset로 안쓰고 있다는 점.
        # 기존 코드는  Validation Dataset에 대해 label이 붙어 있었으나, 공식적으론 구할 방법이 없기 때문에
        # label이 없다고 가정하고, train Dataset을 쪼개서 validation으로 활용.
        # txt_file_name은 train Dataset에 대한 label로 기능

        self._parser.add_argument('--d_name', type=str, default='PIPAL', help='data name for train: PIPAL | KADID | LIVE | CSIQ | TID ')
        self._parser.add_argument('--db_path', type=str, default='dataa/PIPAL/Train_images', help='path to train images : PIPAL | KADID | LIVE | CSIQ | TID in dir')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/', help='models are saved here') # KADID | PIPAL 별로 구분해서 지정하기
        self._parser.add_argument('--txt_file_name', type=str, default='data_list/PIPAL_test.txt', help='training data list: PIPAL | KADID | LIVE | CSIQ | TID')
        self._parser.add_argument('--train_size', type= int, default= 0.8, help= 'train-val ratio') # 이렇게 하면 다양한 데이터에 대해 학습 용이
        # experiment
        self._parser.add_argument('--name', type=str, default='test_ahiq_pipal',
                                  help='name of the experiment. It decides where to store samples and models')
        # device
        # 코어/2 만큼의 num_workers를 할 수 있기 때문에 main core - sub core 조건을 걸어서 진행하면 num_workers를 0으로 하지 않아도 된다.
        # num workers를 늘릴면 좋은점? 더 빨리 데이터를 load해서 보다 빠른 trainig을 진행할 수 있다.
        self._parser.add_argument('--num_workers', type=int, default=8, help='total workers')
        # model
        # path size는 기존 paper를 가지고 보면 8과 16에서 좋은 성능을 볼 수 있었으므로, 둘 중 하나를 샘플로 사용한다.
        # laod_epoch가 사실 거슬리는 파트 중 하나. 본 코드는 IQT를 베이스로 만들었지만 이 부분은 없던 부분이라 더욱 그렇다.
        self._parser.add_argument('--patch_size', type=int, default=8, help='patch size of Vision Transformer')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--ckpt', type=str, default='./checkpoints', help='models to be loaded') # 모델을 어디서 불러올 것인지 설정하는 파트
        self._parser.add_argument('--seed', type=int, default=1777, help='random seed')
        #data process
        self._parser.add_argument('--crop_size', type=int, default=224, help='image size')
        self._parser.add_argument('--num_crop', type=int, default=1, help='random crop times')
        self._parser.add_argument('--ensemble', type=int, default=5, help='ensemble ways of validation')
        
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        #self._opt = self._parser.parse_args()
        self._opt = self._parser.parse_known_args()[0]

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._set_and_check_load_epoch()

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir) #, self._opt.name
        if os.path.exists(models_dir):
            #print('#############')
            if self._opt.load_epoch == -1:
                #print('@@@@@@@@@@@')
                load_epoch = -1
                for file in os.listdir(models_dir):
                    # 위 디렉토리에 있는 파일들 중에서..
                    if file.split('.')[0].endswith('{}_best'.format(self._opt.d_name)):
                        checkpoint = torch.load(os.path.join(models_dir + '{}_best.pth'.format(self._opt.d_name)))
                        load_epoch = checkpoint['epoch']
                        loss = checkpoint['loss']

                    #elif file.split('.')[0].split('_')[0].endswith('{}'.format(self._opt.d_name)):
                    #    checkpoint = torch.load(os.path.join(models_dir, "{}_epoch_".format(self._opt.d_name) + str(
                    #        self._opt.load_epoch) + ".pth"))
                    #    load_epoch= checkpoint['epoch']
                    #    loss = checkpoint['loss']

                        #print(load_epoch)
                self._opt.load_epoch = load_epoch
            else:
                self._opt.load_epoch = 0
        else:
            self._opt.load_epoch = 0

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir) #, self._opt.name
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')