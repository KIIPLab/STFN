from tqdm import tqdm
import os
import torch
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr
import timm
from timm.models.swin_transformer import SwinTransformerBlock

from dataa.Testmode import IQADataset


from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr, kendalltau

from utilss.util import setup_seed, set_logging, SaveOutput
from script.copy_extract import get_first_feature_swin, get_second_feature_swin,get_third_feature_swin
from options.test_options import TestOptions
from options.train_options import TrainOptions
from model.swin_1_mediator import Pixel_Prediction ,patchSplitting , Mediator
from utilss.process_image import ToTensor, RandHorizontalFlip, RandCrop, crop_image, Normalize, five_point_crop


import matplotlib.pyplot as plt


class Test:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.load_model()
        self.init_data()
        self.test()

    # model 생성.  train 때와 동일.
    def create_model(self):
        """모델을 불러오고 구성하는 함수. backbone에 해당하는 resnet과 vit를 불러오고, 둘을 받는 모델을 load합니다."""
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True).cuda()
        self.patchSplitting1 = patchSplitting(dim=512).cuda()  # 3.28에 추가한 거 for merging
        self.mediator1 = Mediator(in_dim=256).cuda()
        self.regressor = Pixel_Prediction().cuda()

    def init_saveoutput(self):
        """resnet과 vit의 layer 별 결과를 저장시키는 파트"""
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.swin.modules():
            if isinstance(layer, SwinTransformerBlock):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)


    def load_model(self):
        print('loading model info...')
        models_dir = self.opt.checkpoints_dir
        #########################################################################
        #model_dir 내에 있는 파일 중, best model 및 epoch 진행이 가장 많이 된 모델을 찾아 정보를 load.
        if os.path.exists(models_dir):

            for file in os.listdir(models_dir):

                if file.split('.')[0].endswith('{}_best'.format(self.opt.d_name)):
                    print('best model is loaded.')
                    print(models_dir)
                    checkpoint = torch.load(os.path.join(models_dir, '{}_best.pth'.format(self.opt.d_name)))
                    self.opt.load_epoch = checkpoint['epoch']
                    self.regressor.load_state_dict((checkpoint['regressor_model_state_dict']))
                    self.patchSplitting1.load_state_dict(checkpoint['patchmerging1_state_dict'])
                    self.mediator1.load_state_dict(checkpoint['mediator1_state_dict'])
                    self.start_epoch = checkpoint['epoch'] + 1
                    loss = checkpoint['loss']
                    break

                elif file.split('.')[0].startswith('{}'.format(self.opt.d_name)):
                    print('no best model, but last model loaded.')
                    load_epoch = 0
                    if file.split('_')[1].startswith('epoch'):

                        load_epoch = max(load_epoch, int(file.split('_')[2].split('.')[0]))
                        self.opt.load_epoch = load_epoch
                        checkpoint = torch.load(os.path.join(models_dir, "{}_epoch_".format(self.opt.d_name) + str(
                            self.opt.load_epoch) + ".pth"))
                        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                        self.patchSplitting1.load_state_dict(checkpoint['patchmerging1_state_dict'])
                        self.mediator1.load_state_dict(checkpoint['mediator1_state_dict'])
                        self.start_epoch = checkpoint['epoch'] + 1
                        loss = checkpoint['loss']
                        break


        else:
            print('there is no trained model. please check the dir. ')
        #####################################################################################################

    # data loader 구성.
    def init_data(self):

        if self.opt.d_name == 'LIVE':
            from dataa.live import IQADataset
        elif self.opt.d_name == 'CSIQ':
            from dataa.csiq import IQADataset
        elif self.opt.d_name == 'TID':
            from dataa.tid import IQADataset
        elif self.opt.d_name == 'KADID':
            from dataa.kadid import IQADataset

        print('initialize data...')
        test_dataset = IQADataset(
            db_path=self.opt.db_path,
            txt_file_name=self.opt.txt_file_name,
            train_mode=False,
            transform=False,
            test_mode = True,
            resize = False # LIVE일 때는 True

        )
        logging.info('number of test scenes: {}'.format(len(test_dataset)))

        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=False
        )


    def test(self):
        print('testing!')
        f = open(os.path.join(self.opt.checkpoints_dir, self.opt.test_file_name), 'w')
        with torch.no_grad():

            total_pred=[]
            total_label=[]

            for data in tqdm(self.test_loader):
                d_img_org = data['d_img_org'].cuda()
                r_img_org = data['r_img_org'].cuda()
                #d_img_name = data['d_img_name']
                pred = 0
                if self.opt.test_mode == True:
                    labels = data['score']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()
                else:
                    labels=0

                for i in range(self.opt.n_ensemble):
                    b, c, h, w = r_img_org.size()
                    if self.opt.n_ensemble > 9:
                        new_h = config.crop_size
                        new_w = config.crop_size
                        top = np.random.randint(0, h - new_h)
                        left = np.random.randint(0, w - new_w)
                        r_img = r_img_org[:, :, top: top + new_h, left: left + new_w]
                        d_img = d_img_org[:, :, top: top + new_h, left: left + new_w]
                    elif self.opt.n_ensemble == 1:
                        r_img = r_img_org
                        d_img = d_img_org
                    else:
                        d_img, r_img = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)
                    d_img = d_img.cuda()
                    r_img = r_img.cuda()

                    _x = self.swin(d_img)
                    first_dis = get_first_feature_swin(self.save_output)  # 56,56
                    second_dis = get_second_feature_swin(self.save_output)  # 28,28

                    self.save_output.outputs.clear()

                    _y = self.swin(r_img)
                    first_ref = get_first_feature_swin(self.save_output)  # 56,56
                    second_ref = get_second_feature_swin(self.save_output)  # 28,28

                    self.save_output.outputs.clear()

                    B, N, C = first_ref.shape
                    H, W = 56, 56
                    first_ref = first_ref.transpose(1, 2).view(B, C, H, W)
                    first_dis = first_dis.transpose(1, 2).view(B, C, H, W)

                    B, N, C = second_ref.shape
                    H, W = 28, 28
                    second_ref = second_ref.transpose(1, 2).view(B, C, H, W)
                    second_dis = second_dis.transpose(1, 2).view(B, C, H, W)


                    "patchSplitting_1: (256,56,56) --> (512,28,28)"
                    second_dis = self.patchSplitting1(second_dis)
                    second_ref = self.patchSplitting1(second_ref)

                    first_dis_ = self.mediator1(second_dis, first_ref)  # 512, 28, 28 , 잔차 ()
                    first_ref_ = self.mediator1(second_ref, first_ref)


                    pred += self.regressor(first_dis_, first_ref_, first_dis, first_ref)

                pred /= self.opt.n_ensemble

                if self.opt.test_mode == False:
                    pass
                #    for i in range(len(d_img_name)):
                    #pred score 기록
                #        line = "%s,%f\n" % (d_img_name[i], float(pred.squeeze()[i]))
                #        f.write(line)


                else:
                    # test mode에서는 epoch이 필요 없으므로 변수는 epoch이 붙지만 사실상 1큐 진행.
                    pred_epoch = pred.data.cpu().numpy()
                    labels_epoch = labels.data.cpu().numpy()  # 10열의 데이터가 나오는데 이는 batch size만큼 불러온 데이터의 모음
                    total_pred = np.append(total_pred, pred_epoch)
                    total_label = np.append(total_label, labels_epoch)

                    print(total_pred)


            if self.opt.test_mode == True:

                #전체 데이터에 대한 예측 값의 평균, 라벨 값의 평균, srcc, krcc, plcc 기록.

                rho_s, _ = spearmanr(np.squeeze(total_pred), np.squeeze(total_label))
                rho_k, _ = kendalltau(np.squeeze(total_pred), np.squeeze(total_label))
                rho_p, _ = pearsonr(np.squeeze(total_pred), np.squeeze(total_label))

                line = '{} / SRCC:{:.4} / KRCC:{:.4} / PLCC:{:.4}\n'.format(
                    self.opt.d_name, rho_s, rho_k, rho_p)

                import scipy.io

                scipy.io.savemat('tid_score.mat', {'score':total_pred})

                f.write(line)



        f.close()


if __name__ == '__main__':
    config = TestOptions().parse()
    #config = TrainOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir)
    setup_seed(config.seed)
    set_logging(config)
    Test(config)
