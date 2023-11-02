from tqdm import tqdm
import os
import torch
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr, kendalltau
import timm
from timm.models.swin_transformer import SwinTransformerBlock


import time

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from utilss.util import setup_seed,set_logging,SaveOutput
from script.feature_extract import get_first_feature_swin, get_second_feature_swin,get_third_feature_swin, get_forth_feature_swin
from options.train_options import TrainOptions
from model.swin_1_mediator import Pixel_Prediction, patchSplitting, Mediator #, Transformer
#from model.swin_2_mediator import Pixel_Prediction, patchSplitting, Mediator
from utilss.process_image import ToTensor, RandHorizontalFlip, AddGaussianNoise, RandCrop, RandomRotation, five_point_crop, Normalize
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset


# IQT에 있던 코드를 각각 def 형식으로 수정한 형태를 갖고 있습니다.
# base_options와 train_options에서 정의한 Argument(Config)가 적용되어 학습을 수행합니다.
# if you save the log in other directory, please commend the code,'tensorboard --logdir Thedirectory'

log_dir = './runs/train_3/1102'

writer = SummaryWriter(log_dir)

class Train:
    def __init__(self, config):
        self.opt = config
        self.create_model()
        self.init_saveoutput()
        self.init_data()
        self.criterion = torch.nn.MSELoss() #loss function :  PLCC loss: this could be work.
        self.optimizer = torch.optim.Adam([
        {'params': self.regressor.parameters(), 'lr': self.opt.learning_rate,'weight_decay':self.opt.weight_decay},
        {'params': self.patchSplitting1.parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay},
        #{'params': self.patchSplitting2.parameters(), 'lr': self.opt.learning_rate,'weight_decay': self.opt.weight_decay},
        # {'params': self.patchSplitting3.parameters(), 'lr': self.opt.learning_rate, 'weight_decay':self.opt.weight_decay},
        {'params': self.mediator1.parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay},
        #{'params': self.mediator2.parameters(), 'lr': self.opt.learning_rate, 'weight_decay': self.opt.weight_decay},
        ])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt.T_max, eta_min=self.opt.eta_min)
        self.load_model()
        self.train()

    def create_model(self):

        self.swin= timm.create_model('swin_base_patch4_window7_224', pretrained=True).cuda()
        self.patchSplitting1 = patchSplitting(dim=512).cuda() # 3.28에 추가한 거 for merging
        self.mediator1 = Mediator(in_dim=256).cuda()
        #self.patchSplitting2 = patchSplitting(dim=1024).cuda()  # 3.28에 추가한 거 for merging
        #self.mediator2 = Mediator(in_dim=512).cuda()
        #self.patchSplitting3 = patchSplitting(dim=2048).cuda()
        #self.mediator3 = Mediator(in_dim=1024).cuda()
        self.regressor = Pixel_Prediction().cuda()
        # timm에서 pretrained swin load

    def init_saveoutput(self):
        """swin 의 layer 별 결과를 저장시키는 파트"""
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.swin.modules():
            if isinstance(layer, SwinTransformerBlock):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

    def init_data(self):

        # PIPAL / LIVE / CSIQ /TID2013 /KADID-10k
        """ 사용할 데이터를 아래와 같이 조건문에 따라 불러오도록 구성했습니다.
            데이터를 선택하는 것은 base_options.py에서 지정한 d_name에 따라 결정됩니다.
            혹시 다른 데이터를 더 추가하고 싶다면 아래와 같은 형태로 포맷을 갖춘뒤 파일을 추가해주면 됩니다."""

        if self.opt.d_name=='PIPAL':
            from dataa.pipal import IQADataset
        elif self.opt.d_name== 'LIVE':
            from dataa.live import IQADataset
        elif self.opt.d_name== 'CSIQ':
            from dataa.csiq import IQADataset
        elif self.opt.d_name=='TID':
            from dataa.tid import IQADataset
        elif self.opt.d_name=='KADID':
            from dataa.kadid import IQADataset

        """학습할 데이터셋을 구성합니다."""

        #
        #앞서 불러온 IDADataset에서 지정한 옵션에 따라 데이터를 불러옵니다.
        ###db_path에서 정의한 디렉토리에 있는 데이터를 txt_file_name에 기록된 데이터 및 점수를 기반으로 읽고,
        ###transform을 적용한 뒤, train set인지 validation set인지 구분을 해줍니다.
        ### 기존 코드는 ref data, dst data의 디렉토리를 따로 지정했으나, 여기선 통합 디렉토리로 바꿨습니다.
        ### train set과 validation set의 구분을 위해 trainset을 8:2로 쪼개도록 변경했습니다.
        ### validation set은 Tensor변환 외에는 추가적인 변환을 적용하지 않았습니다.
        ##

        train_dataset = IQADataset(
            db_path = self.opt.db_path,
            txt_file_name=self.opt.txt_file_name,
            transform=transforms.Compose(
                [
                    RandCrop(self.opt.crop_size, self.opt.num_crop),
                    RandHorizontalFlip(),
                    ToTensor(),
                ]
            ),
            train_mode=True,
            test_mode=False,
        )

        val_dataset = IQADataset(
            db_path = self.opt.db_path,
            txt_file_name=self.opt.txt_file_name,
            transform=transforms.Compose(
                [
                    #RandCrop(self.opt.crop_size, self.opt.num_crop),
                    ToTensor(),
                ]
            ),
            train_mode=False,
            test_mode = False,
            resize=False  # size가 맞지 않는 경우 True로 변환
        )

        """test_dataset = IQADataset(
            db_path=self.opt.db_path,
            txt_file_name= self.opt.txt_file_name,
            train_mode= False,
            transform= False,
            test_mode= True,
            resize = False # 여기도 LIVE때문에 설정 필요 다른 데이터는 사이즈 일정한 것을 보임.
        )"""

        print('number of train scenes: {}'.format(len(train_dataset)))
        print('number of val scenes: {}'.format(len(val_dataset)))


        # train_loader / val_loader는 각각 위에서 만든 train_dataset, val_dataset을 받아
        # 배치사이즈 만큼(데이터 양),그리고 num_workers만큼(데이터 로딩 속도) 코어를 사용해 데이터를 불러옵니다.

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=True
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.opt.batch_size,
            num_workers=self.opt.num_workers,
            drop_last=True,
            shuffle=False
        )

        """self.test_loader = DataLoader(
            dataset= test_dataset,
            batch_size= self.opt.batch_size,
            num_workers= self.opt.num_workers,
            drop_last= True,
            shuffle= False
        )"""

    # 기존 load model은 load_epoch가 -1이라는 의미는 최근 epoch를 학습한 모델을 불러온다는 의미였습니다.
    # 즉, epoch 50 모델인 경우,  50epoch까지 학습한 모델을 불러온다는 의미였는데, 이 부분도 수정을 했습니다.
    # 특정 데이터에 대해 학습한 모델 중 best model이 있는 경우 불러오도록 합니다.
    # best model이 없는 경우에만 epoch가 가장 높은 학습 모델을 불러오도록 합니다.
    # 만약 아무 모델도 없을 경우, load_epoch는 0으로 지정하고 그냥 pass하게 됩니다.
    def load_model(self):
        models_dir = self.opt.checkpoints_dir
        if os.path.exists(models_dir):
            if self.opt.load_epoch == -1:

                for file in os.listdir(models_dir):
                    if file.split('.')[0].endswith('{}_best'.format(self.opt.d_name)):
                        # fine tuning을 위해 사용할 영역.
                        # 모델에 따른 best모델을 불러와서 학습을 계속 진행할 수 있습니다.
                        print('best model is loading...')
                        checkpoint = torch.load(os.path.join(models_dir + '{}_best.pth'.format(self.opt.d_name)))
                        print(checkpoint['epoch'])
                        self.opt.load_epoch = checkpoint['epoch']
                        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                        self.patchSplitting1.load_state_dict(checkpoint['patchSplitting1_state_dict'])
                        #self.patchSplitting2.load_state_dict(checkpoint['patchSplitting2_state_dict'])
                        #self.patchSplitting3.load_state_dict(checkpoint['patchSplitting3_state_dict'])
                        self.mediator1.load_state_dict(checkpoint['mediator1_state_dict'])
                        #self.mediator2.load_state_dict(checkpoint['mediator2_state_dict'])
                        #self.mediator3.load_state_dict(checkpoint['mediator3_state_dict'])
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        self.start_epoch = checkpoint['epoch'] #for finetuning, start training from epoch 0
                        loss = checkpoint['loss']

                        break

                    elif file.startswith("{}".format(self.opt.d_name)):
                        print(file)
                        load_epoch = 0
                        #print(file.split('.')[0].split('_')[2])
                        load_epoch = max(load_epoch, int(file.split('.')[0].split('_')[2]))
                        print('{}th model is loading...'.format(load_epoch))
                        self.opt.load_epoch = load_epoch
                        checkpoint = torch.load(os.path.join(models_dir, "{}_epoch_".format(self.opt.d_name) + str(self.opt.load_epoch) + ".pth"))
                        self.regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
                        self.patchSplitting1.load_state_dict(checkpoint['patchSplitting1_state_dict'])
                        #self.patchSplitting2.load_state_dict(checkpoint['patchSplitting2_state_dict'])
                        #self.patchSplitting3.load_state_dict(checkpoint['patchSplitting3_state_dict'])
                        self.mediator1.load_state_dict(checkpoint['mediator1_state_dict'])
                        #self.mediator2.load_state_dict(checkpoint['mediator2_state_dict'])
                        #self.mediator3.load_state_dict(checkpoint['mediatpr3_state_dict'])
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                        self.start_epoch = checkpoint['epoch']
                        loss = checkpoint['loss']

                        break
                    else:
                        print('No model. So I will train from zero.')
                        self.opt.load_epoch = 0

            else:
                print('No model. So I will train from zero.')
                self.opt.load_epoch = 0
        else:
            print('No Directory. So I will train from zero.')
            self.opt.load_epoch = 0

    """본격적인 train의 진행 vit와 resnet은 eval모드로 진행하므로 학습을 따로 진행하지 않고,
        실질적인 학습은 deform_net과 regressor가 학습한다고 볼 수 있습니다."""

    def train_epoch(self, epoch):
        losses = []
        self.regressor.train()
        self.patchSplitting1.train()
        #self.patchSplitting2.train()
        #self.patchSplitting3.train()
        self.mediator1.train()
        #self.mediator2.train()
        #self.mediator3.train()
        self.swin.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in tqdm(self.train_loader):
            # tqdm은 학습 진행과정을 간단하게 시각화해주는 라이브러리입니다.
            # train_loader를 tqdm으로 덮어주는 방법으로 시각화를 할 수 있습니다.
            d_img_org = data['d_img_org'].cuda()
            r_img_org = data['r_img_org'].cuda()
            labels = data['score']
            labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

            _x = self.swin(d_img_org)
            first_dis = get_first_feature_swin(self.save_output) # 56,56
            second_dis = get_second_feature_swin(self.save_output) # 28,28
            #third_dis = get_third_feature_swin(self.save_output) # 14,14
            #forth_dis = get_forth_feature_swin(self.save_output)
            self.save_output.outputs.clear()


            _y = self.swin(r_img_org)
            first_ref = get_first_feature_swin(self.save_output) # 56,56
            second_ref = get_second_feature_swin(self.save_output) # 28,28
            #third_ref = get_third_feature_swin(self.save_output) #14,14
            #forth_ref = get_forth_feature_swin(self.save_output)
            self.save_output.outputs.clear()

            #B, N, C = forth_ref.shape
            #H, W = 7, 7
            #forth_ref = forth_ref.transpose(1, 2).view(B, C, H, W)
            #forth_dis = forth_dis.transpose(1, 2).view(B, C, H, W)

            #B, N, C = third_ref.shape
            #H, W = 14, 14
            #third_ref = third_ref.transpose(1, 2).view(B, C, H, W)
            #third_dis = third_dis.transpose(1, 2).view(B, C, H, W)

            B, N, C = second_ref.shape
            H, W = 28, 28
            second_ref = second_ref.transpose(1, 2).view(B, C, H, W)
            second_dis = second_dis.transpose(1, 2).view(B, C, H, W)

            B, N, C = first_ref.shape
            H,W = 56,56
            first_ref = first_ref.transpose(1, 2).view(B, C, H, W)
            first_dis = first_dis.transpose(1, 2).view(B, C, H, W)

            "Patch Splitting_3:  (2048,7,7) --> (1024,14,14) "

            #forth_dis = self.patchSplitting3(forth_dis)
            #forth_ref = self.patchSplitting3(forth_ref)

            #third_dis = self.mediator3(forth_dis, third_ref)
            #third_ref = self.mediator3(forth_ref, third_ref)

            "Patch Splitting_2:  (1024,14,14) --> (512,28,28) "

            #third_dis = self.patchSplitting2(third_dis)
            #third_ref = self.patchSplitting2(third_ref)

            #second_dis = self.mediator2(third_dis, second_ref)
            #second_ref = self.mediator2(third_ref, second_ref)


            "PatchSpliting_1: (512,28,28) --> (256,56,56)"

            second_dis = self.patchSplitting1(second_dis)
            second_ref = self.patchSplitting1(second_ref)

            first_dis_ = self.mediator1(second_dis, first_ref) # 512, 28, 28 , 잔차
            first_ref_ = self.mediator1(second_ref, first_ref)



            pred = self.regressor(first_dis_, first_ref_, first_dis, first_ref)

            self.optimizer.zero_grad()
            loss = self.criterion(torch.squeeze(pred), labels)

            losses.append(loss.item())

            # backprop.및 optimizer step 진행 합니다.

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)

        # compute correlation coefficient
        # KRCC도  계산하도록 코드를 추가했으나, 현 단계(2월24일 기준)에선 사용하지 않습니다.
        rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
        #rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

        lr = self.optimizer.param_groups[0]['lr'] # LR :{:.6} . lr


        logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4}  / PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        print('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, np.mean(losses),rho_s,rho_p))
        return np.mean(losses), rho_s, rho_p

    def train(self):
        # 위에서 만든 train_epoch가 실질적인 train 코드였다면,
        # 이 train은 epoch을 진행하면서 loss 계산 후 비교하는 코드입니다.
        best_srocc = 0
        best_plcc = 0
        #best_krcc = 0
        for epoch in range(self.opt.load_epoch, self.opt.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            print('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = self.train_epoch(epoch) # loss val -> loss_tr. loss가 어차피 validation loss update로 진행됨

            if (epoch + 1) % self.opt.val_freq == 0:
                logging.info('Starting eval...')
                logging.info('Running testing in epoch {}'.format(epoch + 1))
                print('Starting eval')
                print('Running val. in epoch {}'.format(epoch + 1))
                loss, rho_s, rho_p = self.eval_epoch(epoch)

                writer.add_scalar('loss/val', loss, epoch)



                print('Eval done!')
                logging.info('Eval done...')

                if rho_s> best_srocc or rho_p > best_plcc :
                    best_srocc = rho_s
                    best_plcc = rho_p
                    #best_srcc = rho_k
                    logging.info('Best now')
                    print('Best now')
                    self.save_model( epoch, "{}_epoch_{}.pth".format(self.opt.d_name, epoch+1), loss, rho_s, rho_p)
                    self.save_model(epoch, "{}_best.pth".format(self.opt.d_name), loss, rho_s, rho_p)

                if epoch % self.opt.save_interval == 0:
                    weights_file_name = "{}_epoch_%d.pth".format(self.opt.d_name) % (epoch+1)
                    self.save_model( epoch, weights_file_name, loss ,rho_s, rho_p)

            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
            print('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
        writer.flush()

    def eval_epoch(self, epoch):
        with torch.no_grad():
            losses = []
            # 구조는 train_epoch와 동일해 보이겠지만, no_grad로 진행하기 때문에 학습은 X
            # loss는 전달되므로, loss 갱신되면 다음 train에서 학습 진행
            self.swin.eval()
            self.patchSplitting1.train()
            #self.patchSplitting2.train()
            #self.patchSplitting3.train()
            self.mediator1.train()
            #self.mediator2.train()
            #self.mediator3.train()
            self.regressor.train()

            # save data for one epoch
            pred_epoch = []
            labels_epoch = []

            for data in tqdm(self.val_loader):
                pred = 0
                for i in range(self.opt.ensemble):
                    d_img_org = data['d_img_org'].cuda()
                    r_img_org = data['r_img_org'].cuda()
                    labels = data['score']
                    labels = torch.squeeze(labels.type(torch.FloatTensor)).cuda()

                    # five_point_crop
                    d_img_org, r_img_org = five_point_crop(i, d_img=d_img_org, r_img=r_img_org, config=self.opt)

                    # get backbone feature map
                    _x = self.swin(d_img_org)
                    first_dis = get_first_feature_swin(self.save_output)  # 56,56
                    second_dis = get_second_feature_swin(self.save_output)  # 28,28
                    #third_dis = get_third_feature_swin(self.save_output)
                    #forth_dis = get_forth_feature_swin(self.save_output)

                    self.save_output.outputs.clear()

                    _y = self.swin(r_img_org)
                    first_ref = get_first_feature_swin(self.save_output)  # 56,56
                    second_ref = get_second_feature_swin(self.save_output)  # 28,28
                    #third_ref = get_third_feature_swin(self.save_output)
                    #forth_ref = get_forth_feature_swin(self.save_output)

                    self.save_output.outputs.clear()

                    #B, N, C = forth_ref.shape
                    #H, W = 7, 7
                    #forth_ref = forth_ref.transpose(1, 2).view(B, C, H, W)
                    #forth_dis = forth_dis.transpose(1, 2).view(B, C, H, W)

                    #B, N, C = third_ref.shape
                    #H, W = 14, 14
                    #third_ref = third_ref.transpose(1, 2).view(B, C, H, W)
                    #third_dis = third_dis.transpose(1, 2).view(B, C, H, W)

                    B, N, C = second_ref.shape
                    H, W = 28, 28
                    second_ref = second_ref.transpose(1, 2).view(B, C, H, W)
                    second_dis = second_dis.transpose(1, 2).view(B, C, H, W)

                    B, N, C = first_ref.shape
                    H, W = 56, 56
                    first_ref = first_ref.transpose(1, 2).view(B, C, H, W)
                    first_dis = first_dis.transpose(1, 2).view(B, C, H, W)

                    "Patch Splitting_3:  (2048,7,7) --> (1024,14,14) "

                    #forth_dis = self.patchSplitting3(forth_dis)
                    #forth_ref = self.patchSplitting3(forth_ref)

                    #third_dis = self.mediator3(forth_dis, third_ref)
                    #third_ref = self.mediator3(forth_ref, third_ref)

                    "Patch Splitting_2:  (1024,14,14) --> (512,28,28) "

                    #third_dis = self.patchSplitting2(third_dis)
                    #third_ref = self.patchSplitting2(third_ref)

                    #second_dis = self.mediator2(third_dis, second_ref)
                    #second_ref = self.mediator2(third_ref, second_ref)

                    "PatchSpliting_1: (512,28,28) --> (256,56,56)"

                    second_dis = self.patchSplitting1(second_dis)
                    second_ref = self.patchSplitting1(second_ref)

                    first_dis_ = self.mediator1(second_dis, first_ref)  # 512, 28, 28 , 잔차
                    first_ref_ = self.mediator1(second_ref, first_ref)


                    pred += self.regressor(first_dis_, first_ref_, first_dis, first_ref) #딥에서 쉘로우로



                pred /= self.opt.ensemble

                # compute loss
                loss = self.criterion(torch.squeeze(pred), labels)
                loss_val = loss.item()
                losses.append(loss_val)

                # save results in one epoch
                pred_batch_numpy = pred.data.cpu().numpy()
                labels_batch_numpy = labels.data.cpu().numpy()
                pred_epoch = np.append(pred_epoch, pred_batch_numpy)
                #print(pred_epoch)
                labels_epoch = np.append(labels_epoch, labels_batch_numpy)
                #print(labels_epoch)

            # compute correlation coefficient
            rho_s, _ = spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))
            #rho_k, _ = kendalltau(np.squeeze(pred_epoch), np.squeeze(labels_epoch)) # 추가 한 KRCC
            rho_p, _ = pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))

            logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s , rho_p))
            print('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
            return np.mean(losses), rho_s, rho_p


    def save_model(self, epoch, weights_file_name, loss, rho_s, rho_p):
        """학습 모델에서 어떤 것을 저장할 지 지정합니다."""

        print('-------------saving weights---------')
        weights_file = os.path.join(self.opt.checkpoints_dir, weights_file_name)
        torch.save({
            'epoch': epoch,
            'regressor_model_state_dict': self.regressor.state_dict(),
            'patchSplitting1_state_dict': self.patchSplitting1.state_dict(),
            #'patchSplitting2_state_dict': self.patchSplitting2.state_dict(),
            #'patchSplitting3_state_dict': self.patchSplitting3.state_dict(),
            'mediator1_state_dict': self.mediator1.state_dict(),
            #'mediator2_state_dict': self.mediator2.state_dict(),
            #'mediator3_state_dict': self.mediator3.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }, weights_file)
        logging.info('Saving weights and model of epoch{}, SRCC:{:.4} ,PLCC:{:.4}'.format(epoch+1, rho_s , rho_p))
        print('Saving weights and model of epoch{}, SRCC:{:.4} ,PLCC:{:.4}'.format(epoch+1, rho_s , rho_p))

# 이전에 발표한 바와 같이 if __name__ == 'main' 조건을 두어 multi processing을 수행할 수 있습니다.(num_workers=2)
if __name__ == '__main__':
    config = TrainOptions().parse()
    config.checkpoints_dir = os.path.join(config.checkpoints_dir) # , config.name
    setup_seed(config.seed)
    set_logging(config)
    # logging.info(config)
    Train(config)
    writer.close()
