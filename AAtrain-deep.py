import torch
from torch.autograd import Variable
import argparse
from datetime import datetime

from lib.binet import ADF_Net

from utils.dataloader import get_loader, test_dataset
from utils.utils import AvgMeter
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from test_isic import mean_dice_np, mean_iou_np
import os
import random
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

net=ADF_Net
name="ADF_Net"
year="2016"
size="2016"
BCE = torch.nn.BCEWithLogitsLoss()
b=str()
def mean_sensitivity_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection
    mask_GT=np.sum(np.abs(y_true), axis=axes)
    smooth = .001
    se = (intersection + smooth) / (mask_GT + smooth)
    return se

def get_specificity(SR, GT, threshold=0.5):
    SR = torch.from_numpy(SR)
    GT = torch.from_numpy(GT)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0) & (GT == 0))
    FP = ((SR == 1) & (GT == 0))

    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)

    return SP

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()



def get_JS(SR, GT, threshold=0.5):
    # JS : Jaccard similarity
    SR = torch.from_numpy(SR)
    GT = torch.from_numpy(GT)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    Inter = torch.sum((SR.byte() + GT.byte()) == 2)
    Union = torch.sum((SR.byte() + GT.byte()) >= 1)

    JS = float(Inter) / (float(Union) + 1e-6)


    return JS

def train(train_loader, model, optimizer, epoch, best_loss):
    model.train()
    loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    accum = 0
    for i, pack in enumerate(train_loader, start=1):
        # ---- data prepare ----
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # ---- forward ----
        map1,map2,out= model(images)
        loss4 = structure_loss(map1, gts)  #
        loss3 = structure_loss(map2, gts)  #
        loss2 = structure_loss(out, gts)  #

        # ---- loss function ----
        loss = 0.5 * loss2 + 0.3 * loss3 + 0.2 * loss4

        # ---- backward ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # ---- recording loss ----
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss2.data, opt.batchsize)
        loss_record4.update(loss2.data, opt.batchsize)
        # loss_record5.update(loss5.data, opt.batchsize)
        # ---- train visualization ----
        # if i % 20 == 0 or i == total_step:
        #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
        #           '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.
        #           format(datetime.now(), epoch, opt.epoch, i, total_step,
        #                  loss_record2.show(), loss_record3.show(), loss_record4.show()))
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  '[lateral-2: {:.4f}, lateral-3: {:0.9f}, lateral-4: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record2.show(), optimizer.state_dict()['param_groups'][0]['lr'], loss_record4.show()))



    save_path = 'snapshots/{}/'.format(opt.train_save)
    os.makedirs(save_path, exist_ok=True)
    # if (epoch + 1) % 1 == 0:
    #     meanloss = tes(model, opt.test_path)
    #     if meanloss > best_loss:
    #         print('new best score:', meanloss)
    #         best_loss = meanloss
    #         torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
    #         print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth' % epoch)
    if epoch==1:
        best_loss=0
    if (epoch + 1) % 1 == 0:
        meanloss,meandc = tes(model, opt.test_path)
        if meanloss > best_loss:
            print('new best score:', meanloss)
            best_loss = meanloss
            torch.save(model.state_dict(), save_path + year + name + '-%d.pth' % epoch)
            print('[Saving Snapshot:]', save_path + year + name + '-%d.pth' % epoch)



    return best_loss


def tes(model, path):
    model.eval()
    mean_loss = []
    mean_DC = []
    if year=="2016":
        vallist=['test']
    else:
        vallist = ['val','test']

    for s in vallist:
        # image_root = '{}/data_{}320.npy'.format(path, s)#for s in ['val', 'test']:
        # gt_root = '{}/mask_{}320.npy'.format(path, s)
        # image_root = '{}/data_{}{}.npy'.format(path, s,size)  # for s in ['val', 'test']:
        # gt_root = '{}/mask_{}{}.npy'.format(path, s,size)
        image_root = '{}/data_{}{}.npy'.format(path, s,size)#for s in ['val', 'test']:
        gt_root = '{}/mask_{}{}.npy'.format(path, s,size)

        test_loader = test_dataset(image_root, gt_root)

        dice_bank = []
        iou_bank = []
        loss_bank = []
        acc_bank = []

        se_bank = []
        sp_bank = []
        score_bank=[]
        JS_bank = []

        for i in range(test_loader.size):
            image, gt = test_loader.load_data()
            image = image.cuda()
            gg=torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda()

            with torch.no_grad():
                # res= model(image)
                map1,map2,res=model(image)
                # res, _, _, _, _, _, _, _, _, _ = model(image)
                # _, res,_,_= model(image)
            loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())

            res = res.sigmoid().data.cpu().numpy().squeeze()
            gt = 1 * (gt > 0.5)
            res = 1 * (res > 0.5)

            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)
            acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])
            JS = get_JS(res, gt)

            se = mean_sensitivity_np(gt, res)
            sp = get_specificity(res, gt)

            #score=(dice+iou)/2

            loss_bank.append(loss.item())
            dice_bank.append(dice)
            iou_bank.append(iou)
            acc_bank.append(acc)

            se_bank.append(se)
            sp_bank.append(sp)
            score_bank.append(dice+iou)

            JS_bank.append(JS)

        print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f},Se: {:.4f},Sp: {:.4f}, Acc: {:.4f},Js: {:.4f}'.
              format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank),np.mean(se_bank),np.mean(sp_bank),np.mean(acc_bank),np.mean(JS_bank)))

        #mean_loss.append(np.mean(dice_bank))
        mean_loss.append(np.mean(score_bank))
        mean_DC.append(np.mean(dice_bank))

    return mean_loss[0],mean_DC[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150, help='epoch number')
    parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
    parser.add_argument('--train_path', type=str,
                        default='data'+b+'/', help='path to train dataset')
    parser.add_argument('--test_path', type=str,
                        default='data'+b+'/', help='path to test dataset')
    parser.add_argument('--train_save', type=str, default='U-net')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")

    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')

    opt = parser.parse_args()
    # if not opt.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    #
    # random.seed(opt.seed)
    # np.random.seed(opt.seed)
    # torch.manual_seed(opt.seed)
    # torch.cuda.manual_seed(opt.seed)

    # ---- build models ----

    model = net().cuda()  # U_Net  TransFuse_S(pretrained=True).cuda()  DoubleViT   #lite_vision_transformer   DAEFormer
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2),weight_decay=opt.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)  # lr_3

    image_root = 'data'+b+'/data_train' + size + '.npy'
    gt_root = 'data'+b+'/mask_train' + size + '.npy'

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
    total_step = len(train_loader)

    print("#" * 20,size+year, "Start Training", "#" * 20)

    best_loss = 1e5
    for epoch in range(1, opt.epoch + 1):
        best_loss = train(train_loader, model, optimizer, epoch, best_loss)
