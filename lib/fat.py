# import torch
# from torchvision import models as resnet_model
# from torch import nn
# # from .DeiT import deit_small_patch16_224 as deit
#
# class FAMBlock(nn.Module):
#     def __init__(self, channels):
#         super(FAMBlock, self).__init__()
#
#         self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
#         self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)
#
#         self.relu3 = nn.ReLU(inplace=True)
#         self.relu1 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x3 = self.conv3(x)
#         x3 = self.relu3(x3)
#         x1 = self.conv1(x)
#         x1 = self.relu1(x1)
#         out = x3 + x1
#
#         return out
#
#
# class DecoderBottleneckLayer(nn.Module):
#     def __init__(self, in_channels, n_filters, use_transpose=True):
#         super(DecoderBottleneckLayer, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
#         self.norm1 = nn.BatchNorm2d(in_channels // 4)
#         self.relu1 = nn.ReLU(inplace=True)
#
#         if use_transpose:
#             self.up = nn.Sequential(
#                 nn.ConvTranspose2d(
#                     in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
#                 ),
#                 nn.BatchNorm2d(in_channels // 4),
#                 nn.ReLU(inplace=True)
#             )
#         else:
#             self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")
#
#         self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
#         self.norm3 = nn.BatchNorm2d(n_filters)
#         self.relu3 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.norm1(x)
#         x = self.relu1(x)
#         x = self.up(x)
#         x = self.conv3(x)
#         x = self.norm3(x)
#         x = self.relu3(x)
#         return x
#
#
# class SEBlock(nn.Module):
#     def __init__(self, channel, r=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // r, channel, bias=False),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         # Squeeze
#         y = self.avg_pool(x).view(b, c)
#         # Excitation
#         y = self.fc(y).view(b, c, 1, 1)
#         # Fusion
#         y = torch.mul(x, y)
#         return y
#
#
# class FAT_Net(nn.Module):
#     def __init__(self, n_channels=3, n_classes=1,pretrained=False):
#         super(FAT_Net, self).__init__()
#
#         transformer= torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
#         #transformer = torch.hub.load('G:\jinx\yolov5', 'custom', 'G:\jinx\yolov5\pt\yolo5s.pt', source='local')
#         resnet = resnet_model.resnet34(pretrained=True)
#         #transformer = torch.load('E:/fenge/TransFuse-1/facebookresearch/deitmain/deit_tiny_distilled_patch16_224-b40b3cf7.pth')
#         # self.resnet = resnet()
#         #
#         # self.resnet.load_state_dict(torch.load('pretrained/resnet34-333f7ec4.pth'))
#         #transformer=load_state_dict(torch.load('facebookresearch/deitmain/deit_tiny_distilled_patch16_224-b40b3cf7.pth'))
#         # transformer = deit(pretrained=pretrained)
#         # checkpoint = 'facebookresearch/deitmain/deit_tiny_distilled_patch16_224-b40b3cf7.pth'
#         # state_dict = torch.load(checkpoint)
#         # model = transformer(pretrained=False)
#         # model.load_state_dict(state_dict)
#
#
#         self.firstconv = resnet.conv1
#         self.firstbn = resnet.bn1
#         self.firstrelu = resnet.relu
#         self.encoder1 = resnet.layer1
#         self.encoder2 = resnet.layer2
#         self.encoder3 = resnet.layer3
#         self.encoder4 = resnet.layer4
#
#         self.patch_embed = transformer.patch_embed
#         self.transformers = nn.ModuleList(
#             [transformer.blocks[i] for i in range(12)]
#         )
#
#         self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
#         self.se = SEBlock(channel=1024)
#         self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)
#
#         self.FAMBlock1 = FAMBlock(channels=64)
#         self.FAMBlock2 = FAMBlock(channels=128)
#         self.FAMBlock3 = FAMBlock(channels=256)
#         self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
#         self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
#         self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])
#
#         filters = [64, 128, 256, 512]
#         self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
#         self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
#         self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
#         self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])
#
#         self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
#         self.final_relu1 = nn.ReLU(inplace=True)
#         self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
#         self.final_relu2 = nn.ReLU(inplace=True)
#         self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)
#
#
#     def forward(self, x):
#         b, c, h, w = x.shape
#
#         e0 = self.firstconv(x)
#         e0 = self.firstbn(e0)
#         e0 = self.firstrelu(e0)
#
#         e1 = self.encoder1(e0)
#         e2 = self.encoder2(e1)
#         e3 = self.encoder3(e2)
#         feature_cnn = self.encoder4(e3)
#
#         emb = self.patch_embed(x)
#         for i in range(12):
#             emb = self.transformers[i](emb)
#         feature_tf = emb.permute(0, 2, 1)
#         feature_tf = feature_tf.view(b, 192, 14, 14)
#         feature_tf = self.conv_seq_img(feature_tf)
#
#         feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
#         feature_att = self.se(feature_cat)
#         feature_out = self.conv2d(feature_att)
#
#         for i in range(2):
#             e3 = self.FAM3[i](e3)
#         for i in range(4):
#             e2 = self.FAM2[i](e2)
#         for i in range(6):
#             e1 = self.FAM1[i](e1)
#         d4 = self.decoder4(feature_out) + e3
#         d3 = self.decoder3(d4) + e2
#         d2 = self.decoder2(d3) + e1
#
#         out1 = self.final_conv1(d2)
#         out1 = self.final_relu1(out1)
#         out = self.final_conv2(out1)
#         out = self.final_relu2(out)
#         out = self.final_conv3(out)
#
#         return out
#
# # if __name__ == '__main__':
# #     input = torch.rand(2, 3, 224, 224)
# #     model = FAT_Net()
# #     out12 = model(input)
# #     print(out12.shape)
# import torch
# from torch.autograd import Variable
# import argparse
# from datetime import datetime
#
#
#
#
# from utils.dataloader import get_loader, test_dataset
# from utils.utils import AvgMeter
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from test_isic import mean_dice_np, mean_iou_np
# import os
# import random
# from torch.optim import lr_scheduler
# import torch.backends.cudnn as cudnn
#
# #net=lite_vision(),U_Net,AttU_Net，CE_Net，CPF_Net,FAT_Net，SwinUnet，UCTransNet
# net=FAT_Net
# #name=fuse,unet,attunet，canet，cenet,cpf，uctrans
# name="fat"
# year="2018"
#
# BCE = torch.nn.BCEWithLogitsLoss()
# def mean_sensitivity_np(y_true, y_pred, **kwargs):
#     """
#     compute mean iou for binary segmentation map via numpy
#     """
#     axes = (0, 1)
#     intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
#     mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
#     union = mask_sum - intersection
#     mask_GT=np.sum(np.abs(y_true), axis=axes)
#     smooth = .001
#     se = (intersection + smooth) / (mask_GT + smooth)
#     return se
#
# def get_specificity(SR, GT, threshold=0.5):
#     SR = torch.from_numpy(SR)
#     GT = torch.from_numpy(GT)
#     SR = SR > threshold
#     GT = GT == torch.max(GT)
#
#     # TN : True Negative
#     # FP : False Positive
#     TN = ((SR == 0) & (GT == 0))
#     FP = ((SR == 1) & (GT == 0))
#
#     SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
#
#     return SP
#
# def structure_loss(pred, mask):
#     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
#     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
#     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
#
#     pred = torch.sigmoid(pred)
#     inter = ((pred * mask) * weit).sum(dim=(2, 3))
#     union = ((pred + mask) * weit).sum(dim=(2, 3))
#     wiou = 1 - (inter + 1) / (union - inter + 1)
#     return (wbce + wiou).mean()
#
# # def structure_loss(pred, mask):
# #     weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
# #     wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
# #     wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
# #
# #     pred = torch.sigmoid(pred)
# #     inter = ((pred * mask) * weit).sum(dim=(2, 3))
# #     union = ((pred + mask) * weit).sum(dim=(2, 3))
# #     wiou = 1 - (inter + 1) / (union - inter + 1)
# #     return (wbce + wiou).mean()
#
# def get_JS(SR, GT, threshold=0.5):
#     # JS : Jaccard similarity
#     SR = torch.from_numpy(SR)
#     GT = torch.from_numpy(GT)
#     SR = SR > threshold
#     GT = GT == torch.max(GT)
#
#     Inter = torch.sum((SR.byte() + GT.byte()) == 2)
#     Union = torch.sum((SR.byte() + GT.byte()) >= 1)
#
#     JS = float(Inter) / (float(Union) + 1e-6)
#
#     return JS
# #
# # class focal_loss(preds, labels):
# #     def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
# #         super(focal_loss, self).__init__()
# #         self.size_average = size_average
# #         if isinstance(alpha, list):
# #             assert len(alpha) == num_classes
# #             self.alpha = torch.Tensor(alpha)
# #         else:
# #             assert alpha < 1
# #             self.alpha = torch.zeros(num_classes)
# #             self.alpha[0] += alpha
# #             self.alpha[1:] += (1 - alpha)
# #
# #         self.gamma = gamma
# #
# #     def forward(self, preds, labels):
# #         # assert preds.dim()==2 and labels.dim()==1
# #         preds = preds.view(-1, preds.size(-1))
# #         self.alpha = self.alpha.to(preds.device)
# #         preds_softmax = F.softmax(preds, dim=1)
# #         preds_logsoft = torch.log(preds_softmax)
# #
# #         # focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
# #         preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
# #         preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
# #         self.alpha = self.alpha.gather(0, labels.view(-1))
# #         # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
# #         loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
# #
# #         loss = torch.mul(self.alpha, loss.t())
# #         if self.size_average:
# #             loss = loss.mean()
# #         else:
# #             loss = loss.sum()
# #         return loss
# def train(train_loader, model, optimizer, epoch, best_loss):
#     model.train()
#     loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
#     accum = 0
#     for i, pack in enumerate(train_loader, start=1):
#         # ---- data prepare ----
#         images, gts = pack
#         images = Variable(images).cuda()
#         gts = Variable(gts).cuda()
#
#         # ---- forward ----
#         #out,x_v_d1,x_v_d2,x_v_d3,x_v_d4,x_v_d5,x_v_d6,x_v_d7,x_v_d8,x_v_d9= model(images)
#         out= model(images)# 分别是cnn和trans混合输入4，，3是trans分支，，2是最后融合输出
#         #out, x_v_d1, x_v_d2, x_v_d3 = model(images)
#         # x_v_d2,x_v_d3,out=model(images)
#
#         # ---- loss function ----
#         #loss2 = structure_loss(lateral_map_2, gts)  #
#         # aa=focal_loss(out, gts)
#         loss2 = structure_loss(out, gts)  #
#
#         #
#         #
#         loss = loss2
#         # loss4 = structure_loss(out, gts)
#         # loss3 = structure_loss(x_v_d2, gts)
#         # loss2 = structure_loss(x_v_d3, gts)
#
#
#         # loss =  loss3+ loss2 +  loss4
#
#         # ---- backward ----
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
#         optimizer.step()
#         optimizer.zero_grad()
#
#         # ---- recording loss ----
#         loss_record2.update(loss2.data, opt.batchsize)
#         loss_record3.update(loss2.data, opt.batchsize)
#         loss_record4.update(loss2.data, opt.batchsize)
#         # loss_record5.update(loss5.data, opt.batchsize)
#         # ---- train visualization ----
#         # if i % 20 == 0 or i == total_step:
#         #     print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
#         #           '[lateral-2: {:.4f}, lateral-3: {:0.4f}, lateral-4: {:0.4f}]'.
#         #           format(datetime.now(), epoch, opt.epoch, i, total_step,
#         #                  loss_record2.show(), loss_record3.show(), loss_record4.show()))
#         # ---- train visualization ----
#         if i % 20 == 0 or i == total_step:
#             print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
#                   '[lateral-2: {:.4f}, lateral-3: {:0.9f}, lateral-4: {:0.4f}]'.
#                   format(datetime.now(), epoch, opt.epoch, i, total_step,
#                          loss_record2.show(), optimizer.state_dict()['param_groups'][0]['lr'], loss_record4.show()))
#
#
#
#     save_path = 'snapshots/{}/'.format(opt.train_save)
#     os.makedirs(save_path, exist_ok=True)
#     # if (epoch + 1) % 1 == 0:
#     #     meanloss = tes(model, opt.test_path)
#     #     if meanloss > best_loss:
#     #         print('new best score:', meanloss)
#     #         best_loss = meanloss
#     #         torch.save(model.state_dict(), save_path + 'TransFuse-%d.pth' % epoch)
#     #         print('[Saving Snapshot:]', save_path + 'TransFuse-%d.pth' % epoch)
#     if epoch==1:
#         best_loss=0
#     if (epoch + 1) % 1 == 0:
#         meanloss = tes(model, opt.test_path)
#         if meanloss > best_loss:
#             print('new best score:', meanloss)
#             best_loss = meanloss
#             torch.save(model.state_dict(), save_path + year + name + '-%d.pth' % epoch)
#             print('[Saving Snapshot:]', save_path + year + name + '-%d.pth' % epoch)
#     return best_loss
#
#
# def tes(model, path):
#     model.eval()
#     mean_loss = []
#
#     for s in ['valid', 'test']:
#         image_root = '{}/data_{}224.npy'.format(path, s)#for s in ['val', 'test']:
#         gt_root = '{}/mask_{}224.npy'.format(path, s)
#         test_loader = test_dataset(image_root, gt_root)
#
#         dice_bank = []
#         iou_bank = []
#         loss_bank = []
#         acc_bank = []
#
#         se_bank = []
#         sp_bank = []
#         score_bank=[]
#         JS_bank = []
#
#         for i in range(test_loader.size):
#             image, gt = test_loader.load_data()
#             image = image.cuda()
#
#             with torch.no_grad():
#                 res= model(image)
#                 # res, _, _, _, _, _, _, _, _, _ = model(image)
#                 # _, res,_,_= model(image)
#             loss = structure_loss(res, torch.tensor(gt).unsqueeze(0).unsqueeze(0).cuda())
#
#             res = res.sigmoid().data.cpu().numpy().squeeze()
#             gt = 1 * (gt > 0.5)
#             res = 1 * (res > 0.5)
#
#             dice = mean_dice_np(gt, res)
#             iou = mean_iou_np(gt, res)
#             acc = np.sum(res == gt) / (res.shape[0] * res.shape[1])
#             JS = get_JS(res, gt)
#
#             se = mean_sensitivity_np(gt, res)
#             sp = get_specificity(res, gt)
#
#             #score=(dice+iou)/2
#
#             loss_bank.append(loss.item())
#             dice_bank.append(dice)
#             iou_bank.append(iou)
#             acc_bank.append(acc)
#
#             se_bank.append(se)
#             sp_bank.append(sp)
#             score_bank.append(dice+iou)
#
#             JS_bank.append(JS)
#
#         print('{} Loss: {:.4f}, Dice: {:.4f}, IoU: {:.4f},Se: {:.4f},Sp: {:.4f}, Acc: {:.4f},Js: {:.4f}'.
#               format(s, np.mean(loss_bank), np.mean(dice_bank), np.mean(iou_bank),np.mean(se_bank),np.mean(sp_bank),np.mean(acc_bank),np.mean(JS_bank)))
#
#         #mean_loss.append(np.mean(dice_bank))
#         mean_loss.append(np.mean(score_bank))
#
#     return mean_loss[0]
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epoch', type=int, default=150, help='epoch number')
#     parser.add_argument('--lr', type=float, default=7e-5, help='learning rate')
#     parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
#     parser.add_argument('--grad_norm', type=float, default=2.0, help='gradient clipping norm')
#     parser.add_argument('--train_path', type=str,
#                         default='data/', help='path to train dataset')
#     parser.add_argument('--test_path', type=str,
#                         default='data/', help='path to test dataset')
#     parser.add_argument('--train_save', type=str, default='U-net')
#     parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
#     parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')
#     parser.add_argument("--seed", type=int, default=1234, help="random seed")
#     parser.add_argument("--deterministic", type=int, default=1, help="whether to use deterministic training")
#
#     opt = parser.parse_args()
#     # if not opt.deterministic:
#     #     cudnn.benchmark = True
#     #     cudnn.deterministic = False
#     # else:
#     #     cudnn.benchmark = False
#     #     cudnn.deterministic = True
#     #
#     # random.seed(opt.seed)
#     # np.random.seed(opt.seed)
#     # torch.manual_seed(opt.seed)
#     # torch.cuda.manual_seed(opt.seed)
#
#     # ---- build models ----
#
#     model = net().cuda()  # U_Net  TransFuse_S(pretrained=True).cuda()  DoubleViT   #lite_vision_transformer   DAEFormer
#     params = model.parameters()
#     optimizer = torch.optim.Adam(params, opt.lr, betas=(opt.beta1, opt.beta2))
#     scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)  # lr_3
#
#     image_root = '{}/data_train224.npy'.format(opt.train_path)
#     gt_root = '{}/mask_train224.npy'.format(opt.train_path)
#
#     train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize)
#     total_step = len(train_loader)
#
#     print("#" * 20, "Start Training", "#" * 20)
#
#     best_loss = 1e5
#     for epoch in range(1, opt.epoch + 1):
#         best_loss = train(train_loader, model, optimizer, epoch, best_loss)
import torch
from torchvision import models as resnet_model
from torch import nn


class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out


class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y


class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(FAT_Net, self).__init__()

        transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.patch_embed = transformer.patch_embed
        self.transformers = nn.ModuleList(
            [transformer.blocks[i] for i in range(12)]
        )

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


    def forward(self, x):
        b, c, h, w = x.shape

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        emb = self.patch_embed(x)
        for i in range(12):
            emb = self.transformers[i](emb)
        feature_tf = emb.permute(0, 2, 1)
        feature_tf = feature_tf.view(b, 192, 14, 14)
        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out