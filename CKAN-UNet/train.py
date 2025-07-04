import os
import torch
import math
import torch.utils.data as Data
import argparse
import numpy as np
import sys
from tqdm import tqdm
import random
from ptflops import get_model_complexity_info
from distutils.version import LooseVersion
from Datasets.ISIC2018 import ISIC2018_dataset
from utils.transform import ISIC2018_transform, ISIC2018_transform_320, ISIC2018_transform_newdata
from Models.CKAN_UNet import CKANUNet
from utils.dice_loss import get_soft_label, val_dice_isic
from utils.dice_loss import Intersection_over_Union_isic
from utils.dice_loss_github import SoftDiceLoss_git, CrossentropyND,IoULoss
from utils.KDloss import KDloss
from utils.evaluation import AverageMeter
from utils.binary import assd, dc, jc, precision, sensitivity, specificity, F1, ACC
from torch.optim import lr_scheduler
from time import *
from PIL import Image
import subprocess

Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'A': ISIC2018_transform, 'B': ISIC2018_transform_320, "C": ISIC2018_transform_newdata}


def get_gpu_memory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], stdout=subprocess.PIPE)
    memory_info = result.stdout.decode('utf-8').strip().split(',')
    memory_used = int(memory_info[0])
    memory_total = int(memory_info[1])
    return memory_used, memory_total


def losscd(out_c, target, W, num_classes=2):
    soft_dice_loss2 = SoftDiceLoss_git(batch_dice=False, dc_log=False)
    CE_loss_F = CrossentropyND()
    iou_loss = IoULoss(batch_dice=False)

    target_soft_a = get_soft_label(target, num_classes)
    target_soft = target_soft_a.permute(0, 3, 1, 2)

    dice_loss_f = soft_dice_loss2(out_c, target_soft)
    ce_loss_f = CE_loss_F(out_c, target)
    iou_loss_f = iou_loss(out_c, target)

    loss_f = W[0]*dice_loss_f + W[1]*ce_loss_f + W[2]*iou_loss_f
    return loss_f


def losskd(encoder, decoder, final, lambda_x):
    Loss = KDloss(lambda_x=lambda_x)
    loss = Loss(encoder, decoder, final)
    return loss


class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def train(train_loader, model, scheduler, optimizer, args, epoch):
    losses = AverageMeter()
    flag_GPU = True

    model.train()
    for step, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = x.float().cuda()
        target = y.float().cuda()

        logits, encoder, decoder, final = model(image)

        # ---- loss function ----
        lamda = [0.2, 0.2 , 10, 0.1]
        loss_cd = losscd(logits, target, W=lamda )
        loss_kd = losskd(encoder, decoder, final, lamda[3])
        loss = loss_cd + loss_kd
        # loss = loss_cd
        losses.update(loss.data, image.size(0))

        del encoder, decoder, final
        torch.cuda.empty_cache()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        optimizer.step()

        if step % (math.ceil(float(len(train_loader.dataset)) / args.batch_size)) == 0:
            print('current lr: {} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                optimizer.state_dict()['param_groups'][0]['lr'],
                epoch, step * len(image), len(train_loader.dataset),
                       100. * step / len(train_loader), losses=losses))

    print('The average loss:{losses.avg:.4f}'.format(losses=losses))


def valid(valid_loader, model, optimizer, args, epoch, best_score, best_score1):
    isic_Jaccard = []
    isic_dc = []
    isic_iou = []

    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader), total=len(valid_loader), mininterval=0.001):
        image = t.float().cuda()
        target = k.float().cuda()

        out_f = model(image, False)  # model output
        if isinstance(out_f, list) or isinstance(out_f, tuple):
            output = out_f[-1]
        else:
            output = out_f

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()

        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_iou = Intersection_over_Union_isic(output_dis_test, target_test, 1)
        iou_np = isic_b_iou.data.cpu().numpy()

        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)
        isic_iou.append(iou_np)

    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_dc_mean = np.average(isic_dc)
    isic_iou_mean = np.average(isic_iou)

    net_score = isic_Jaccard_mean + isic_dc_mean
    net_score1 = isic_iou_mean + isic_dc_mean

    print('The ISIC Dice score: {dice: .4f};The ISIC IOU score: {jc: .4f}'.format(
        dice=isic_dc_mean, jc=isic_iou_mean))

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if net_score > max(best_score):
        best_score.append(net_score)
        modelname = args.ckpt + '/' + 'best_score_dj' + '_' + args.data + '_checkpoint.pth.tar'
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    if net_score1 > max(best_score1):
        best_score1.append(net_score1)
        print(best_score1)
        modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)


def test(test_loader, model, args, date_type, save_img=True):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    isic_dice = []
    isic_iou = []
    # isic_assd = []
    isic_acc = []
    isic_sensitive = []
    isic_specificy = []
    isic_precision = []
    isic_f1_score = []
    isic_Jaccard_M = []
    isic_Jaccard_N = []
    isic_Jaccard = []
    isic_dc = []
    infer_time = []

    print(
        "******************************************************************** {} || start **********************************".format(
            date_type) + "\n")

    modelname = args.ckpt + '/' + 'best_score' + '_' + date_type + '_checkpoint.pth.tar'
    img_saved_dir_root = os.path.join(args.ckpt, "segmentation_result")
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    flag_GPU = True

    model.eval()
    for step, (name, img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        with torch.no_grad():
            image = img.float().cuda()
            target = lab.float().cuda()

            begin_time = time()
            out_f = model(image, False)

            if flag_GPU:
                memory_used, memory_total = get_gpu_memory()
                flag_GPU = False

            if isinstance(out_f, list) or isinstance(out_f, tuple):
                output = out_f[-1]
            else:
                output = out_f

            end_time = time()
            pred_time = end_time - begin_time
            infer_time.append(pred_time)

            output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)

            """
            save segmentation result
            """
            if save_img:
                if date_type == "ISIC2018" and args.val_folder == "folder1":
                    npy_path = os.path.join(args.root_path, 'image', name[0])
                    img = np.load(npy_path)
                    im = Image.fromarray(np.uint8(img))
                    im_path = name[0].split(".")[0] + "_img" + ".png"
                    img_saved_dir = os.path.join(img_saved_dir_root, name[0].split(".")[0])
                    if not os.path.isdir(img_saved_dir):
                        os.makedirs(img_saved_dir)
                    im.save(os.path.join(img_saved_dir, im_path))

                    target_np = target.squeeze().cpu().numpy()
                    label = Image.fromarray(np.uint8(target_np * 255))
                    label_path = name[0].split(".")[0] + "_label" + ".png"
                    label.save(os.path.join(img_saved_dir, label_path))

                    seg_np = output_dis.squeeze().cpu().numpy()
                    seg = Image.fromarray(np.uint8(seg_np * 255))
                    seg_path = name[0].split(".")[0] + "_seg" + ".png"
                    seg.save(os.path.join(img_saved_dir, seg_path))
                else:
                    pass
            else:
                pass

            output_dis_test = output_dis.permute(0, 2, 3, 1).float()
            target_test = target.permute(0, 2, 3, 1).float()
            output_soft = get_soft_label(output_dis, 2)
            target_soft = get_soft_label(target, 2)

            label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
            output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

            isic_b_dice = val_dice_isic(output_soft, target_soft, 2)  # the dice
            isic_b_iou = Intersection_over_Union_isic(output_dis_test, target_test, 1)  # the iou
            # isic_b_asd = assd(output_arr[:, :, 1], label_arr[:, :, 1])                                     # the assd
            isic_b_acc = ACC(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the accuracy
            isic_b_sensitive = sensitivity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the sensitivity
            isic_b_specificy = specificity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the specificity
            isic_b_precision = precision(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the precision
            isic_b_f1_score = F1(output_dis_test.cpu().numpy(), target_test.cpu().numpy())  # the F1
            isic_b_Jaccard_m = jc(output_arr[:, :, 1], label_arr[:, :, 1])  # the Jaccard melanoma
            isic_b_Jaccard_n = jc(output_arr[:, :, 0], label_arr[:, :, 0])  # the Jaccard no-melanoma
            isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
            isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())

            dice_np = isic_b_dice.data.cpu().numpy()
            iou_np = isic_b_iou.data.cpu().numpy()

            isic_dice.append(dice_np)
            isic_iou.append(iou_np)
            # isic_assd.append(isic_b_asd)
            isic_acc.append(isic_b_acc)
            isic_sensitive.append(isic_b_sensitive)
            isic_specificy.append(isic_b_specificy)
            isic_precision.append(isic_b_precision)
            isic_f1_score.append(isic_b_f1_score)
            isic_Jaccard_M.append(isic_b_Jaccard_m)
            isic_Jaccard_N.append(isic_b_Jaccard_n)
            isic_Jaccard.append(isic_b_Jaccard)
            isic_dc.append(isic_b_dc)

    all_time = np.sum(infer_time)
    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

    # isic_assd_mean = np.average(isic_assd)
    # isic_assd_std = np.std(isic_assd)

    isic_acc_mean = np.average(isic_acc)
    isic_acc_std = np.std(isic_acc)

    isic_sensitive_mean = np.average(isic_sensitive)
    isic_sensitive_std = np.std(isic_sensitive)

    isic_specificy_mean = np.average(isic_specificy)
    isic_specificy_std = np.std(isic_specificy)

    isic_precision_mean = np.average(isic_precision)
    isic_precision_std = np.std(isic_precision)

    isic_f1_score_mean = np.average(isic_f1_score)
    iisic_f1_score_std = np.std(isic_f1_score)

    isic_Jaccard_M_mean = np.average(isic_Jaccard_M)
    isic_Jaccard_M_std = np.std(isic_Jaccard_M)

    isic_Jaccard_N_mean = np.average(isic_Jaccard_N)
    isic_Jaccard_N_std = np.std(isic_Jaccard_N)

    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_Jaccard_std = np.std(isic_Jaccard)

    isic_dc_mean = np.average(isic_dc)
    isic_dc_std = np.std(isic_dc)

    print(f"GPU Memory Used: {memory_used} MB")
    print(f"GPU Memory Total: {memory_total} MB")
    print('The mean dice: {isic_dice_mean: .4f}; The dice std: {isic_dice_std: .4f}'.format(
        isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std))
    print('The mean IoU: {isic_iou_mean: .4f}; The IoU std: {isic_iou_std: .4f}'.format(
        isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))
    print('The mean ACC: {isic_acc_mean: .4f}; The ACC std: {isic_acc_std: .4f}'.format(
        isic_acc_mean=isic_acc_mean, isic_acc_std=isic_acc_std))
    print('The mean sensitive: {isic_sensitive_mean: .4f}; The sensitive std: {isic_sensitive_std: .4f}'.format(
        isic_sensitive_mean=isic_sensitive_mean, isic_sensitive_std=isic_sensitive_std))
    print('The mean specificy: {isic_specificy_mean: .4f}; The specificy std: {isic_specificy_std: .4f}'.format(
        isic_specificy_mean=isic_specificy_mean, isic_specificy_std=isic_specificy_std))
    print('The mean precision: {isic_precision_mean: .4f}; The precision std: {isic_precision_std: .4f}'.format(
        isic_precision_mean=isic_precision_mean, isic_precision_std=isic_precision_std))
    print('The mean f1_score: {isic_f1_score_mean: .4f}; The f1_score std: {iisic_f1_score_std: .4f}'.format(
        isic_f1_score_mean=isic_f1_score_mean, iisic_f1_score_std=iisic_f1_score_std))
    print('The mean Jaccard_M: {isic_Jaccard_M_mean: .4f}; The Jaccard_M std: {isic_Jaccard_M_std: .4f}'.format(
        isic_Jaccard_M_mean=isic_Jaccard_M_mean, isic_Jaccard_M_std=isic_Jaccard_M_std))
    print('The mean Jaccard_N: {isic_Jaccard_N_mean: .4f}; The Jaccard_N std: {isic_Jaccard_N_std: .4f}'.format(
        isic_Jaccard_N_mean=isic_Jaccard_N_mean, isic_Jaccard_N_std=isic_Jaccard_N_std))
    print('The mean Jaccard: {isic_Jaccard_mean: .4f}; The Jaccard std: {isic_Jaccard_std: .4f}'.format(
        isic_Jaccard_mean=isic_Jaccard_mean, isic_Jaccard_std=isic_Jaccard_std))
    print('The mean dc: {isic_dc_mean: .4f}; The dc std: {isic_dc_std: .4f}'.format(
        isic_dc_mean=isic_dc_mean, isic_dc_std=isic_dc_std))
    print('The inference time: {time: .4f}'.format(time=all_time))

    print(
        "******************************************************************** {} || end **********************************".format(
            date_type) + "\n")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    best_score = [0]
    best_score1 = [0]
    start_epoch = args.start_epoch
    print('loading the {0},{1},{2} dataset ...'.format('train', 'test', 'test'))

    if args.data == 'ISIC2018':
        trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train',
                                           with_name=False, transform=Test_Transform[args.transform])
        validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                           with_name=False, transform=Test_Transform[args.transform])
        testset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                          with_name=True, transform=Test_Transform[args.transform])

        trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                      num_workers=6)
        validloader = Data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
        testloader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)

    model = CKANUNet(num_classes=2, channels=3)
    model = model.cuda()

    print(
        "------------------------------------------------------------------------------------------------------------")
    flops, params = get_model_complexity_info(model, (3, args.out_size[0], args.out_size[1]),
                                              as_strings=True, print_per_layer_stat=False)
    print("ptflops test result: ")
    print("Flops: {}".format(flops))
    print("Params: " + params)
    print(
        "------------------------------------------------------------------------------------------------------------")
    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
    #
    print("Start training .................................................................................")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        scheduler.step()
        train(trainloader, model, scheduler, optimizer, args, epoch)
        valid(validloader, model, optimizer, args, epoch, best_score, best_score1)
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)
    print('Training Done! Start testing....................................................................')

    test(testloader, model, args, args.data, save_img=args.save_img)
    print('Testing Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Path related arguments
    parser.add_argument('--root_path', default="/home/yuanjun/ISIC/processed_ISIC2018",help='Data storage path')
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    parser.add_argument('--ckpt', default='/home/yuanjun/ISIC/result/result_ISIC2018_7_4',help='Save result path')

    parser.add_argument('--transform', default='C')
    parser.add_argument('--out_size', default=(224, 224), help='the output image size')
    parser.add_argument('--val_folder', default='folder3', type=str, help='folder1、folder2、folder3、folder4、folder5')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--save_img', type=str, default=True, help='whether save segmentation result')

    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=1800, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--lr_rate', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--num_classes', default=2, type=int, help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=2000, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='', help='the checkpoint that resumes from')

    args = parser.parse_args()

    main(args)
