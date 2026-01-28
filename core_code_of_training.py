import argparse
import logging
import os
import random
import shutil
import sys
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_dilation
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import RandomCrop, TwoStreamBatchSampler, \
    PancreasOverlapDis, RandomCropOverlap96Dis, \
    ToTensorOverlapDis, RandomRotFlipDis, LAHeartPreDis, ToTensorPreDis, BratsPreDis
from networks.net_factory import net_factory, update_ema_variables
from utils import ramps, losses, test_patch


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


class NegEntropy(object):
    def __call__(self, outputs_list):
        entropy_results = torch.tensor([.0]).cuda()
        for i in outputs_list:
            current_prob = torch.softmax(i, dim=0).clamp(min=1e-6, max=1.)
            entropy_results += torch.sum(current_prob.log() * current_prob, dim=0).mean()
        return entropy_results / len(outputs_list)


def rand_bbox(size, lam=None):
    # past implementation
    W = size[2]
    H = size[3]
    B = size[0]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)
    cx = np.random.randint(size=[B, ], low=int(W / 8), high=W)
    cy = np.random.randint(size=[B, ], low=int(H / 8), high=H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def cut_mix(volume=None, mask=None, gt=None):
    mix_label1 = torch.zeros_like(gt).cuda()
    mix_label1[0] = gt[0]
    mix_label1[1] = mask[0][1]

    mix_label2 = torch.zeros_like(gt).cuda()
    mix_label2[0] = gt[0]
    mix_label2[1] = mask[1][1]

    mix_data = torch.zeros_like(gt).cuda().unsqueeze(1).float()
    mix_data[0] = volume[0]
    mix_data[1] = volume[3]

    u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(gt.size(), lam=np.random.beta(4, 4))

    i = 0
    mix_data[0, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        volume[2, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    mix_label1[0, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        mask[0][0, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    mix_label2[0, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        mask[1][0, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    i = 1
    mix_data[1, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        volume[1, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    mix_label1[1, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        gt[1, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]
    mix_label2[1, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
        gt[1, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

    return mix_data, [mix_label1, mix_label2]

def cut_mix_mask_box(volume=None, mask=None, pse=None, gt=None):
    mix_label1 = torch.zeros_like(mask).cuda()
    mix_label2 = torch.zeros_like(mask).cuda()
    mix_data = torch.zeros_like(mask).cuda().float()
    mask_box = torch.zeros_like(mask).cuda() # [bs, 1, x, y, z]
    gt_ = gt.unsqueeze(1).float()
    pse_1 = pse[0].unsqueeze(1).float()
    pse_2 = pse[1].unsqueeze(1).float()

    for i in range(len(gt_)):
        lam = np.random.beta(4, 4)
        mask_np = mask[i, 0].cpu().numpy() # [x, y, z]
        coords = np.argwhere(mask_np > 0)

        if coords.size == 0:
            return None, None

        center = np.mean(coords, axis=0).astype(int)
        H, W, _ = volume.shape[2:]
        box_H, box_W = int(lam * H), int(lam * W)

        h_start = max(0, center[0] - box_H // 2)
        h_end = min(H, center[0] + box_H // 2)

        w_start = max(0, center[1] - box_W // 2)
        w_end = min(W, center[1] + box_W // 2)

        box_mask = torch.zeros_like(volume[i, 0]).bool().cuda()
        box_mask[h_start:h_end, w_start:w_end, :] = True
        box_mask = box_mask.unsqueeze(0).float()
        mask_box[i] = box_mask

    mix_data[0] = (1-mask_box[0]) * volume[0] + mask_box[0] * volume[2]
    mix_label1[0] = (1-mask_box[0]) * gt_[0] + mask_box[0] * pse_1[0]
    mix_label2[0] = (1 - mask_box[0]) * gt_[0] + mask_box[0] * pse_2[0]

    mix_data[1] = (1-mask_box[1]) * volume[3] + mask_box[1] * volume[1]
    mix_label1[1] = (1-mask_box[1]) * pse_1[1] + mask_box[1] * gt_[1]
    mix_label2[1] = (1 - mask_box[1]) * pse_2[1] + mask_box[1] * gt_[1]

    return mix_data, [mix_label1.squeeze(1).long(), mix_label2.squeeze(1).long()]


def create_model(model, ema=False):
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BraTS2019', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='TCSeg', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=20000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=250, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=25, help='trained samples')
parser.add_argument('--seed', type=int, default=2222, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--min_t', type=float, default=0.1, help='thresh to get reliable negative samples')
parser.add_argument('--max_t', type=float, default=0.85, help='thresh to get reliable positive samples')
parser.add_argument('--p_t', type=float, default=0.05, help='consistency thresh for unlabeled data')

args = parser.parse_args()
args.exp = args.exp + '_' + str(args.min_t) + '_' + str(args.max_t) + '_' + str(args.p_t) + '_' + str(args.seed)
snapshot_path = args.root_path + "checkpoints/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path + 'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/Pancreas/'
    args.max_samples = 62
elif args.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path + 'data/BraTS2019/'
    args.max_samples = 250

train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


def semi_loss(inputs, targets,
              threshold=0.6,
              neg_threshold=0.3,
              outputs_seg_two=None,
              mask_t=None,
              ema_mask_t=None,
              p_t=0.05,
              n_out=None,
              conf_mask=True):

    if not conf_mask:
        raise NotImplementedError

    seg1_prob = F.softmax(outputs_seg_two[0], dim=1)[labeled_bs:, 1]  # shape: B, ...
    seg2_prob = F.softmax(outputs_seg_two[1], dim=1)[labeled_bs:, 1]
    consistency_mask = ((torch.abs(seg1_prob - seg2_prob) <= p_t)
                        & (torch.abs(mask_t[0][labeled_bs:, 1] - mask_t[1][labeled_bs:, 1]) <= 2 * p_t)
                        & (torch.abs(ema_mask_t[0][:, 1] - ema_mask_t[1][:, 1]) <= 2 * p_t)
                        & (torch.abs(mask_t[n_out][labeled_bs:, 1] - F.softmax(outputs_seg_two[n_out], dim=1)[labeled_bs:, 1]) <= 2 * p_t))

    targets_prob = F.softmax(targets, dim=1)
    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = (pos_weight >= threshold) & consistency_mask
    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = (neg_weight < neg_threshold) & consistency_mask
    y_tilde = torch.argmax(targets, dim=1)
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(torch.log_softmax(inputs, dim=1),
                                       y_tilde, reduction="none")
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]

    if not torch.any(neg_mask):
        negative_loss_mat = torch.tensor([.0], device=targets.device)
    else:
        inverse_prob = torch.clamp(1 - F.softmax(inputs, dim=1), min=1e-6, max=1.0)
        negative_loss_mat = F.nll_loss(inverse_prob.log(), (1 - y_tilde), reduction="none")
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]

    return positive_loss_mat.mean() + negative_loss_mat.mean(), None


def get_proto(pred_soft, outfeats, conf_mask=None):
    index = torch.argmax(pred_soft, dim=1)  # [B,W,H,D]
    one_hot = F.one_hot(index, num_classes).permute(0, 4, 1, 2, 3).float()  # [B,N,W,H,D]
    if conf_mask is not None:
        one_hot = one_hot * conf_mask.unsqueeze(1)  # [B,1,W,H,D]
    masked_feats = outfeats.unsqueeze(1) * one_hot.unsqueeze(2)  # [B,N, C, W,H,D]
    proto_sum = masked_feats.sum(dim=[3, 4, 5])  # [B,N,C]
    count = one_hot.sum(dim=[2, 3, 4]).unsqueeze(-1) + 1e-6  # [B,N,1]
    prototype_bank = proto_sum / count  # [B,N,C]
    prototype_bank = F.normalize(prototype_bank, dim=-1)  # normalize
    # teacher_f: [B, C, W,H,D]
    # prototype_bank: [B,N,C] → reshape for broadcast
    proto = prototype_bank.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,N,C,1,1,1]
    return proto


def get_mask_from_proto(outputs_seg_two, n_out_, outfeats_two):
    conf_mask2 = ((F.softmax(outputs_seg_two[n_out_], dim=1)[:, 1] >= args.max_t) | (
                F.softmax(outputs_seg_two[n_out_], dim=1)[:, 1] <= args.min_t)).float()  # [B,W,H,D]
    proto2 = get_proto(F.softmax(outputs_seg_two[n_out_], dim=1), outfeats_two[n_out_], conf_mask2)  # [B,N,C,1,1,1]
    mask_t2 = F.cosine_similarity(outfeats_two[n_out_].unsqueeze(1), proto2, dim=2)  # [B,N,W,H,D]
    return mask_t2, proto2


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(sys.argv[0])

    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    model_ema = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")

    model = create_model(model)
    model_ema = create_model(model_ema, ema=True)
    if args.dataset_name == "LA":
        db_train = LAHeartPreDis(base_dir=train_data_path,
                                 split='train',
                                 transform=transforms.Compose([
                                     RandomRotFlipDis(),
                                     RandomCrop(patch_size),
                                     ToTensorPreDis(),
                                 ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = PancreasOverlapDis(base_dir=train_data_path,
                               split='train',
                               transform=transforms.Compose([
                                   RandomCropOverlap96Dis(patch_size),
                                   ToTensorOverlapDis(),
                               ]))
    elif args.dataset_name == "BraTS2019":
        db_train = BratsPreDis(base_dir=train_data_path,
                               split='train',
                               transform=transforms.Compose([
                                   RandomRotFlipDis(),
                                   RandomCrop(patch_size),
                                   ToTensorPreDis(),
                               ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train,
                             batch_sampler=batch_sampler,
                             num_workers=4,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    logging.info("{} itertations per epoch".format(len(trainloader)))
    pixel_criterion = losses.ce_loss_mask
    consistency_criterion = nn.CrossEntropyLoss(reduction='none')
    dice_loss = losses.Binary_dice_loss
    ce_loss = CrossEntropyLoss()
    dis_loss = nn.MSELoss()
    uncert_crit = nn.MSELoss(reduction='none')
    pse_dis_loss = nn.MSELoss(reduction='none')
    saddle_reg = NegEntropy()


    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
    save_last_path = os.path.join(snapshot_path, '{}_last_model.pth'.format(args.model))

    for epoch_num in iterator:
        for batch_idx, normal_batch in enumerate(trainloader):
            if args.dataset_name == "Pancreas_CT":
                normal_batch = normal_batch[0]
                volume_batch, label_batch, dis_batch = normal_batch['data'], normal_batch['label'], normal_batch['dis']
            else:
                volume_batch, label_batch, dis_batch = normal_batch['image'], normal_batch['label'], normal_batch['dis']
            iter_num += 1

            volume_batch, label_batch, dis_batch = volume_batch.cuda(), label_batch.cuda(), dis_batch.cuda()
            gt_ratio = torch.sum(label_batch == 0) / len(torch.ravel(label_batch))
            dis_batch = dis_batch.unsqueeze(1)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            model.train()
            loss = 0
            outputs_seg_two, _, outfeats_two = model(volume_batch)
            num_outputs = len(outputs_seg_two)

            if iter_num > 0:
                seg_list = []
                ema_seg_output = []
                for o in range(num_outputs):
                    seg_list.append(torch.softmax(outputs_seg_two[o], dim=1)[labeled_bs:, 1, ...])
                with torch.no_grad():
                    ema_seg_output_, _, ema_outfeats_two = model_ema(volume_batch[labeled_bs:])
                    for n_ in range(num_outputs):
                        seg_list.append(torch.softmax(ema_seg_output_[n_], dim=1)[:, 1, ...])
                        ema_seg_output.append(torch.argmax(torch.softmax(ema_seg_output_[n_], dim=1), dim=1, keepdim=True).squeeze(1))

            for n_out in range(num_outputs):
                for n_out_ in range(num_outputs):
                    if n_out == n_out_: continue

                    outputs = outputs_seg_two[n_out]

                    """ supervised part """
                    sup_ce_loss = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs]).mean()
                    outputs_soft = F.softmax(outputs, dim=1)
                    sup_dice_loss = dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1).mean()
                    loss_sup = (sup_ce_loss + sup_dice_loss) / 4.0

                    if n_out == 0 and n_out_ == 1:
                        spatial_loss = F.mse_loss(F.softmax(outputs_seg_two[0], dim=1), F.softmax(outputs_seg_two[1], dim=1), reduction='mean')

                        mask_t1, proto1 = get_mask_from_proto(outputs_seg_two, 0, outfeats_two)
                        mask_t2, proto2 = get_mask_from_proto(outputs_seg_two, 1, outfeats_two)
                        spatial_loss += F.mse_loss(mask_t1, mask_t2, reduction='mean') / 4.0

                        with torch.no_grad():
                            ema_mask_t1, ema_proto1 = get_mask_from_proto(ema_seg_output_, n_out, ema_outfeats_two)
                            ema_mask_t2, ema_proto2 = get_mask_from_proto(ema_seg_output_, n_out_, ema_outfeats_two)

                        spatial_loss += F.mse_loss(mask_t1, outputs_soft, reduction='mean') / 4.0
                    else:
                        spatial_loss += F.mse_loss(mask_t2, F.softmax(outputs_seg_two[1], dim=1), reduction='mean') / 4.0

                    """ unsupervised part """
                    # pseudo-label
                    pseudo_label = outputs_seg_two[n_out_].detach()
                    loss_crc, _ = semi_loss(inputs=outputs[labeled_bs:],
                                            targets=pseudo_label[labeled_bs:],
                                            threshold=args.max_t,
                                            neg_threshold=args.min_t,
                                            outputs_seg_two=outputs_seg_two,
                                            mask_t=[mask_t1, mask_t2],
                                            ema_mask_t=[ema_mask_t1, ema_mask_t2],
                                            p_t=args.p_t,
                                            n_out=n_out_,
                                            conf_mask=True)

                    # calculates the translated loss for loss1
                    loss_trans = 1.0 * spatial_loss

                    # the overall semi-supervised loss
                    loss_unsup = loss_crc + loss_trans
                    loss += loss_sup + loss_unsup * consistency_weight

            if iter_num > 100:
                mix_loss = 0
                seg_fuse = torch.stack(seg_list, dim=0)  # [4, bs, x, y, z]
                seg_fuse = seg_fuse.detach()
                seg_fuse_avg = torch.mean(seg_fuse, dim=0)  # [bs, x, y, z]
                seg_fuse_avg_bin = torch.where(
                    (seg_fuse_avg >= args.min_t) & (seg_fuse_avg <= args.max_t),
                    torch.tensor(1, device=seg_fuse_avg.device, dtype=seg_fuse_avg.dtype),
                    torch.tensor(0, device=seg_fuse_avg.device, dtype=seg_fuse_avg.dtype)
                )

                seg_diff1 = torch.abs(seg_fuse[0] - seg_fuse[1])
                seg_diff2 = torch.abs(seg_fuse[2] - seg_fuse[3])
                seg_diff3 = torch.abs(mask_t1[labeled_bs:, 1] - mask_t2[labeled_bs:, 1])
                seg_diff4 = torch.abs(ema_mask_t1[:, 1] - ema_mask_t2[:, 1])
                diff_mask = (seg_diff1 > args.p_t) | (seg_diff2 > args.p_t) | (seg_diff3 > 2*args.p_t) | (seg_diff4 > 2*args.p_t)

                high_conf_mask = 1 - seg_fuse_avg_bin
                branch_diff_mask = diff_mask.float() * high_conf_mask  # [bs, x, y, z]
                mask_all = branch_diff_mask + seg_fuse_avg_bin
                mask_all = mask_all >= 1.0
                mask_all = mask_all.cpu().numpy()

                closed_masks = []

                for i in range(mask_all.shape[0]):
                    closed_3d = binary_closing(mask_all[i], structure=np.ones((3, 3, 3)))
                    size = random.randint(1, 7)
                    random_structure = np.random.rand(size, size, size) < 0.5
                    dilated_mask_np = binary_dilation(closed_3d, structure=random_structure)
                    if np.sum(dilated_mask_np) < 100:
                        closed_masks.append(np.zeros_like(dilated_mask_np))
                    else:
                        labeled_mask_dilated, num_features = ndimage.label(dilated_mask_np)
                        volumes = ndimage.sum(dilated_mask_np, labeled_mask_dilated, range(1, num_features + 1))
                        largest_component_label = np.argmax(volumes) + 1
                        largest_component_mask = (labeled_mask_dilated == largest_component_label)
                        closed_masks.append(largest_component_mask)
                processed_mask_np = np.stack(closed_masks, axis=0)
                processed_mask_np = torch.from_numpy(np.expand_dims(processed_mask_np, axis=1)).cuda()

                if iter_num % 2 == 0:
                    cut_mix_data, cut_mix_label = cut_mix(volume_batch, ema_seg_output, label_batch[:labeled_bs])
                else:
                    cut_mix_data, cut_mix_label = cut_mix_mask_box(volume_batch, processed_mask_np.float(), ema_seg_output, label_batch[:labeled_bs])
                    if cut_mix_data is None:
                        logging.info('cut mix data is None, regenerate.')
                        cut_mix_data, cut_mix_label = cut_mix(volume_batch, ema_seg_output, label_batch[:labeled_bs])

                outputs_seg_mix, outputs_reg_mix, _ = model(cut_mix_data.float())
                for n_out in range(num_outputs):
                    for n_out_ in range(num_outputs):
                        if n_out == n_out_: continue
                        outputs = outputs_seg_mix[n_out]
                        mix_label = cut_mix_label[n_out_]
                        mix_loss += F.cross_entropy(outputs,  mix_label).mean()

                loss += mix_loss / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f' % (iter_num, loss))

            update_ema_variables(model, model_ema, iter_num)

            if iter_num >= 5000 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name == "LA":
                    # LA has no validation set
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=18,
                                                          stride_z=4,
                                                          dataset_name='LA')
                elif args.dataset_name == "Pancreas_CT":
                    # Pan has no validation set
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='Pancreas_CT')
                elif args.dataset_name == "BraTS2019":
                    # if we have validation set, we can use the following test strategy
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='BraTS2019val')

                logging.info('val iteration %d : avg dsc: %03f' % (iter_num, dice_sample))

                if dice_sample >= best_dice:
                    # most
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_best_path))
                    best_dice = dice_sample
                model.train()

            # if we have validation set, we can use the following test strategy
            if iter_num >= max_iterations:
                if args.dataset_name == "BraTS2019":
                    model.load_state_dict(torch.load(save_best_path))
                    model.eval()
                    dice_sample = test_patch.var_all_case(model,
                                                          num_classes=num_classes,
                                                          patch_size=patch_size,
                                                          stride_xy=16,
                                                          stride_z=16,
                                                          dataset_name='BraTS2019test')
                    logging.info('test: avg dsc: %03f' % (dice_sample))
                else:
                    torch.save(model.state_dict(), os.path.join(snapshot_path, '{}_last_model.pth'.format(args.model)))
                break

        if iter_num >= max_iterations:
            net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
            iterator.close()
            break