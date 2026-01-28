import argparse
import os
from os.path import join

import h5py
import math
import numpy as np
import torch
from medpy import metric
from skimage.measure import label
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BraTS2019', help='dataset_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--gpu', type=str, default='25', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-procssing?')

FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


num_classes = 2
if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    FLAGS.root_path = '../data/LA'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list_ = f.readlines()
    image_list = [
        FLAGS.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list_
    ]

elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    FLAGS.root_path = '../data/Pancreas'
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list_ = f.readlines()
    image_list = [FLAGS.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list_]

elif FLAGS.dataset_name == "BraTS2019val":
    with open('./data/BraTS2019/val.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["./data/BraTS2019/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]

elif FLAGS.dataset_name == "BraTS2019test":
    with open('./data/BraTS2019/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = ["./data/BraTS2019/data/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def calculate_metric_percase(pred, gt):
    dice = metric.binary.dc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, hd, asd


def test_calculate_metric():
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_mode_path = os.path.join(snapshot_path, '{}_last_model.pth'.format(FLAGS.model))

    # most sota methods use best model, which is saved according to validation performance on test set.
    # so their test performance may be much better than the one using last model.

    # save_mode_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))

    net.load_state_dict(torch.load(save_mode_path), strict=False)
    print("init weight from {}".format(save_mode_path))
    net.eval()

    if FLAGS.dataset_name == "LA":
        test_all_case(FLAGS.model,
                                   2,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(112, 112, 80),
                                   stride_xy=18,
                                   stride_z=4,
                                   save_result=False,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)
    elif FLAGS.dataset_name == "Pancreas_CT":
        test_all_case(FLAGS.model,
                                   2,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(96, 96, 96),
                                   stride_xy=16,
                                   stride_z=16,
                                   save_result=False,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)

    elif FLAGS.dataset_name == "BraTS2019":
        test_all_case(FLAGS.model,
                                   2,
                                   net,
                                   image_list,
                                   num_classes=num_classes,
                                   patch_size=(96, 96, 96),
                                   stride_xy=16,
                                   stride_z=16,
                                   save_result=False,
                                   test_save_path=test_save_path,
                                   metric_detail=FLAGS.detail,
                                   nms=FLAGS.nms)


def test_all_case(model_name,
                  num_outputs,
                  model,
                  image_list,
                  num_classes,
                  patch_size=(112, 112, 80),
                  stride_xy=18,
                  stride_z=4,
                  save_result=False,
                  test_save_path=None,
                  preproc_fn=None,
                  metric_detail=1,
                  nms=0):

    if save_result:
        if not os.path.exists(test_save_path):
            os.makedirs(test_save_path)
        print(test_save_path)

    loader = tqdm(image_list) if not metric_detail else image_list
    ith = 0
    total_metric = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction = test_single_output(model,
                                                   image,
                                                   stride_xy,
                                                   stride_z,
                                                   patch_size)

        if nms:
            # all results are not post-processed by default
            prediction = getLargestCC(prediction)


        if np.sum(prediction) == 0:
            metric_temp = (0, 0, 100, 100)
        else:
            metric_temp = calculate_metric_percase(prediction, label[:])


        if metric_detail:
            with open(snapshot_path + '/{}_performance.txt'.format(model_name), 'a+') as f:
                f.writelines('patient {}  is {} \n'.format(image_path, metric_temp))
                print('patient {} is {} \n'.format(image_path, metric_temp))

        total_metric += np.asarray(metric_temp)

        if save_result:
            if FLAGS.dataset_name == "LA":
                item = image_path.split('/')[4]
            elif FLAGS.dataset_name == "Pancreas_CT":
                item = image_path.split('/')[4][:-8]
            elif FLAGS.dataset_name == "BraTS2019":
                item = image_path.split('/')[4][:-5]
            np.save(test_save_path + '/{}.npy'.format(item), prediction.astype(np.uint8))
        ith += 1

    avg_metric = total_metric / len(image_list)
    print('average metric is {}'.format(avg_metric))

    with open(snapshot_path + '/{}_performance.txt'.format(model_name), 'a+') as f:
        f.writelines('average metric is {} \n'.format(avg_metric))


def test_single_output(net, image, stride_xy, stride_z, patch_size):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
                       mode='constant',
                       constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1

    score_map = np.zeros((1,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y_logit = list(net(test_patch))
                    y = torch.softmax(y_logit[0], dim=1).cpu().data.numpy()[0, 1, :, :, :]

                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    = cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] + 1

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = (score_map[0] > 0.5).astype(int)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]

    return label_map


if __name__ == '__main__':
    path_list = ['../checkpoints/LA_TCSeg_0.1_0.85_0.05_4444_16_labeled/vnet']
    for path in path_list:
        snapshot_path = path
        test_save_path = join(path[:-10], "{}_predictions".format(FLAGS.model))
        metric = test_calculate_metric()
        print(metric)