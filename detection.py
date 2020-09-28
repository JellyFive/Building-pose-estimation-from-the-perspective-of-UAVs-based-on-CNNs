from torch_lib.dataset_det import *
from torch_lib.posenet import Model, OrientationLoss, FocalLoss
from library.Math import *
from library.Plotting import *

import os
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg, resnet, mobilenet, inception_v3
import numpy as np
import torch.nn.functional as F


import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
# mpl.use('Qt5Agg')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_corners(dimensions, location, rotation_x, rotation_y, rotation_z):
    R_x = np.array([[1, 0, 0],
                    [0, +np.cos(rotation_x), -np.sin(rotation_x)],
                    [0, +np.sin(rotation_x), +np.cos(rotation_x)]],
                   dtype=np.float32)
    R_y = np.array([[+np.cos(rotation_y), 0, +np.sin(rotation_y)],
                    [0, 1, 0],
                    [-np.sin(rotation_y), 0, +np.cos(rotation_y)]],
                   dtype=np.float32)
    R_z = np.array([[+np.cos(rotation_z), -np.sin(rotation_z), 0],
                    [+np.sin(rotation_z), +np.cos(rotation_z), 0],
                    [0, 0, 1]],
                   dtype=np.float32)
    R = np.dot(R_x, R_y)
    h, w, l = dimensions
    # 世界坐标系的建立
    x_corners = [l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [-w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, w / 2]

    corners_3D = np.dot(R, [x_corners, y_corners, z_corners])

    location = np.array(location)

    corners_3D += location.reshape((3, 1))
    return corners_3D


def draw_projection(corners, P2, ax, color):
    projection = np.dot(P2, np.vstack([corners, np.ones(8, dtype=np.int32)]))
    projection = (projection / projection[2])[:2]
    orders = [[0, 1, 2, 3, 0],
              [4, 5, 6, 7, 4],
              [2, 6], [3, 7],
              [1, 5], [0, 4]]
    for order in orders:
        ax.plot(projection[0, order], projection[1, order],
                color=color, linewidth=2)
    return


def array_dist(pred, target):    # return np.linalg.norm(pred - target, 2)
    return abs(target - pred)


def main():

    weights_path = '/home/lab/Desktop/wzndeep/posenet-build--location/weights/'
    model_lst = [x for x in sorted(
        os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s' % model_lst[-1])
        base_model = mobilenet.mobilenet_v2(pretrained=True)  # 加载模型并设置为预训练模式
        model = Model(base_model=base_model).cuda()
        checkpoint = torch.load(weights_path + '/%s' % model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # defaults to /eval
    eval_path = '/home/lab/Desktop/wzndeep/BuildingData/realBuilding'
    dataset = Dataset(eval_path)  # 自定义的数据集

    all_images = dataset.all_objects()
    for key in sorted(all_images.keys()):

        start_time = time.time()

        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for detectedObject in objects:
            label = detectedObject.label
            # theta_ray = detectedObject.theta_ray
            input_img = detectedObject.img

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img
            input_tensor.float().cuda()

            [out_loc, orient_patch, conf_patch, orient_yaw,
                conf_yaw] = model(input_tensor)

            out_loc = out_loc.cpu().data.numpy()

            orient_patch = orient_patch.cpu().data.numpy()[0, :, :]
            conf_patch = conf_patch.cpu().data.numpy()[0, :]
            orient_yaw = orient_yaw.cpu().data.numpy()[0, :, :]
            conf_yaw = conf_yaw.cpu().data.numpy()[0, :]

            argmax_patch = np.argmax(conf_patch)
            orient_patch = orient_patch[argmax_patch, :]
            cos = orient_patch[0]
            sin = orient_patch[1]
            patch = np.arctan2(sin, cos)
            patch += dataset.angle_bins[argmax_patch]

            argmax_yaw = np.argmax(conf_yaw)
            orient_yaw = orient_yaw[argmax_yaw, :]
            cos_yaw = orient_yaw[0]
            sin_yaw = orient_yaw[1]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            yaw += dataset.angle_bins[argmax_yaw]
            if (yaw > (2 * np.pi)):
                yaw -= (2 * np.pi)

        print('Estimated patch|Estimated yaw: {:.3f}/{:.3f}'.format(
            patch, yaw))
        print('loc estimated', out_loc[0])

        print('-' * 50)

        # image = Image.open(truth_img)
        # fig = plt.figure(figsize=(8, 8))

        # 绘制3DBBOX
        # ax = fig.gca()
        # ax.grid(False)
        # ax.set_axis_off()
        # ax.imshow(image)
        # estmate_corners = get_corners(dim, truth_loc, patch, yaw, r_z)
        # draw_projection(estmate_corners, cam_to_img, ax, 'green')

        # plt.show()
        plt.savefig(
            '/home/lab/Desktop/wzndeep/posenet-build--location/output-image/{}_proj'.format(key))
        plt.close()


if __name__ == '__main__':
    main()
