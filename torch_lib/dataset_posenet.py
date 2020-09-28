import cv2
import numpy as np
import os
import random
import math

import torch
from torchvision import transforms
from torch.utils import data

from library.File import *


# TODO: clean up where this is


def generate_bins(bins):
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2  # center of the bin

    return angle_bins


class Dataset(data.Dataset):
    def __init__(self, path, bins=8, overlap=0.1):

        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"
        # use a relative path instead?

        # TODO: which camera cal to use, per frame or global one?
        self.proj_matrix = get_P(os.path.abspath(os.path.dirname(
            os.path.dirname(__file__)) + '/camera_cal/calib_cam_to_cam.txt'))

        self.ids = [x.split('.')[0] for x in sorted(
            os.listdir(self.top_img_path))]  # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins  # 每一个Bin之间的间隔
        # self.interval = np.pi / (2*bins)  # 每一个Bin之间的间隔--patch
        for i in range(1, bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2  # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []

        for i in range(0, bins):
            self.bin_ranges.append(((i*self.interval - overlap) % (2*np.pi),
                                    (i*self.interval + self.interval + overlap) % (2*np.pi)))

        self.object_list = self.get_objects(self.ids)  # 长宽高

        # pre-fetch all labels
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            id = obj[0]
            line_num = obj[1]
            label = self.get_label(id, line_num)
            if id != last_id:
                self.labels[id] = {}
                last_id = id

            self.labels[id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None

    # should return (Input, Label)

    def __getitem__(self, index):
        id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if id != self.curr_id:
            self.curr_id = id
            self.curr_img = cv2.imread(self.top_img_path + '%s.jpg' % id)

        label = self.labels[id][str(line_num)]
        # P doesn't matter here
        obj = DetectedObject(
            self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)

        return obj.img, label

    def __len__(self):
        return len(self.object_list)

    def get_objects(self, ids):  # 长宽高
        objects = []
        for id in ids:
            with open(self.top_label_path + '%s.txt' % id) as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array([float(line[8]), float(
                        line[9]), float(line[10])], dtype=np.double)

                    objects.append((id, line_num))
        return objects

    def get_label(self, id, line_num):
        lines = open(self.top_label_path + '%s.txt' % id).read().splitlines()
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):

        bin_idxs = []

        def is_between(min, max, angle):
            max = (max - min) if (max - min) > 0 else (max - min) + 2*np.pi
            angle = (angle - min) if (angle -
                                      min) > 0 else (angle - min) + 2*np.pi
            return angle < max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def eulerAnglesToQu(self, theta):

        q = np.array([math.cos(theta[0]/2)*math.cos(theta[1]/2)*math.cos(theta[2]/2)+math.sin(theta[0]/2)*math.sin(theta[1]/2)*math.sin(theta[2]/2),
                      math.sin(theta[0]/2)*math.cos(theta[1]/2)*math.cos(theta[2]/2) -
                      math.cos(theta[0]/2)*math.sin(theta[1]/2) *
                      math.sin(theta[2]/2),
                      math.cos(theta[0]/2)*math.sin(theta[1]/2)*math.cos(theta[2]/2) +
                      math.sin(theta[0]/2)*math.cos(theta[1]/2) *
                      math.sin(theta[2]/2),
                      math.cos(theta[0]/2)*math.cos(theta[1]/2)*math.sin(theta[2]/2) -
                      math.sin(theta[0]/2)*math.sin(theta[1]/2) *
                      math.cos(theta[2]/2)
                      ])

        return q

    def translation(rotation_x, rotation_y, T):
        R_x = np.array([[1, 0, 0],
                        [0, +np.cos(rotation_x), -np.sin(rotation_x)],
                        [0, +np.sin(rotation_x), +np.cos(rotation_x)]],
                       dtype=np.float32)
        R_y = np.array([[+np.cos(rotation_y), 0, +np.sin(rotation_y)],
                        [0, 1, 0],
                        [-np.sin(rotation_y), 0, +np.cos(rotation_y)]],
                       dtype=np.float32)

        R = np.dot(R_x, R_y)

        T = T

        loc, error, rank, s = np.linalg.lstsq(R, T, rcond=None)

        return loc

    def format_label(self, line):
        line = line[:-1].split(' ')

        Class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        # Alpha = line[3] # what we will be regressing
        Patch = line[14]
        Yaw = line[15]
        Roll = line[16]

        if Yaw < 0:
            Yaw += 2 * np.pi
        # # if Patch >= 0 and Patch < (np.pi / 2):
        # #     Patch = Patch
        # # elif Patch >= (np.pi / 2) and Patch < np.pi:
        # #     Patch = Patch - (np.pi / 2)
        # # elif Patch >= np.pi and Patch < (3 * np.pi / 2):
        # #     Patch = Patch - np.pi
        # # elif Patch >= (3 * np.pi / 2) and Patch < 2 * np.pi:
        # #     Patch = Patch - (3 * np.pi / 2)
        # # else:
        # #     Patch = 0

        if Yaw >= 0 and Yaw < (np.pi / 2):
            Yaw = Yaw
        elif Yaw >= (np.pi / 2) and Yaw < np.pi:
            Yaw = Yaw - (np.pi / 2)
        elif Yaw >= np.pi and Yaw < (3 * np.pi / 2):
            Yaw = Yaw - np.pi
        elif Yaw >= (3 * np.pi / 2) and Yaw < 2 * np.pi:
            Yaw = Yaw - (3 * np.pi / 2)
        else:
            Yaw = 0

        top_left = (int(round(line[4])), int(round(line[5])))
        bottom_right = (int(round(line[6])), int(round(line[7])))
        Box_2D = [top_left, bottom_right]

        Dimension = np.array([line[8], line[9], line[10]],
                             dtype=np.double)  # height, width, length

        Location = [line[11], line[12], line[13]]  # x, y, z
        loction = np.array(Location, dtype=np.float32)
        # bring the KITTI center up to the middle of the object
        # Location[1] -= Dimension[0] / 2

        Rotation = np.array([line[14], line[15], line[16]], dtype=np.double)
        Qu = self.eulerAnglesToQu(Rotation)

        # patch
        Orientation_patch = np.zeros((self.bins, 2))
        Confidence_patch = np.zeros(self.bins)
        # raw
        Orientation_yaw = np.zeros((self.bins, 2))
        Confidence_yaw = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        # angle = Yaw + np.pi
        angle_patch = Patch
        angle_yaw = Yaw
        # angle = Patch + np.pi
        # patch is [0..pi/2]
        # angle = Patch

        bin_idxs_patch = self.get_bin(angle_patch)

        for bin_idx in bin_idxs_patch:
            angle_diff_patch = angle_patch - self.angle_bins[bin_idx]

            Orientation_patch[bin_idx, :] = np.array(
                [np.cos(angle_diff_patch), np.sin(angle_diff_patch)])
            Confidence_patch[bin_idx] = 1

        bin_idxs_yaw = self.get_bin(angle_yaw)

        for bin_idx in bin_idxs_yaw:
            angle_diff_yaw = angle_yaw - self.angle_bins[bin_idx]

            Orientation_yaw[bin_idx, :] = np.array(
                [np.cos(angle_diff_yaw), np.sin(angle_diff_yaw)])
            Confidence_yaw[bin_idx] = 1

        label = {
            'Class': Class,
            'Box_2D': Box_2D,
            'Dimensions': Dimension,
            'Location': loction,
            'Patch': Patch,
            'Yaw': Yaw,
            'Orientation_patch': Orientation_patch,
            'Confidence_patch': Confidence_patch,
            'Orientation_yaw': Orientation_yaw,
            'Confidence_yaw': Confidence_yaw,
            'Qu': Qu
        }

        return label

    # will be deprc soon
    def parse_label(self, label_path):
        buf = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line[:-1].split(' ')

                Class = line[0]
                if Class == "DontCare":
                    continue

                for i in range(1, len(line)):
                    line[i] = float(line[i])

                Alpha = line[3]  # what we will be regressing
                Patch = line[14]
                Yaw = line[15]
                Roll = line[16]

                if Yaw < 0:
                    Yaw += 2 * np.pi
                # # if Patch >= 0 and Patch < (np.pi / 2):
                # #     Patch = Patch
                # # elif Patch >= (np.pi / 2) and Patch < np.pi:
                # #     Patch = Patch - (np.pi / 2)
                # # elif Patch >= np.pi and Patch < (3 * np.pi / 2):
                # #     Patch = Patch - np.pi
                # # elif Patch >= (3 * np.pi / 2) and Patch < 2 * np.pi:
                # #     Patch = Patch - (3 * np.pi / 2)
                # # else:
                # #     Patch = 0

                if Yaw >= 0 and Yaw < (np.pi / 2):
                    Yaw = Yaw
                elif Yaw >= (np.pi / 2) and Yaw < np.pi:
                    Yaw = Yaw - (np.pi / 2)
                elif Yaw >= np.pi and Yaw < (3 * np.pi / 2):
                    Yaw = Yaw - np.pi
                elif Yaw >= (3 * np.pi / 2) and Yaw < 2 * np.pi:
                    Yaw = Yaw - (3 * np.pi / 2)
                else:
                    Yaw = 0

                top_left = (int(round(line[4])), int(round(line[5])))
                bottom_right = (int(round(line[6])), int(round(line[7])))
                Box_2D = [top_left, bottom_right]

                # height, width, length
                Dimension = [line[8], line[9], line[10]]
                Location = [line[11], line[12], line[13]]  # x, y, z
                loction = np.array(Location, dtype=np.float32)
                # bring the KITTI center up to the middle of the object
                # Location[1] -= Dimension[0] / 2

                Rotation = np.array([line[14], line[15], line[16]], dtype=np.double)
                Qu = self.eulerAnglesToQu(Rotation)

                buf.append({
                    'Class': Class,
                    'Box_2D': Box_2D,
                    'Dimensions': Dimension,
                    'Location': loction,
                    'Alpha': Alpha,
                    'Yaw': Yaw,
                    'Patch': Patch,
                    'Roll': Roll,
                    'Qu': Qu
                })
        return buf

    # will be deprc soon
    def all_objects(self):
        data = {}
        for id in self.ids:
            data[id] = {}
            img_path = self.top_img_path + '%s.png' % id
            img = cv2.imread(img_path)
            data[id]['Image'] = img_path

            # using p per frame
            calib_path = self.top_calib_path + '%s.txt' % id
            proj_matrix = get_calibration_cam_to_image(calib_path)

            # using P_rect from global calib file
            proj_matrix = self.proj_matrix

            data[id]['Calib'] = proj_matrix

            label_path = self.top_label_path + '%s.txt' % id
            labels = self.parse_label(label_path)
            objects = []
            for label in labels:
                box_2d = label['Box_2D']
                detection_class = label['Class']
                objects.append(DetectedObject(
                    img, detection_class, box_2d, proj_matrix, label=label))

            data[id]['Objects'] = objects

        return data


"""
What is *sorta* the input to the neural net. Will hold the cropped image and
the angle to that image, and (optionally) the label for the object. The idea
is to keep this abstract enough so it can be used in combination with YOLO
"""


class DetectedObject:
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):

        if isinstance(proj_matrix, str):  # filename
            # proj_matrix = get_P(proj_matrix)
            proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = self.calc_theta_ray(img, box_2d, proj_matrix)
        self.img = self.format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class

    def calc_theta_ray(self, img, box_2d, proj_matrix):
        width = img.shape[1]
        fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
        center = (box_2d[1][0] + box_2d[0][0]) / 2
        dx = center - (width / 2)

        mult = 1
        if dx < 0:
            mult = -1
        dx = abs(dx)
        angle = np.arctan((2*dx*np.tan(fovx/2)) / width)
        angle = angle * mult

        return angle

    def format_img(self, img, box_2d):

        # Should this happen? or does normalize take care of it. YOLO doesnt like
        # img=img.astype(np.float) / 255

        # torch transforms
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        process = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        # crop image
        pt1 = box_2d[0]
        pt2 = box_2d[1]
        crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
        crop = cv2.resize(src=crop, dsize=(224, 224),
                          interpolation=cv2.INTER_CUBIC)

        # recolor, reformat
        batch = process(crop)

        return batch
