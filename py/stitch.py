# -*- coding: utf-8 -*-
"""
Created on 2020-05-07 11:57:29
@Author: xxx
@Version : 1.1
"""
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import pymssql        #需忽略警告的模块

from enum import Enum
from typing import List, Tuple, Union
from slice import slice
import tensorflow as tf
import vgg16
import unittest
import os
import random
from numpy import *

import cv2
import numpy as np


import k_means
import ransac
import blend


def show_image(image: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).show()


class Method(Enum):

    SURF = cv2.xfeatures2d.SURF_create
    SIFT = cv2.xfeatures2d.SIFT_create
    ORB = cv2.ORB_create


colors = ((123, 234, 12), (23, 44, 240), (224, 120, 34), (21, 234, 190),
          (80, 160, 200), (243, 12, 100), (25, 90, 12), (123, 10, 140))


class Area:

    def __init__(self, *points):

        self.points = list(points)

    def is_inside(self, x: Union[float, Tuple[float, float]], y: float=None) -> bool:
        if isinstance(x, tuple):
            x, y = x
        raise NotImplementedError()


class Matcher():

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SIFT, threshold=800, ratio=400) -> None:
        """输入两幅图像，计算其特征值
        此类用于输入两幅图像，计算其特征值，输入两幅图像分别为numpy数组格式的图像，
        其中的method参数要求输入SURF、SIFT或者ORB，threshold参数为特征值检测所需的阈值。

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            method (Enum, optional): Defaults to Method.SIFT. 特征值检测方法
            ratio (int, optional): Defaults to 400. L2-Net特征向量比重
            threshold (int, optional): Defaults to 800. 特征值阈值

        """

        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.ratio = ratio
        self.ros1: np.ndarray = None
        self.ros2: np.ndarray = None
        self.loc1: List = None
        self.loc2: List = None
        self.threshold = threshold

        self._keypoints1: List[cv2.KeyPoint] = None
        self._descriptors1: np.ndarray = None
        self._keypoints2: List[cv2.KeyPoint] = None
        self._descriptors2: np.ndarray = None

        if self.method == Method.ORB:
            # error if not set this
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # self.matcher = cv2.BFMatcher(crossCheck=True)
            self.matcher = cv2.FlannBasedMatcher()

        self.match_points = []

        self.image_points1 = np.array([])
        self.image_points2 = np.array([])

    def compute_keypoint(self) -> None:
        """计算特征点
        利用给出的特征值检测方法对图像进行特征值检测。

        Args:
            image (np.ndarray): 图像
        """
        print('Compute keypoint by SIFT')

        feature = self.method.value(self.threshold)
        self._keypoints1, self._descriptors1 = feature.detectAndCompute(
            self.image1, None)
        self._keypoints2, self._descriptors2 = feature.detectAndCompute(
            self.image2, None)

    def get_the_ros(self):
        """选取待匹配的特征块
                将灰度图像选取特定区域，并转换维度为(?, 32, 32, 1)
                """
        self.ros1, self.loc1 = slice(self.image1)
        self.ros2, self.loc2 = slice(self.image2)

    def compute_kepoint_by_L2_Net(self) -> None:
        """
        通过VGG计算描述向量+PCA
        筛选并存储描述向量和keypoint。

        """
        self.get_the_ros()
        print('Compute keypoint by VGG')
        batch1 = self.ros1.shape[0]
        batch2 = self.ros2.shape[0]
        batch = 64

        batch_num = 0
        print("First image")
        while batch_num < batch1:
            if batch_num + batch > batch1:
                batch_end = batch1
            else:
                batch_end = batch_num + batch
            with tf.Session() as sess:
                images1 = tf.placeholder("float", [batch_end - batch_num, 64, 64, 3])
                feed_dict1 = {images1: self.ros1[batch_num:batch_end, :, :, :]}
                print('Processing %d image ...' % batch_num)
                vgg1 = vgg16.Vgg16()
                with tf.name_scope("content_vgg"):
                    vgg1.build(images1)
                ros1_descriptors1 = np.array(self.ratio * sess.run(vgg1.prob, feed_dict=feed_dict1), dtype=np.float32)
                ros1_descriptors1 = np.squeeze(ros1_descriptors1)
                # print("PCA ...")
                ros1_descriptors1, _, _ = pca(ros1_descriptors1, 128)
                ros1_descriptors1 = np.array(ros1_descriptors1, dtype=np.float32)

                print("Image %d compute finish ... Processing ..." % batch_end)
                self._descriptors1 = np.concatenate((self._descriptors1, ros1_descriptors1), axis=0)
                self._keypoints1.extend(self.loc1[batch_num:batch_end])
                # print("Processing finish ...")
            batch_num = batch_end

        batch_num = 0
        print("Second image")
        while batch_num < batch2:
            if batch_num + batch > batch2:
                batch_end = batch2
            else:
                batch_end = batch_num + batch
            with tf.Session() as sess:
                images2 = tf.placeholder("float", [batch_end - batch_num, 64, 64, 3])
                feed_dict2 = {images2: self.ros2[batch_num:batch_end, :, :, :]}
                print('Processing %d image ...' % batch_num)
                vgg2 = vgg16.Vgg16()
                with tf.name_scope("content_vgg"):
                    vgg2.build(images2)
                ros2_descriptors2 = np.array(self.ratio * sess.run(vgg2.prob, feed_dict=feed_dict2), dtype=np.float32)
                ros2_descriptors2 = np.squeeze(ros2_descriptors2)
                # print("PCA ...")
                ros2_descriptors2, _, _ = pca(ros2_descriptors2, 128)
                ros2_descriptors2 = np.array(ros2_descriptors2, dtype=np.float32)
                print("Image %d compute finish ... Processing ..." % batch_end)
                self._descriptors2 = np.concatenate((self._descriptors2, ros2_descriptors2), axis=0)
                self._keypoints2.extend(self.loc2[batch_num:batch_end])
                # print("Processing finish ...")
            batch_num = batch_end

        # 描述符比较小，均0.0x 需将其扩大


    def match(self, max_match_lenth=20, threshold=0.04, show_match=False):
        """对两幅图片计算得出的特征值进行匹配，对ORB来说使用OpenCV的BFMatcher算法，而对于其他特征检测方法则使用FlannBasedMatcher算法。

            max_match_lenth (int, optional): Defaults to 20. 最大匹配点数量
            threshold (float, optional): Defaults to 0.04. 默认最大匹配距离差
            show_match (bool, optional): Defaults to False. 是否展示匹配结果
        """

        self.compute_keypoint()
        self.compute_kepoint_by_L2_Net()

        '''计算两张图片中的配对点，并至多取其中最优的`max_match_lenth`个'''
        self.match_points = sorted(self.matcher.match(
            self._descriptors1, self._descriptors2), key=lambda x: x.distance)

        match_len = min(len(self.match_points), max_match_lenth)

        # in case distance is 0
        max_distance = max(2 * self.match_points[0].distance, 20)

        for i in range(match_len):
            if self.match_points[i].distance > max_distance:
                match_len = i
                break
        print('max distance: ', self.match_points[match_len].distance)
        print("Min distance: ", self.match_points[0].distance)
        print('match_len: ', match_len)
        assert(match_len >= 4)
        self.match_points = self.match_points[:match_len]

        if show_match:
            img3 = cv2.drawMatches(self.image1, self._keypoints1, self.image2, self._keypoints2,
                                   self.match_points, None, flags=0)
            show_image(img3)
            # cv2.imwrite('../resource/3-sift-match.jpg', img3)

        '''由最佳匹配取得匹配点对，并进行形变拼接'''
        image_points1, image_points2 = [], []
        for i in self.match_points:
            image_points1.append(self._keypoints1[i.queryIdx].pt)
            image_points2.append(self._keypoints2[i.trainIdx].pt)

        self.image_points1 = np.float32(image_points1)
        self.image_points2 = np.float32(image_points2)

        # print(image_points1)


def get_weighted_points(image_points: np.ndarray):

    average = np.average(image_points, axis=0)

    max_index = np.argmax(np.linalg.norm((image_points - average), axis=1))
    return np.append(image_points, np.array([image_points[max_index]]), axis=0)


def pca(image, rate=128):
    mean_value = mean(image, axis=0)
    image = image-mean_value
    c = cov(image, rowvar=0)
    eigvalue, eigvector = linalg.eig(mat(c))

    index_vec = np.argsort(-eigvalue)
    n_largest_index = index_vec[:rate]
    t = eigvector[:, n_largest_index]
    new_image = np.dot(image, t)
    return new_image, t, mean_value

class Stitcher:

    def __init__(self, image1: np.ndarray, image2: np.ndarray, method: Enum=Method.SIFT, use_kmeans=False):
        """输入图像和匹配，对图像进行拼接
        目前采用简单矩阵匹配和平均值拼合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二
            matcher (Matcher): 匹配结果
            use_kmeans (bool): 是否使用kmeans 优化点选择
        """

        self.image1 = image1
        self.image2 = image2
        self.method = method
        self.use_kmeans = use_kmeans
        self.matcher = Matcher(image1, image2, method=method)
        self.M = np.eye(3)

        self.image = None

    def stich(self, show_result=True, max_match_lenth=40, show_match_point=True, use_partial=False, use_new_match_method=False, use_gauss_blend=True):
        """对图片进行拼合

            show_result (bool, optional): Defaults to True. 是否展示拼合图像
            show_match_point (bool, optional): Defaults to True. 是否展示拼合点
        """
        self.matcher.match(max_match_lenth=max_match_lenth,
                           show_match=show_match_point)

        if self.use_kmeans:
            self.image_points1, self.image_points2 = k_means.get_group_center(
                self.matcher.image_points1, self.matcher.image_points2)
        else:
            self.image_points1, self.image_points2 = (
                self.matcher.image_points1, self.matcher.image_points2)

        if use_new_match_method:
            self.M = ransac.GeneticTransform(self.image_points1, self.image_points2).run()
        else:
            self.M, _ = cv2.findHomography(
                self.image_points1, self.image_points2, method=cv2.RANSAC)
            # self.M = ransac.Ransac(self.image_points1, self.image_points2).run()

        print("Good points and average distance: ", ransac.GeneticTransform.get_value(
            self.image_points1, self.image_points2, self.M))

        left, right, top, bottom = self.get_transformed_size()
        # print(self.get_transformed_size())
        width = int(max(right, self.image2.shape[1]) - min(left, 0))
        height = int(max(bottom, self.image2.shape[0]) - min(top, 0))
        print(width, height)
        # width, height = min(width, 10000), min(height, 10000)
        if width * height > 8000 * 5000:
            # raise MemoryError("Too large to get the combination")
            factor = width*height/(8000*5000)
            width = int(width/factor)
            height = int(height/factor)

        if use_partial:
            self.partial_transform()

        # 移动矩阵
        self.adjustM = np.array(
            [[1, 0, max(-left, 0)],  # 横向
             [0, 1, max(-top, 0)],  # 纵向
             [0, 0, 1]
             ], dtype=np.float64)
        # print('adjustM: ', adjustM)
        self.M = np.dot(self.adjustM, self.M)
        transformed_1 = cv2.warpPerspective(
            self.image1, self.M, (width, height))
        transformed_2 = cv2.warpPerspective(
            self.image2, self.adjustM, (width, height))

        self.image = self.blend(transformed_1, transformed_2, use_gauss_blend=use_gauss_blend)

        if show_match_point:
            for point1, point2 in zip(self.image_points1, self.image_points2):
                point1 = self.get_transformed_position(tuple(point1))
                point1 = tuple(map(int, point1))
                point2 = self.get_transformed_position(tuple(point2), M=self.adjustM)
                point2 = tuple(map(int, point2))

                # cv2.line(self.image, point1, point2, random.choice(colors), 3)
                cv2.circle(self.image, point1, 10, (20, 20, 255), 5)
                cv2.circle(self.image, point2, 8, (20, 200, 20), 5)
        if show_result:
            show_image(self.image)

    def blend(self, image1: np.ndarray, image2: np.ndarray, use_gauss_blend=True) -> np.ndarray:
        """对图像进行融合

        Args:
            image1 (np.ndarray): 图像一
            image2 (np.ndarray): 图像二

        Returns:
            np.ndarray: 融合结果
        """

        mask = self.generate_mask(image1, image2)
        print("Blending")
        if use_gauss_blend:
            result = blend.gaussian_blend(image1, image2, mask, mask_blend=10)
        else:
            result = blend.direct_blend(image1, image2, mask, mask_blend=0)

        return result

    def generate_mask(self, image1: np.ndarray, image2: np.ndarray):
        """生成供融合使用的遮罩，由变换后图像的垂直平分线来构成分界线

        Args:
            shape (tuple): 遮罩大小

        Returns:
            np.ndarray: 01数组
        """
        print("Generating mask")
        # x, y
        center1 = self.image1.shape[1] / 2, self.image1.shape[0] / 2
        center1 = self.get_transformed_position(center1)
        center2 = self.image2.shape[1] / 2, self.image2.shape[0] / 2
        center2 = self.get_transformed_position(center2, M=self.adjustM)
        # 垂直平分线 y=-(x2-x1)/(y2-y1)* [x-(x1+x2)/2]+(y1+y2)/2
        x1, y1 = center1
        x2, y2 = center2

        # note that opencv is (y, x)
        def function(y, x, *z):
            return (y2 - y1) * y < -(x2 - x1) * (x - (x1 + x2) / 2) + (y2 - y1) * (y1 + y2) / 2

        mask = np.fromfunction(function, image1.shape)

        # mask = mask&_i2+mask&i1+i1&_i2
        mask = np.logical_and(mask, np.logical_not(image2)) \
            + np.logical_and(mask, image1)\
            + np.logical_and(image1, np.logical_not(image2))

        return mask

    def get_transformed_size(self) ->Tuple[int, int, int, int]:
        """计算形变后的边界
        计算形变后的边界，从而对图片进行相应的位移，保证全部图像都出现在屏幕上。

        Returns:
            Tuple[int, int, int, int]: 分别为左右上下边界
        """

        conner_0 = (0, 0)  # x, y
        conner_1 = (self.image1.shape[1], 0)
        conner_2 = (self.image1.shape[1], self.image1.shape[0])
        conner_3 = (0, self.image1.shape[0])
        points = [conner_0, conner_1, conner_2, conner_3]

        # top, bottom: y, left, right: x
        top = min(map(lambda x: self.get_transformed_position(x)[1], points))
        bottom = max(
            map(lambda x: self.get_transformed_position(x)[1], points))
        left = min(map(lambda x: self.get_transformed_position(x)[0], points))
        right = max(map(lambda x: self.get_transformed_position(x)[0], points))

        return left, right, top, bottom

    def get_transformed_position(self, x: Union[float, Tuple[float, float]], y: float=None, M=None) -> Tuple[float, float]:
        """求得某点在变换矩阵（self.M）下的新坐标

        Args:
            x (Union[float, Tuple[float, float]]): x坐标或(x,y)坐标
            y (float, optional): Defaults to None. y坐标，可无
            M (np.ndarray, optional): Defaults to None. 利用M进行坐标变换运算

        Returns:
            Tuple[float, float]:  新坐标
        """

        if isinstance(x, tuple):
            x, y = x
        p = np.array([x, y, 1])[np.newaxis].T
        if M is not None:
            M = M
        else:
            M = self.M
        pa = np.dot(M, p)
        return pa[0, 0] / pa[2, 0], pa[1, 0] / pa[2, 0]


def equalhist(image):
    """
    图像预处理，进行直方图均衡化。
    Args:
        image: 输入图像

    Returns:
        均衡化后的图像

    """
    (B, G, R) = cv2.split(image)
    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)
    image = cv2.merge([B, G, R])
    return image


def main():
    unittest.main()


if __name__ == "__main__":
    import time
    # main()
    os.chdir(os.path.dirname(__file__))

    start_time = time.time()
    img1 = cv2.imread("../example/11.png")
    img2 = cv2.imread("../example/12.png")

    img1 = equalhist(img1)
    img2 = equalhist(img2)

    stitcher = Stitcher(img1, img2, Method.SIFT, False)
    stitcher.stich(max_match_lenth=50, use_partial=False, use_new_match_method=1, use_gauss_blend=0)

    # cv2.imwrite('../example/merge.jpg', stitcher.image)

    print("Time: ", time.time() - start_time)
    print("M: ", stitcher.M)
