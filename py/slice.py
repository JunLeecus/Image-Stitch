"""
Created on 2020-05-07 11:57:29
@Author: xxx
@Version : 1
"""
import cv2
import numpy as np
from typing import List

def slice(img):
    """
    将图像分割成32*32的小块，并构建成(?, 32, 32, 1)以输入L2-Net。
    边缘不足32*32的部分，采取由边缘回退32像素进行裁取，会有部分重叠。
    采取补零可能会影像拼接准确性。
    Args:
        img: 输入图像

    Returns:
        ros：region of select，shape为(?, 32, 32, 1)的图像
        keypoints：region of select的位置、尺寸、角度等参数

    """
    print('Processing image ...')
    image_size = 32
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[1]
    w = img.shape[0]

    image_w = 0
    ros: np.ndarray = None
    keypoints: List[cv2.KeyPoint] = []

    while image_w < w:
        image_h = 0
        if image_w + image_size <= w:
            while image_h < h:
                if image_h + image_size <= h:
                    image = img[image_w:image_w + image_size, image_h:image_h + image_size]
                    image = np.expand_dims(image, axis=0)
                    image = np.expand_dims(image, axis=3)
                    if ros is None:
                        ros = image
                        keypoints.append(cv2.KeyPoint(x=image_w + 16, y=image_h + 16
                                                             , _size=image_size
                                                             , _angle=-1
                                                             , _response=0.018
                                                             , _octave=1
                                                             , _class_id=-1))
                    else:
                        ros = np.concatenate([ros, image], axis=0)
                        keypoints.append(cv2.KeyPoint(x=image_w + 16, y=image_h + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    image_h = image_h + image_size
                else:
                    image = img[image_w:image_w + image_size, h - image_size:]
                    image = np.expand_dims(image, axis=0)
                    image = np.expand_dims(image, axis=3)
                    if ros is None:
                        ros = image
                        keypoints.append(cv2.KeyPoint(x=image_w + 16, y=h - image_size + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    else:
                        ros = np.concatenate([ros, image], axis=0)
                        keypoints.append(cv2.KeyPoint(x=image_w + 16, y=h - image_size + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    break
            image_w = image_w + image_size
        else:
            while image_h < h:
                if image_h + image_size <= h:
                    image = img[w - image_size:, image_h:image_h + image_size]
                    image = np.expand_dims(image, axis=0)
                    image = np.expand_dims(image, axis=3)
                    if ros is None:
                        ros = image
                        keypoints.append(cv2.KeyPoint(x=w - image_size + 16, y=image_h + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    else:
                        ros = np.concatenate([ros, image], axis=0)
                        keypoints.append(cv2.KeyPoint(x=w - image_size + 16, y=image_h + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    image_h = image_h + image_size
                else:
                    image = img[w - image_size:, h - image_size:]
                    image = np.expand_dims(image, axis=0)
                    image = np.expand_dims(image, axis=3)
                    if ros is None:
                        ros = image
                        keypoints.append(cv2.KeyPoint(x=w - image_size + 16, y=h - image_size + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    else:
                        ros = np.concatenate([ros, image], axis=0)
                        keypoints.append(cv2.KeyPoint(x=w - image_size + 16, y=h - image_size + 16
                                                      , _size=image_size
                                                      , _angle=-1
                                                      , _response=0.018
                                                      , _octave=1
                                                      , _class_id=-1))
                    break
            image_w = image_w + image_size
    print('Image is slice to %d region.' % ros.shape[0])
    return ros, keypoints
