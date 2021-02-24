#!/usr/bin/python3
# -*- coding:utf-8 -*-
import os, shutil
import random
import numpy as np
import cv2
import time
import threading


def resize_and_padding(img, dist_w, dist_h):
    src_h, src_w = img.shape
    resize_w = int(float(src_w) * float(dist_h) / float(src_h))
    if resize_w >= dist_w:
        return cv2.resize(img, (dist_w, dist_h))
    else:
        img = cv2.resize(img, (resize_w, dist_h))
        # right padding 0
        return np.pad(img, ((0, 0), (0, dist_w - resize_w)), 'constant', constant_values=(0, 0))


def clip(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def change_brightness_and_contrast(img):
    alpha = 0.8 + 0.4 * random.random()
    beta = -20 + 40 * random.random()
    img = img.astype('float')
    img[:, :, :3] = img[:, :, :3] * (1.0 * alpha) + beta
    return clip(img)


def add_random_resize(img):
    src_h, src_w, ch = img.shape
    ratio = 0.3 + random.random() * 0.1
    img = cv2.resize(img, (int(ratio * float(src_w)), int(ratio * float(src_h))))
    return img


def add_salt_noise(img):
    for i in range(int(100 * random.random())):
        temp_y = np.random.randint(0, img.shape[0])
        temp_x = np.random.randint(0, img.shape[1])
        img[temp_y][temp_x] = int(255 * random.random())
    return img


def add_gaussian_noise(img, mean=0, var=0.001):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, img.shape)
    out = img + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


def add_erode_dilate(img):
    if random.random() < 0.5:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.erode(img, kernel)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.dilate(img, kernel)
    return img


def add_gaussian_blur(img):
    kernel = random.choice([1, 3, 5, 7])
    img = cv2.GaussianBlur(img, (kernel, kernel), 0)
    return img


def add_motion_blur(img):
    img = np.array(img)
    degree = random.choice([2, 4, 6, 8])
    angle = 45
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)

    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)

    return blurred


def change_HSV(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h = h * (0.8 + np.random.random() * 0.2)
    s = s * (0.3 + np.random.random() * 0.7)
    v = v * (0.2 + np.random.random() * 0.8)
    # hue_shift = np.random.uniform(-5, 5)
    # h = h + hue_shift
    # sat_shift = np.random.uniform(-5, 5)
    # s = s + sat_shift
    # val_shift = np.random.uniform(-5, 5)
    # v = v + val_shift
    # final_hsv = cv2.merge((h, s, v))
    hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2] = h, s, v
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def add_gamma_correction(img):
    gamma = 0.6 + random.random() * 0.9
    img = np.power(img / float(np.max(img)), gamma)
    img = np.uint8(img * 255)
    img = clip(img)
    return img


def img_augmentation(img):
    global i
    # str='newimg%c',i
    # 随机resize
    if random.random() < 1:
        img = add_random_resize(img)
    # 随机改变亮度对比度
    if random.random() < 0.3:
        img = change_brightness_and_contrast(img)
    # 随机修改HSV
    if random.random() < 0.3:
        img = change_HSV(img)
    # 随机gamma修正
    if random.random() < 0.3:
        img = add_gamma_correction(img)
    # 随机高斯模糊
    if random.random() < 0.3:
        img = add_gaussian_blur(img)
    # 随机运动模糊
    if random.random() < 0.1:
        img = add_motion_blur(img)
    # 随机腐蚀膨胀
    if random.random() < 0.1:
        img = add_erode_dilate(img)
    # 随机加上点噪声
    # if random.random()<0.2:
    #    img = add_salt_noise(img)
    # 随机加上高斯噪声
    if random.random() < 0.3:
        img = add_gaussian_noise(img)

    return img


def creat_dir(imagepath,newimgpath):
    if not os.path.exists(newimgpath):
        os.mkdir(newimgpath)
        print('创建图像增强path')
    filelist = os.listdir(imagepath)
    return filelist


def main_image_aug(imagepath, newimgpath):
    index = 0
    filelist = creat_dir(imagepath, newimgpath)
    for i in filelist:
        # if i.startswith('gen'):
        tmplist = os.listdir(os.path.join(imagepath, i))
        tmpimg = os.path.join(imagepath, i)
        for v, j in enumerate(tmplist):
            print('开始处理第{}个文件夹的第{}张图片。。。'.format(index + 1, v + 1))
            #  print(type(j))
            if j.endswith('.jpg') or j.endswith('.JPG') or j.endswith('.jpeg'):
                img = cv2.imread(os.path.join(tmpimg, j))
                img = img_augmentation(img)
                # new_addre=os.path.join(newimgpath, i+'_new')  ## 创建存储新的文件夹
                new_addre = os.path.join(newimgpath, i)
                if not os.path.exists(new_addre):
                    os.mkdir(new_addre)
                add_new = 'new' + j  ## 创建新的存储图片地址路径
                dst = os.path.join(new_addre, add_new)
                cv2.imwrite(dst, img)
            else:
                src = os.path.join(tmpimg, j)
                # dst=os.path.join(newimgpath, i+'_new')
                dst = os.path.join(newimgpath, i)
                if not os.path.exists(dst):
                    os.mkdir(dst)
                new_addre = os.path.join(dst, j)
                shutil.copyfile(src, new_addre)
        index += 1
        print(index)
        print('已处理完成{}个文件夹。。。'.format(index))



if __name__ == "__main__":
    start_time = time.time()
    # imagepath = "../test/test1/"
    # newimgpath = "../test/test3/"
    imagepath = "/home/data/data2020/tmp_xinzeng/"
    newimgpath = "/home/data/data2020/tmp_xinzeng/"
    main_image_aug(imagepath, newimgpath)
    end_time = time.time()
    cost_time = (end_time - start_time) / 60
    print('cost time:{}min'.format(cost_time))
