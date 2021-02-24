#!/usr/bin/python3
# -*- coding:utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import threading
import time

Datagen = ImageDataGenerator(rotation_range=23,
                             shear_range=0.32,
                             zoom_range=0.32,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='nearest',
                             featurewise_center=False,
                             samplewise_center=False,
                             featurewise_std_normalization=False,
                             samplewise_std_normalization=False,
                             zca_whitening=False,
                             zca_epsilon=1e-6,
                             width_shift_range=0.22,
                             height_shift_range=0.22,
                             brightness_range=None,
                             channel_shift_range=0.,
                             cval=0.,
                             rescale=None,
                             # preprocessing_function=preprocess_input,
                             data_format=None,
                             validation_split=0.0,
                             )


# 还有其他一些参数，具体请看：https://keras.io/preprocessing/image/ ，如去均值，标准化，ZCA白化，旋转，
# 偏移，翻转，缩放等


def data_aug(Datagen, imagepath, newimgpath, aug_factor):
    filelist = os.listdir(imagepath)

    index = 0
    for i in filelist:
        tmplist = os.listdir(os.path.join(imagepath, i))
        tmpimg = os.path.join(imagepath, i)
        save_path = os.path.join(newimgpath, i)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for v, j in enumerate(tmplist):
            print('开始处理第{}个文件夹的第{}张图片。。。'.format(index + 1, v + 1))
            #  print(type(j))
            if j.endswith('.jpg') or j.endswith('.JPG'):
                # img=cv2.imread(os.path.join(tmpimg,j))
                path = os.path.join(tmpimg, j)
                print(path)
                img = load_img(path)  # 获取一个PIL图像
                x_img = img_to_array(img)
                x_img = x_img.reshape((1,) + x_img.shape)
                m = 0
                for img_batch in Datagen.flow(x_img, batch_size=1, save_to_dir=save_path, save_prefix='hand',
                                              save_format='jpeg'):
                    m += 1
                    if m >= aug_factor:  ## aug_factor代表的是增强因子
                        break
        index += 1
        print('已处理完成{}个文件夹。。。'.format(index))


def main_data_aug(imagepath,newimgpath,aug_factor):
    if not os.path.exists(newimgpath):
        os.mkdir(newimgpath)
        print('没有该文件夹，创建新文件夹！')
    else:
        pass
    print(f'主线程开始时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')
    # 初始化线程
    t1 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    t2 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    t3 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    t4 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    t5 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    # t6 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    # t7 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    # t8 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    # t9 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))
    # t10 = threading.Thread(target=data_aug, args=(Datagen, imagepath, newimgpath, aug_factor))

    # 开启线程
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    # t6.start()
    # t7.start()
    # t8.start()
    # t9.start()
    # t10.start()

    # 等待运行结束
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    # t6.join()
    # t7.join()
    # t8.join()
    # t9.join()
    # t10.join()

    print(f'主线程结束时间：{time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == "__main__":
    aug_factor = 1  ## 数据增强因子，最终图片生成的数量倍数是==线程数*aug_factor
    # imagepath = '../test/test3/'
    # newimgpath = '../test/test3/'
    imagepath = "/home/data/data2020/tmp_xinzeng/"
    newimgpath = "/home/data/data2020/tmp_xinzeng/"
    main_data_aug(imagepath, newimgpath, aug_factor)

