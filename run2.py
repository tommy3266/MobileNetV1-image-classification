#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
from data_augment import main_data_aug
from image_aug_threading import main_image_aug
from remove_little_image import main_del_small_iamge
from images_split_train_val import split_ori_datasets
from transfer_learning import create_dir, train_model
from check_image import check_bad_image


def main(is_mobilenetV2):
    aug_factor = 1  ## 数据增强因子，最终图片生成的数量倍数是==线程数*aug_factor
    imagepath = '../data2020/new_image_data2020/'  ## 原始数据图像文件夹
    newimgpath = '../data2020/new_image_data2020_aug2/'  ## 数据增强至文件夹
    target_path = '../data2020/new_image_data2020_aug_split2/'  ## 训练集和验证集划分的目标文件夹
    train_image_path = '../data2020/new_image_data2020_aug_split2/train/' ## 训练集
    val_image_path = '../data2020/new_image_data2020_aug_split2/val/'     ## 验证集
    test_path = '../data2020/new_image_data2020_aug_split2/test/'         ## 测试集（暂不启用）
    bad_image_path = './tmp_bad_images2/'

    def aug_split():
        main_data_aug(imagepath, newimgpath, aug_factor)  ## 数据图像增强==aug_factor * 开启的线程数
        main_image_aug(newimgpath, newimgpath)  ## 数据图像增强一倍
        main_del_small_iamge(newimgpath)  ## 删除size小于2k的图像
        check_bad_image(newimgpath, bad_image_path)  ## 将无法读取的图像移走
        split_ori_datasets(target_path, newimgpath, train_image_path, val_image_path, test_path)  ## 训练集、验证集划分，8:2

    # aug_split()

    num = len(os.listdir(train_image_path))
    alpha = 0.25
    class_num = num
    epoch = 10
    train_batch_size = 16
    val_batch_size = 64
    root_model_path = './model_other1208/'
    last_model_path = './js_model_other1208'
    best_model_path = './js_model_best_other1208'
    filepath = os.path.join(root_model_path, 'model_best.hdf5') ## val_acc最高的model
    filepath1 = os.path.join(root_model_path, 'model_best_weights.hdf5')  ## val_loss最小的model（启用）

    dict_class_save_path = os.path.join(root_model_path,'dict_class.txt')
    model_path = os.path.join(root_model_path,'keras_model_')
    history_path = os.path.join(root_model_path, 'history.txt')

    list_path = [root_model_path, last_model_path, best_model_path]
    create_dir(list_path)

    train_model(alpha,
                class_num,
                train_image_path,
                val_image_path,
                train_batch_size,
                val_batch_size,
                dict_class_save_path,
                epoch,
                model_path,
                last_model_path,
                best_model_path,
                filepath,
                filepath1,
                history_path,
                is_mobilenetV2=is_mobilenetV2)


if __name__ == '__main__':
    main(is_mobilenetV2=False)





