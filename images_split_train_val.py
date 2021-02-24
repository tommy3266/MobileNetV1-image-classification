import os
import random
from shutil import copy2

"""将原始图片集，按照比例划分为训练集、验证集、测试集"""


def getDir(filepath):
    pathlist = os.listdir(filepath)
    return pathlist


def mkTotalDir(data_path):
    if data_path:
        pass
    else:
        os.makedirs(data_path)
    dic = ['train', 'val', 'test']
    for i in range(0, 3):
        current_path = data_path + dic[i] + '/'
        # 这个函数用来判断当前路径是否存在，如果存在则创建失败，如果不存在则可以成功创建
        isExists = os.path.exists(current_path)
        if not isExists:
            os.makedirs(current_path)
            print('successful ' + dic[i])
        else:
            print('is existed')
    return


def getClassesMes(source_path):
    classes_name_list = getDir(source_path)
    classes_num = len(classes_name_list)
    return classes_name_list, classes_num



def mkClassDir(source_path, change_path):
    classes_name_list, classes_num = getClassesMes(source_path)
    for i in range(0, classes_num):
        current_class_path = os.path.join(change_path, classes_name_list[i])
        isExists = os.path.exists(current_class_path)
        if not isExists:
            os.makedirs(current_class_path)
            print('successful ' + classes_name_list[i])
        else:
            print('is existed')


"""
source_path:原始多类图像的存放路径
train_path:训练集图像的存放路径
validation_path:验证集图像的存放路径
test_path:测试集图像的存放路径
"""


def divideTrainValidationTest(source_path, train_path, validation_path, test_path):
    """先获取n类图像的名称列表和类别数目"""
    classes_name_list, classes_num = getClassesMes(source_path)
    mkClassDir(source_path, train_path)
    mkClassDir(source_path, validation_path)
    mkClassDir(source_path, test_path)

    for i in range(0, classes_num):
        source_image_dir = os.listdir(source_path + classes_name_list[i] + '/')
        random.shuffle(source_image_dir)
        train_image_list = source_image_dir[0:int(0.8 * len(source_image_dir))]
        # validation_image_list = source_image_dir[int(0.7 * len(source_image_dir)):int(0.9 * len(source_image_dir))]
        # test_image_list = source_image_dir[int(0.9 * len(source_image_dir)):]
        validation_image_list = source_image_dir[int(0.8 * len(source_image_dir)):]

        for train_image in train_image_list:
            origins_train_image_path = source_path + classes_name_list[i] + '/' + train_image
            new_train_image_path = train_path + classes_name_list[i] + '/'
            copy2(origins_train_image_path, new_train_image_path)
        for validation_image in validation_image_list:
            origins_validation_image_path = source_path + classes_name_list[i] + '/' + validation_image
            new_validation_image_path = validation_path + classes_name_list[i] + '/'
            copy2(origins_validation_image_path, new_validation_image_path)
        # for test_image in test_image_list:
        #     origins_test_image_path = source_path + classes_name_list[i] + '/' + test_image
        #     new_test_image_path = test_path + classes_name_list[i] + '/'
        #     copy2(origins_test_image_path, new_test_image_path)


def split_ori_datasets(target_path, source_path, train_path, validation_path, test_path):
    mkTotalDir(target_path)
    divideTrainValidationTest(source_path, train_path, validation_path, test_path)



if __name__ == '__main__':
    source_path = "/home/data/data2020/tmp_xinzeng/"
    target_path = "/home/data/data2020/tmp_xinzeng2/"
    train_path = "/home/data/data2020/tmp_xinzeng2/train/"
    validation_path = '/home/data/data2020/tmp_xinzeng2/val/'
    test_path = '/home/data/data2020/tmp_xinzeng2/test/'
    split_ori_datasets(target_path, source_path, train_path, validation_path, test_path)

