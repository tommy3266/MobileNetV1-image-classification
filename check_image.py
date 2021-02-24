import os
import cv2
import shutil


def check_bad_image(dir,path):
    bad_list = []

    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

    subdir_list = os.listdir(dir)
    for i, d in enumerate(subdir_list):
        print('check 第{}个文件夹。。。'.format(i+1))
        dpath = os.path.join(dir, d)
        class_list = os.listdir(dpath)
        for klass in class_list:
            f_path = os.path.join(dpath, klass)
            try:
                img = cv2.imread(f_path)
                size = img.shape
            except:
                print('file {} is not a valid image file '.format(f_path))
                bad_list.append(f_path)

    print(bad_list)

    for i in bad_list:
        shutil.move(i, path)


if __name__ == '__main__':
    dir1 = '../image_data/val_aug5/'
    dir2 = '../image_data/train_aug5/'
    path = './tmp_bad_images/'
    check_bad_image(dir1, path)
    check_bad_image(dir2, path)

