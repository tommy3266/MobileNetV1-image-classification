import os


def rename_file(path, key_word='new'):
    for dir in os.listdir(path):
        path2 = os.path.join(path, dir)
        print(path2)
        for file in os.listdir(path2):
            ori_file_path = os.path.join(path2, file)
            print(ori_file_path)
            ori_name = file.split('.')[0]
            ext = file.split('.')[1]
            newname = str(ori_name) + str(key_word) + str('.') + str(ext)
            print(newname)
            new_name_path = os.path.join(path2, newname)
            os.rename(ori_file_path, new_name_path)
            print('success : ' + newname)


if __name__ == '__main__':
    # path = '/home/data/data2020/tmp_xinzeng2/train'
    # path2 = '/home/data/data2020/tmp_xinzeng2/val'
    # rename_file(path)
    # rename_file(path2)
    path = '/home/data/data2020/tmp_xinzeng/'
    rename_file(path)
