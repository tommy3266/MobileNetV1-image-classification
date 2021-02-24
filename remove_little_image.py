import os


def del_small_file(file_name):
    size = os.path.getsize(file_name)
    file_size = 2 * 1024
    if size < file_size:
        print('remove', size, file_name)
        os.remove(file_name)


def main_del_small_iamge(path):
    for (root, dirs, files) in os.walk(path):
        for dirc in dirs:
            print('正在扫描检查{}文件夹。。。'.format(dirc))
            pic_path = os.path.join(root, dirc)
            for file in os.listdir(pic_path):
                file = pic_path + '/' + file
                del_small_file(file)


if __name__ == '__main__':
    # path = '../test/test1/'
    path = "/home/data/data2020/tmp_xinzeng/"
    main_del_small_iamge(path)
