import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

path = './model/model_best.hdf5'
test_img1 = './test_images/1571230395241.jpg'  ##笔记本电脑

img = load_img(test_img1, target_size=(224, 224))  # this is a PIL image
x = img_to_array(img)  ## 将图片转换为数组
image = np.expand_dims(x, axis=0)  # 扩充一个维度
image = image.astype('float32')
image -= 127.5
image /= 127.5

with open('./model/dict_class.txt', 'r') as file:
    a = file.read()
    dict_class = eval(a)
dict_class_name = {}
for k, v in dict_class.items():
    dict_class_name[v] = k

model2 = load_model(path)
result = model2.predict(image, batch_size=1)
result1 = np.argmax(result, axis=1)

dict1 = {}
with open('./labels_ch.txt', 'r', encoding='utf-8') as file:
    lst = file.readlines()
for i in lst:
    dict1[i.split(',', 1)[0]] = i.split(',', 1)[1].strip('\n')

# result = model2.predict_generator(test_generator)
print('预测值结果列表：', result)
print('预测结果的类别索引值：', result1)
print('预测结果最大概率值：', result[0][result1[0]])
print('finally predict class name encoding:', dict_class_name[result1[0]])
print('最终预测的类别的中文名：', dict1[dict_class_name[result1[0]]])
# print(model2.evaluate_generator(test_generator))
# print('Predicted:', decode_predictions(result, top=3)[0])
