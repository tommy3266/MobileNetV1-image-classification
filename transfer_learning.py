#! /usr/bin/python3
# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import os
import subprocess
import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
import tensorflowjs as tfjs
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

start_time = time.time()


def train_model(alpha,
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
                is_mobilenetV2):

    if is_mobilenetV2:
        base_model = MobileNetV2(input_shape=[128, 128, 3],
                                 alpha=0.35,
                                 weights='imagenet',
                                 input_tensor=None,
                                 pooling=None,
                                 classes=None,  ## note:当include_top=False，且weights='imagenet'的时候要把classes=None；
                                 include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
    else:
        base_model = MobileNet(input_shape=[128, 128, 3],
                               alpha=alpha,
                               depth_multiplier=1,
                               dropout=0.5,
                               weights='imagenet',
                               input_tensor=None,
                               pooling=None,
                               classes=None,  ## note:当include_top=False，且weights='imagenet'的时候要把classes=None；
                               include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    preds = Dense(class_num, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)

    for layer in model.layers[:20]:
        layer.trainable = True

    for layer in model.layers[20:]:
        layer.trainable = True

    # 下面是训练imagenet的训练集和验证集
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                       horizontal_flip=True,
                                       vertical_flip=True,
                                       # featurewise_center=True,
                                       # samplewise_center=True,
                                       # featurewise_std_normalization=True,
                                       # samplewise_std_normalization=True,
                                       # zca_whitening=False,
                                       # zca_epsilon=1e-06,
                                       rotation_range=30,
                                       # width_shift_range=2.0,
                                       # height_shift_range=2.0,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       channel_shift_range=0.0,
                                       fill_mode='nearest',
                                       cval=0.0,
                                       # rescale=1/255,
                                       data_format=None,
                                       # validation_split=0.0,
                                       # dtype=None
                                       )  # included in our dependencies

    train_generator = train_datagen.flow_from_directory(train_image_path,
                                                        # this is where you specify the path to the main data folder
                                                        target_size=(128, 128),
                                                        color_mode='rgb',
                                                        batch_size=train_batch_size,
                                                        class_mode='categorical',
                                                        shuffle=True)

    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     # featurewise_center=True,
                                     # samplewise_center=True,
                                     # featurewise_std_normalization=True,
                                     # samplewise_std_normalization=True,
                                     # zca_whitening=False,
                                     # zca_epsilon=1e-06,
                                     rotation_range=30,
                                     # width_shift_range=2.0,
                                     # height_shift_range=2.0,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     channel_shift_range=0.0,
                                     fill_mode='nearest',
                                     cval=0.0,
                                     # rescale=1/255,
                                     data_format=None,
                                     # validation_split=0.0,
                                     # dtype=None
                                     )  # included in our dependencies

    val_generator = val_datagen.flow_from_directory(val_image_path,
                                                    # this is where you specify the path to the main data folder
                                                    target_size=(128, 128),
                                                    color_mode='rgb',
                                                    batch_size=val_batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

    # train_generator.shuffle
    print(train_generator.shuffle)
    print(train_generator.class_indices)
    print(train_generator.classes)
    dict_class = train_generator.class_indices
    with open(dict_class_save_path, 'w') as file:
        file.write(str(dict_class))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Adam optimizer
    # loss function will be categorical cross entropy
    # evaluation metric will be accuracy

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./tmp/logo_log', histogram_freq=0, write_graph=True,
                                             write_images=True)
    checkpoint_save_best_model = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                 mode='max')
    checkpoint_save_best_model1 = ModelCheckpoint(filepath1, monitor='val_loss', verbose=1, save_best_only=True,
                                                  mode='min',
                                                  period=1)
    plot_loss_callback = keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=1, mode='min',
                                                  min_delta=0.01, cooldown=1, min_lr=0.00000001)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.03, patience=3, verbose=1, mode='min',
                                               baseline=None, restore_best_weights=False)
    # callbacks_list = [checkpoint_save_best_model,plot_loss_callback,tbCallBack,reduce_lr]
    callbacks_list = [checkpoint_save_best_model, early_stop, reduce_lr, checkpoint_save_best_model1]

    step_size_train = train_generator.n // train_generator.batch_size

    if os.path.exists(filepath1):
        model.load_weights(filepath1, by_name=True)
        # 若成功加载前面保存的参数，输出下列信息
        print("checkpoint_loaded")
    for epo in range(1, epoch+1):
        print('now is the epoch:', epo)
        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=step_size_train,
                                      ##steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
                                      validation_data=val_generator,
                                      validation_steps=54,
                                      ##validation_steps = TotalvalidationSamples / ValidationBatchSize
                                      callbacks=callbacks_list,
                                      # class_weight={0: 1, 1: 1, 2: 2, 3: 1},                      # class_weight=None,
                                      #  use_multiprocessing=True,
                                      workers=24,
                                      shuffle=True,
                                      epochs=1)

        model.save(model_path + str(epo) + '.h5')
        with open(history_path, 'a+') as file:
            file.write(str(history.history))
            file.write('\n')
    tfjs.converters.save_keras_model(model, last_model_path)
    try:
        model_best = load_model(filepath1)
        tfjs.converters.save_keras_model(model_best, best_model_path)
    except:
        print('val_loss最小的模型tfjs转换失败。')
        end_time = time.time()
        cost_time = (end_time - start_time) / 60
        print('train model cost time:{}min!'.format(cost_time))


def create_dir(list_path):
    for path in list_path:
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)


if __name__ == '__main__':
    alpha = 0.25
    class_num = 49
    epoch = 3
    filepath = './model/model_best.hdf5'
    filepath1 = './model/model_best_weights.hdf5'
    train_image_path = '../image_data/train_aug5/'
    val_image_path = '../image_data/val_aug5/'
    train_batch_size = 16
    val_batch_size = 64
    dict_class_save_path = './model/dict_class.txt'
    model_path = './model/keras_model_'
    history_path = './model/history.txt'
    last_model_path = './js_model'
    best_model_path = './js_model_best'

    list_path = ['./model', last_model_path, best_model_path]
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
                is_mobilenetV2=False)





