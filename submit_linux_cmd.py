#!/usr/bin/python3
# -*- coding:utf-8 -*-

import subprocess

'''
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/data_augment.py > /home/data/AI_discovering2020/log1128_data_augment.log 2>&1 &

nohup /usr/bin/python3 -u /home/data/AI_discovering2020/image_aug_threading.py > /home/data/AI_discovering2020/log1128_image_aug.log 2>&1 &

1、原始图像增强之后执行，暂未增加other分类，使用mobilenetV1：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log1128V1.log 2>&1 &

2、原始图像增强之后执行，暂未增加other分类，使用mobilenetV2：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run.py > /home/data/AI_discovering2020/log1128V2.log 2>&1 &

3、原始图像增强之后执行，增加other分类，使用mobilenetV1：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log_other_1128V1.log 2>&1 &

4、原始图像增强之后执行，增加other分类，使用mobilenetV2：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run.py > /home/data/AI_discovering2020/log_other1128V2.log 2>&1 &

5、原始图像增强之后执行，增加other分类，使用alpha=0.5：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log_other1129V1_0.5.log 2>&1 &

6、2020.12.3日删除奥妙洗洁精其他类中的大部分图像（与植澈重复）重新增强，删除全部的花木星球其他、金纺其他、金纺洗衣液（准确率很低）,新增旁氏洁面乳黄新和中华牙膏：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log_other_1203V1.log 2>&1 &

7、2020.12.4日,重拍多芬洗发乳、奥妙高效洗洁精、清扬洗发露、清扬沐浴露、植澈消毒液类别的照片并数据增强，重新训练，epoch=10：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log_other_1204V1.log 2>&1 &

8、2020.12.7日,删除奥妙高效洗洁精、清扬洗发露中部分模糊和侧面照片并重新数据增强加入训练验证集，以1204版的model_best_weights.hdf5模型为基础重新训练，epoch=4：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log_other_1207V1.log 2>&1 &

8、2020.12.8日,重拍奥妙高效洗洁精、清扬洗发露图片，着重俯视角并重新数据增强加入原有训练验证集，重新训练，epoch=10：
nohup /usr/bin/python3 -u /home/data/AI_discovering2020/run2.py > /home/data/AI_discovering2020/log_other_1208V1.log 2>&1 &


work：tensorflowjs==2.7.0  将多个分片模型文件合并，减少group.bin文件的个数，从tfjs_layers_model to tfjs_layers_model
weight_shard_size_bytes 4194304 限制为4M
tensorflowjs_converter --input_format tfjs_layers_model --output_format tfjs_layers_model js_model5_2_1225/model.json split_model/ --weight_shard_size_bytes 4194304

直接从h5模型转换为tfjs_layers_model模型：
tensorflowjs_converter --input_format keras --output_format tfjs_layers_model model_other/model_best_weights.hdf5 js_model_best_other_split/ --weight_shard_size_bytes 4194304

直接从h5模型转换为tfjs_graph_model模型：
tensorflowjs_converter --input_format keras --output_format tfjs_graph_model model_other/model_best_weights.hdf5 tfjs_graph_model/ --weight_shard_size_bytes 4194304

cp -r Mineral_water aomiaoxijiejing_other aomiaoxiyiye_other bags blueair_other bonsai books booth chair computer counter cup desk duofen_other fanshilin_other floor flower huamuxingqiu_other huayangxingqiu_other jinfang_other keyboard lishimuyulu_other lishixifalu_other Mineral water pangshi_other people qingyang_other shoes tile trousers truliva_other zhonghuayagao_other ../tmp/

mv aomiaochufang aomiaoguoshuye aomiaoxiyiye aomiaoxijiejing duofenjiemianpaopao duofenmuyuru duofenxifaru fanshilin_3 fanshilin_4 fanshilin_5 fanshilin_6 huayangxingqiu jinfangxiyiye landuoba lishimuyuru ../tmp/

使用ffmpeg结合视频，将图片的背景替换掉（视频当前路径，图片在./tmp路径下）：
ffmpeg -y -stream_loop 9 -i 40a8fada3c33717556d0732af01c28fc.mp4 ./tmp/image-%3d.png
ffmpeg -y -stream_loop 1 -i 1.mp4 ./tmp/image-%3d.png
ffmpeg -y -i tmp1/1.mp4 -i tmp1/img/1.jpg -filter_complex "[1:v]colorkey=0x2e693d:0.3:0[dg];[dg]scale=400:-1[above];[0:v][above]overlay=x=300:y=100" -map tmp3/image-%3d.jpg

'''

## 执行命令，将other分类复制到新的训练数据集

cmd1 = 'cd /home/data/AI_discovering2/train_aug5 && cp -r Mineral_water aomiaoxijiejing_other aomiaoxiyiye_other bags blueair_other bonsai books booth chair computer counter cup desk duofen_other fanshilin_other floor flower huamuxingqiu_other huayangxingqiu_other jinfang_other keyboard lishimuyulu_other lishixifalu_other Mineral water pangshi_other people qingyang_other shoes tile trousers truliva_other zhonghuayagao_other ../../data2020/new_image_data2020_aug_split/train/'
process1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=None, shell=True)
# result = process1.stdout.readlines()

cmd2 = 'cd /home/data/AI_discovering2/val_aug5 && cp -r Mineral_water aomiaoxijiejing_other aomiaoxiyiye_other bags blueair_other bonsai books booth chair computer counter cup desk duofen_other fanshilin_other floor flower huamuxingqiu_other huayangxingqiu_other jinfang_other keyboard lishimuyulu_other lishixifalu_other Mineral water pangshi_other people qingyang_other shoes tile trousers truliva_other zhonghuayagao_other ../../data2020/new_image_data2020_aug_split/val/'
process2 = subprocess.Popen(cmd2, stdout=subprocess.PIPE, stderr=None, shell=True)
# result2 = process1.stdout.readlines()



