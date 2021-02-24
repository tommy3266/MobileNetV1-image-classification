## 将保存的h5模型加载并转为tensorflowjs格式（此时的h5模型包含模型的结构、权重、参数配置）
import tensorflowjs as tfjs
from keras.models import load_model
filepath = './model5_2_1225/model_best.hdf5'
model = load_model(filepath,compile=False)
tfjs.converters.save_keras_model(model, './js_model_best5_2_1225')


