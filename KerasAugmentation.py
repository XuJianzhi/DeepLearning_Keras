# 参考：http://blog.csdn.net/weiwei9363/article/details/78635674

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np


'''
    featurewise_center：布尔值，使输入数据集去中心化（均值为0）, 按feature执行
    samplewise_center：布尔值，使输入数据的每个样本均值为0
    featurewise_std_normalization：布尔值，将输入除以数据集的标准差以完成标准化, 按feature执行
    samplewise_std_normalization：布尔值，将输入的每个样本除以其自身的标准差
    zca_whitening：布尔值，对输入数据施加ZCA白化
    zca_epsilon: ZCA使用的eposilon，默认1e-6
    rotation_range：整数，数据提升时图片随机转动的角度
    width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    shear_range：浮点数，剪切强度（逆时针方向的剪切变换角度）
    zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    channel_shift_range：浮点数，随机通道偏移的幅度
    fill_mode：；‘constant’，‘nearest’，‘reflect’或‘wrap’之一，当进行变换时超出边界的点将根据本参数给定的方法进行处理
    cval：浮点数或整数，当fill_mode=constant时，指定要向超出边界的点填充的值
    horizontal_flip：布尔值，进行随机水平翻转
    vertical_flip：布尔值，进行随机竖直翻转
    rescale: 重放缩因子,默认为None. 如果为None或0则不进行放缩,否则会将该数值乘到数据上(在应用其他变换之前)
    preprocessing_function: 将被应用于每个输入的函数。该函数将在任何其他修改之前运行。该函数接受一个参数，为一张图片（秩为3的numpy array），并且输出一个具有相同shape的numpy array
    data_format：字符串，“channel_first”或“channel_last”之一，代表图像的通道维的位置。该参数是Keras 1.x中的image_dim_ordering，“channel_last”对应原本的“tf”，“channel_first”对应原本的“th”。以128x128的RGB图像为例，“channel_first”应将数据组织为（3,128,128），而“channel_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channel_last”
'''

# 指定参数
# rotation_range 旋转
# width_shift_range 左右平移
# height_shift_range 上下平移
# zoom_range 随机放大或缩小
img_generator = ImageDataGenerator(
    rotation_range = 45,		#旋转
    width_shift_range = 0.3,	#左右平移，比例
    height_shift_range = 0.3,	#上下平移，比例
    zoom_range = 0.2,			#横纵拉伸，比例
    shear_range = 0.2,			#斜拉，含水平翻转
    #featurewise_center = True,	#没变化
    #zca_whitening = True,		#没变化
    #zca_epsilon = 0.1，			#没变化
    #channel_shift_range = 0.2,	#没变化
    horizontal_flip = True,		#水平翻转
    #rescale = 1.2,				#部分色彩变暗(<1.0)或变亮(>1.0)
    )
    
# 导入并显示图片
img_path = '/home/m/桌面/333/1.jpg'
img = image.load_img(img_path)
#plt.imshow(img) 
#plt.show()

# 将图片转为数组
x = image.img_to_array(img)
# 扩充一个维度
x = np.expand_dims(x, axis=0)
# 生成图片
gen = img_generator.flow(x, batch_size=1)

# 显示生成的图片
plt.figure()
for i in range(3):
    for j in range(3):
        x_batch = next(gen)
        idx = (3*i) + j
        plt.subplot(3, 3, idx+1)
        plt.imshow(x_batch[0]/256)	#plt.imshow()的参数是一个三维array
x_batch.shape
plt.show()


####################################
#上面是对于一张图的，下面是对于多张图的
####################################

from keras.datasets import cifar10
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

def preprocess_data(x):
    x /= 255
    x -= 0.5
    x *= 2
    return x
    
# 预处理
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train = preprocess_data(x_train)
x_test = preprocess_data(x_test)

# one-hot encoding
n_classes = 10
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)

# 取 20% 的训练数据
x_train_part = x_train[:10000]
y_train_part = y_train[:10000]

print(x_train_part.shape, y_train_part.shape)

# 建立一个简单的卷积神经网络
def build_model():
    model = Sequential()

    model.add(Conv2D(64, (3,3), input_shape=(32,32,3)))
    model.add(Activation('relu'))
    model.add(BatchNormalization(scale=False, center=False))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(scale=False, center=False))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model

# 训练参数
batch_size = 128
epochs = 20    
    
    
model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_part, y_train_part, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.1)

##############################################
#上面是加augmentation之前，下面是加之后
##############################################

from keras.utils import generic_utils

# 设置生成参数
img_generator = ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    zoom_range = 0.2
    )

model_2 = build_model()
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation后，数据变多了，因此我们需要更的训练次数
for e in range(epochs*4):
    print('Epoch', e)
    print('Training...')
    progbar = generic_utils.Progbar(x_train_part.shape[0])
    batches = 0

    for x_batch, y_batch in img_generator.flow(x_train_part, y_train_part, batch_size=batch_size, shuffle=True):
        loss,train_acc = model_2.train_on_batch(x_batch, y_batch)
        batches += x_batch.shape[0]
        if batches > x_train_part.shape[0]:
            break		# generator是永远循环的，所以得手动跳出
        progbar.add(x_batch.shape[0], values=[('train loss', loss),('train acc', train_acc)])
        
loss, acc = model_2.evaluate(x_test, y_test, batch_size=32)
print('Loss: ', loss)
print('Accuracy: ', acc)

























