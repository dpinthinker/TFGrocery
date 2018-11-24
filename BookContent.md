# TensorFlow Lite Android部署介绍

## 1、简要介绍

TensorFlow Lite是TensorFlow在移动和嵌入式设备上的轻量级解决方案，目前只能用于预测，还不能进行训练。TensorFLow Lite针对移动和嵌入设备开发，具有如下三个特点：

* 轻量
* 跨平台
* 快速

目前TensorFlow Lite已经支持Android、iOS、Raspberry等设备，本章会基于Android设备上的部署方法进行讲解，内容包括模型保存、转换和部署。

## 2、模型训练和保存

我们以keras模型训练和保存为例进行讲解，如下是keras官方的mnist模型训练样例。

```python
'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

创建mnist_cnn.py文件，将以上内容拷贝进去，并在最后加上如下一行代码：

```python
model.save('mnist_cnn.h5')
```

在终端中执行mnist_cnn.py文件，如下：

```python
python mnist_cnn.py
```

*注：该过程需要连接网络获取mnist.npz文件（https://s3.amazonaws.com/img-datasets/mnist.npz），会被保存到$HOME/.keras/datasets/。如果网络连接存在问题，可以通过其他方式获取mnist.npz后，直接保存到该目录即可。*

执行过程会比较久，执行结束后，会产生在当前目录产生`mnist_cnn.h5`文件（HDF5格式），就是keras训练后模型，其中已经包含了训练后的模型结构和权重等信息。

该模型可以在服务器端，可以直接通过keras.models.load_model("mnist_cnn.h5")加载，然后进行推测；在移动设备需要将HDF5模型文件转换为TensorFlow Lite的格式，然后提供相应平台提供的Interpreter加载，然后进行推测。

## 3、模型转换

不能直接在移动端部署，因为模型大小和运行效率比较低，最终需要通过工具转化为Flat Buffer格式的模型。

谷歌提供了多种转换方式：

* tflight_convert：>= TensorFlow 1.9，本次讲这个
* TOCO：>= TensorFlow 1.7
* 通过代码转换

tflight_convert跟tensorflow是一起下载的，笔者通过brew安装的python，pip安装tf-nightly后tflight_convert路径如下：

```shell
/usr/local/opt/python/Frameworks/Python.framework/Versions/3.6/bin
```

实际上，应该是/usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/bin，但是软链接到了如上路径。如果命令行不能执行到tflight_convert，则在~/.bash_profile（macOS）或~/.bashrc（Linux）添加如下环境变量：

```shell
 export PATH="/usr/local/opt/python/Frameworks/Python.framework/Versions/3.6/bin:$PATH"      
```

然后执行

```shell
source ~/.bash_profile
```

或

```shell
source ~/.bashrc
```

在命令执行

```shell
tflight_convert -h
```

输出结果如下，则证明安装配置成功。

```shell
usage: tflite_convert [-h] --output_file OUTPUT_FILE
                      (--graph_def_file GRAPH_DEF_FILE | --saved_model_dir SAVED_MODEL_DIR | --keras_model_file KERAS_MODEL_FILE)
                      [--output_format {TFLITE,GRAPHVIZ_DOT}]
                      [--inference_type {FLOAT,QUANTIZED_UINT8}]
                      [--inference_input_type {FLOAT,QUANTIZED_UINT8}]
                      [--input_arrays INPUT_ARRAYS]
```

下面我们开始转换模型，具体命令如下：

```shell
tflite_convert --keras_model_file=./mnist_cnn.h5 --output_file=./mnist_cnn.tflite
```

到此，我们已经得到一个可以运行的TensorFlow Lite模型了，即`mnist_cnn.tflite`。

*注：这里只介绍了keras HDF5格式模型的转换，其他模型转换建议参考：https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/tflite_convert/cmdline_examples.md*

## 4、Android部署

现在开始在Android环境部署，对于国内的读者，需要先给Android Studio配置proxy，因为gradle编译环境需要获取相应的资源，请大家自行解决，这里不再赘述。

### 4.1 配置app/build.gradle

新建一个Android Project，打开app/build.gradle添加如下信息

```groovy
android {
    aaptOptions {
        noCompress "tflite"
    }
}

repositories {
    maven {
        url 'https://google.bintray.com/tensorflow'
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite:1.10.0'
}
```

其中，

1、aaptOptions设置tflite文件不压缩，确保后面tflite文件可以被Interpreter正确加载。

2、org.tensorflow:tensorflow-lite的最新版本号，可以在这里查询https://bintray.com/google/tensorflow/tensorflow-lite，目前最新的是1.10.0版本。

设置好后，sync和build整个工程，如果build成功说明，配置成功。

### 4.2 添加tflite文件到assets文件夹

在app目录先新建assets目录，并将`mnist_cnn.tflite`文件保存到assets目录。重新编译apk，检查新编译出来的apk的assets文件夹是否有`mnist_cnn.tflite`文件。

使用apk analyzer查看新编译出来的apk，存在如下目录即编译打包成功。

```shell
assets
   |__mnist_cnn.tflite
```

### 4.3 加载模型

使用如下函数将`mnist_cnn.tflite`文件加载到memory-map中，作为Interpreter实例化的输入。

```java
private static final String MODEL_PATH = "mnist_cnn.tflite";
    
/** Memory-map the model file in Assets. */
private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
}
```

实例化Interpreter，其中this为当前acitivity

```java
tflite = new Interpreter(loadModelFile(this));
```

### 4.4 运行输入

我们使用mnist test测试集中的某张图片作为输入，mnist图像大小28*28，单像素。这样我们输入的数据需要设置成如下格式。

```java
/** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
private ByteBuffer imgData = null;

private static final int DIM_BATCH_SIZE = 1;
private static final int DIM_PIXEL_SIZE = 1;

private static final int DIM_IMG_WIDTH = 28;
private static final int DIM_IMG_HEIGHT = 28;

protected void onCreate() {
    imgData = ByteBuffer.allocateDirect(
        4 * DIM_BATCH_SIZE * DIM_IMG_WIDTH * DIM_IMG_HEIGHT * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
}
```

将mnist图片转化成ByteBuffer，并保持到imgData中。

```java
/** Preallocated buffers for storing image data in. */
private int[] intValues = new int[DIM_IMG_WIDTH * DIM_IMG_HEIGHT];

/** Writes Image data into a {@code ByteBuffer}. */
private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
        return;
    }

    // Rewinds this buffer. The position is set to zero and the mark is discarded.
    imgData.rewind();
    
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    for (int i = 0; i < DIM_IMG_WIDTH; ++i) {
        for (int j = 0; j < DIM_IMG_HEIGHT; ++j) {
            final int val = intValues[pixel++];
            imgData.putFloat(val);
        }
    }
}
```

convertBitmapToByteBuffer的输出即为模型运行的输入。

### 4.5 运行输出

定义一个1*10的多维数组，因为我们只有1个batch和10个label（TODO：need double check），具体代码如下：

```java
private float[][] labelProbArray = new float[1][10];
```

运行结束后，每个二级元素都是一个label的概率。

### 4.6 运行及结果处理

开始运行模型，具体代码如下：

```java
tflite.run(imgData, labelProbArray);
```

针对某个图片，运行后labelProbArray的内容如下，也就是各个label识别的概率。

```java
index 0 prob is 0.0
index 1 prob is 0.0
index 2 prob is 0.0
index 3 prob is 1.0
index 4 prob is 0.0
index 6 prob is 0.0
index 7 prob is 0.0
index 8 prob is 0.0
index 9 prob is 0.0
```

接下来，我们要做的就是根据对这些概率进行排序，找出Top的label并界面呈现给用户

## 5、总结

至此，整个TensorFlow Lite的部署就完成了，包含四个阶段：

1. 模型训练和保存：我们使用的是keras Squential类的save函数
2. 模型转换：我们使用的tflite_convert工具
3. Android部署：配置build.gradle和assets，通过memory-map加载图片并转化为ByteBuffer作为输入和固定维数的float数组作为输出，最后调用Interpreter.run()
4. 处理和显示运行结果

## 6、附录

1. [TF Lite Command-line tools](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/convert/cmdline_examples.md)
2. [TF Lite Android App](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/java/demo/README.md)
3. [Google TF Lite Codelab](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets-2-tflite/#2)
4. [TensorFlow Lite Example](https://github.com/nex3z/tflite-mnist-android)
5. [What I know about TensorFlow Lite](https://www.slideshare.net/kstan2/introduction-to-tensorflow-lite)
6. [TensorFlow Lite for mobile developers (Google I/O '18)](https://www.youtube.com/watch?v=ByJnpbDd-zc&t=1719s)
