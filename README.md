# DOLG-Retrieval-of-Landmark
  本代码为厦门大学人工智能系2018级本科生的毕业论文《面向互联网图像的地标识别与检索技术研究》中DOLG模型的实现，作者CXX，邮箱：1074593699@qq.com。  
  代码说明文档将分为六个部分：环境配置，数据准备，训练模型，提取索引集描述符，评价模型以及识别与检索。
## 文件说明
```angular2html
|-- data                  存放训练集、测试集以及索引集
  |-- train.zip             台中市地标数据集——训练集
  |-- test.zip              台中市地标数据集——测试集
  |-- index.zip             台中市地标数据集——索引集
  |-- train.csv             存放训练集的图像名称和对应地标编号
  |-- landmark.txt          地标编号对应的地标名称
|-- dataset               数据集处理相关代码
  |-- dataset.py
  |-- transform.py
|-- lightning_logs        模型保存
  |-- version_1
    |-- checkpoints
      |-- epoch=44-step=11250.ckpt  DOLG模型-44轮
    |-- hparams.yaml
|-- model                 DOLG模型
  |-- arcface.py            损失率
  |-- dolg.py               模型主体
  |-- gem_pool.py           GeM Pooling
|-- config.py             模型配置参数
|-- train.py              训练代码
|-- cal_mAP_testset.py    测试代码：计算测试集在索引集上的mAP和mAP@10
|-- Testset_mAP_log.txt   测试过程信息
|-- Recognition.py        地标识别功能
|-- Retrieval_index.py    提取索引集图像描述符
|-- index_feature_DOLG.h5 索引集所有的图像描述符集合
|-- Retrieval.py          地标检索功能
|-- Retrieval_Results.txt 地标检索结果
|-- requirements.txt	  Python环境配置
|-- README.md		  说明文档
```
## 环境配置
### 服务器配置
- CPU：Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz
- GPU：NVIDIA A100-PCIE-40GB x 2
- 内存：512GB
- 显存：40GB x 2
- 操作系统：CentOS Linux release 7.9.2009 (Core) 64 位
### Python环境配置
+ PyTorch
+ PyTorch Lightning
+ timm
+ sklearn
+ pandas
+ jpeg4py
+ albumentations
+ python3
+ CUDA
+ numpy
+ os
+ h5py
+ argparse
### 更加简便地安装环境
	$ pip install -r requirements.txt
## 数据准备
对于Google Landmarks Dataset v2数据集，从该链接获得：https://github.com/cvdfoundation/google-landmark  
对于GLDv2-clean Dataset数据集：[kaggle competition dataset](https://www.kaggle.com/c/landmark-retrieval-2021).  
对于台中市地标数据集：已经在```./data```目录下，train.zip、test.zip以及index.zip分别表示训练集（50类地标，2500张图像）、测试集（100张图像）以及索引集（10000张图像）

训练集的格式应当按照如下格式存放：  
```
data
├── train_clean.csv
└── train
    └── ###
        └── ###
            └── ###
                └── ###.jpg
```
其中，train_clean.csv中应当标明训练的图片文件名称以及对应的类别：
```
id	landmark_id
###	0
###	12
###	3
...	...
```  
测试集和索引集图片的存取格式如下：
```
data
├── index
    └── {label_id}###.jpg
		└── ...
└── test
    └── {label_id}###.jpg
		└── ...
```  
其中，测试集和索引集目录下的图片名称的前两位，应是所属地标图像的编号，  
e.g. 00123.jpg和12888.jpg分别表示第1类和第13类地标。
## 训练模型
模型超参数配置可通过```./data```目录下的```config.py```文件进行修改，配置完成后，在目录下使用如下命令进行训练：  
	```
	$ python train.py
	```  
模型保存路径：  
	```
	./data/lightning_logs/version_#/checkpoints/*.ckpt
	```  
本文模型的测试以及识别与检索功能实现皆是在```./data/lightning_logs/version_1/checkpoints/epoch=44-step=11250.ckpt```模型上实现，如要读取另外的保存模型，请自行更改相应代码
## 提取索引集图像描述符
对于台中市地标数据集的索引集使用如下代码提取图像描述符：  
	```
	$ python Retrieval_index.py -database ./data/index -index index_feature_DOLG.h5
	```  
运行完后将会在根目录下生成	```./index_feature_DOLG.h5```文件，包含索引集所有地标图像的图像描述符。  
```Retrieval_index.py```的参数说明如下：  
1. ```-database {DIR_PATH}```, e.g.```-database ./data/index```，表示索引集文件夹路径
2. ```-index {FILENAME}.h5```, e.g.```-index index_feature_DOLG.h5``` 表示索引集所有图像的图像描述符集合的存储路径
## 评价模型
对台中市地标数据集的测试集进行测试，并计算mAP以及mAP@10指标：  
	```
	$ python cal_mAP_testset.py -testdir ./data/test -index index_feature_DOLG.h5
	```  
运行完成后，在命令行或者根目录下的```Testset_mAP_log.txt```查看测试结果。  
```cal_mAP_testset.py```的参数说明如下：  
1. ```-testdir {DIR_PATH}```, e.g.```-testdir ./data/test```，表示测试集文件夹路径
2. ```-index {FILENAME}.h5```, e.g.```-index index_feature_DOLG.h5``` 表示索引集所有图像的图像描述符集合的存储路径
## 识别与检索
### 识别功能
如要对特定的某一图像进行识别，判断其属于哪类地标，请使用如下代码  
	```
	$ python Recognition.py {FILENAME}.jpg
	```  
e.g.  
	```
	$ python Recognition.py ./data/test/00093.jpg
	```  
在命令行中查看识别结果
### 检索功能
在台中市地标数据集的索引集下对某一特定图像进行检索，并输出最有可能的10个图像路径：  
	```
	python Retrieval.py -query ./data/test/00093.jpg -index index_feature_DOLG.h5 -result ./data/index
	```  
运行完成后，在命令行或者根目录下的```Retrieval_Results.txt```查看测试结果。  
```Retrieval.py```的参数说明如下：  
1. ```-query {FILENAME}.jpg```, e.g.```-query ./data/test/000931.jpg```，表示询问图像路径
2. ```-index {FILENAME}.h5```, e.g.```-index index_feature_DOLG.h5``` 表示索引集所有图像的图像描述符集合的存储路径
3. ```-result {DIR_PATH}```, e.g.```-result ./data/index``` 表示索引集文件夹路径
## 联系作者
如对本文代码有何建议或疑问，可以联系:1074593699@qq.com
