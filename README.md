# DOLG-Retrieval-of-Landmark
  本代码为厦门大学人工智能系2018级本科生的毕业论文《面向互联网图像的地标识别与检索技术研究》中DOLG模型的实现，作者CXX，邮箱：1074593699@qq.com。说明文档分为四个部分，文件说明，环境配置，数据准备，训练模型，评价模型以及识别与检索。
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

## 数据准备
## 训练模型
## 评价模型
## 识别与检索
