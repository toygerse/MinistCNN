
## 基于minist数据集复现CNN通用模型（Lenet-5 & ResNet）
### 1 数据集介绍

  数据来源于 *kaggle* 的 *Digit Recognizer*，分为 *train.csv*，*test.csv*，*sample_submission.csv* 三个文件。具体介绍及下载见  [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer/data?select=sample_submission.csv).
  
### 2 问题说明

- 复现CNN通用模型
- 应用模型于minist数据集  

### 3 代码结构

主要应用上一次minist识别的代码，如下：

```python
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as f
from torch.utils.data import DataLoader,Dataset

dataNum = 0
correctNum = 0

testRoad = "./MNIST_dataset/test.csv"
trainRoad = "./MNIST_dataset/train.csv"
sampleRoad = "./MNIST_dataset/sample_submission.csv"
predictRoad = "./MNIST_dataset/sample_submission_predict.csv"

#读取数据文件，忽略第一行 skiprows=1，返回string列表
def read_csv(fileRoad):
    return np.loadtxt(fileRoad, dtype=str, skiprows=1)

#unsqueeze(0)为了在卷积中给minist扩充为一个通道
class Mydata(Dataset):
    def __init__(self, dir):
        self.datas = read_csv(dir)
        self.rootDir = dir

    def __getitem__(self, index):
        data = [float(i) for i in self.datas[index].split(',')]
        if len(data)==785:
            label = int(data[0])
            sample = torch.tensor(data[1:]).reshape(28,28).unsqueeze(0)
            return sample,label
        elif len(data)==784:
            sample = torch.tensor(data).reshape(28,28).unsqueeze(0)
            return sample

    def __len__(self):
        return len(self.datas)

#定义网络结构
class Net(nn.Module):
    def __init__(self):
    	#定义网络
    def forward(self,x):
    	#定义前向传播
       
#计算准确率
def count_accuracy(pred_y, label):
    global dataNum, correctNum
    for i in range(len(label)):
        dataNum =dataNum +1
        if pred_y[i] == label[i]:
            correctNum = correctNum +1

#测试模型准确率
def test_existModle(myNet):
    trainData = Mydata(trainRoad)
    trainLoader = DataLoader(trainData, batch_size=150, shuffle=True)
    for input,label in trainLoader:
        output = myNet(input)
        count_accuracy(torch.max(output, 1)[1].numpy(), label.numpy())
    print('accuracy = %.3f%%' % (float(correctNum * 100) / dataNum))

#预测并存入CSV文件
def predict(myNet):
    n = 0
    batch_size = 200
    try:
        pd.read_csv(predictRoad, encoding='utf-8')
        print("file sample_submission_predict.csv exist")
    except:
        sampleForm = pd.read_csv(sampleRoad, encoding='utf-8')
        dataset =Mydata(testRoad)
        testdata = DataLoader(dataset,batch_size=batch_size,shuffle=False)
        print("start to predict sample_submission.csv and save...")
        for datas in testdata:
            output = myNet(datas)
            results = torch.max(output, 1)[1]
            for result in results:
                sampleForm['Label'].loc[n] = result.item()
                n = n + 1
        sampleForm.to_csv(predictRoad, encoding='utf-8', index=False)
        print("complete save sample_submission_predict.csv")


if __name__ == '__main__':
    try:
        myNet = torch.load('MINIST_model.pkl')
        print("MINIST_model exist and have been loaded")
        print("testing accuracy...")
        test_existModle(myNet)
        predict(myNet)
    except:
        trainData = Mydata(trainRoad)
        trainLoader = DataLoader(trainData,batch_size=150,shuffle=True)
        myNet = Net()
        epochList = []
        lossList = []
        count = 0
        print("can't find MINIST_model, start training...")
        for epoch in range(1,41):
            correctNum = 0
            dataNum = 0
            lossNum = 0
            for inputs,label in trainLoader:
                predY = myNet.forward(inputs)
                loss = myNet.loss_func(predY,label)
                count_accuracy(torch.max(predY, 1)[1].numpy(),label.numpy())
                lossNum = lossNum + loss.item()
                myNet.optimizer.zero_grad()
                loss.backward()
                myNet.optimizer.step()
            lossList.append(float(lossNum) / dataNum)
            epochList.append(epoch-1)
            accuracy = float(correctNum*100)/dataNum
            print('epoch:%d | accuracy = %.3f%%' % (epoch, accuracy))

        print("training complete")
        if accuracy>=95:
            torch.save(myNet, 'MINIST_model.pkl')
            print("model saved")
        else:
            print("accuracy is too low,didn't saved this model")
        plt.plot(epochList,lossList)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.show()
```
 
  仅需重新编写网络结构类，需要注意的是，由于卷积函数需要输入数据的通道数，从CSV文件中读取后的数据仅为二维，要用 **unsqueeze()** 这个函数对数据维度进行扩充。

> *torch.squeeze(input, dim=None, out=None)*：去除那些维度大小为1的维度  
> 
> *torch.unbind(tensor, dim=0)*：去除某个维度   
> 
> *torch.unsqueeze(input, dim, out=None)*：在指定位置添加维度

##### [**参考**](https://www.zhihu.com/question/389021909/answer/1217800110)

### 4 LeNet-5 模型    

   1. **模型介绍**
 
<div  align="center">    
<img src="https://img-blog.csdnimg.cn/img_convert/6f0ddd628a64d04911c1853c4e8edc27.png" alt="LeNet网络图" align="middle" />
  <div align="center"><I>LeNet网络图</I></div>
</div>

- Input Layer：1 * 32 * 32图像
- Conv1 Layer：包含6个卷积核，kernal size：5 * 5，parameters:（ 5 * 5 + 1 ）* 6=156个
- Subsampling Layer：average pooling，size：2 * 2
                                  Activation Function：sigmoid
- Conv3 Layer：包含16个卷积核，kernal size：5 * 5  -> 16个Feature Map
- Subsampling Layer：average pooling，size：2 * 2
- Conv5 Layer：包含120个卷积核，kernal size：5 * 5
- Fully Connected Layer：Activation Function：sigmoid
- Output Layer：Gaussian connection  

##### [**参考**](https://www.shuzhiduo.com/A/QW5YqRVK5m/)

2. **代码实现**

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.AvgPool2d(2, 2),
            nn.Sigmoid(),
            nn.Conv2d(6, 16, 5),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.network(x)
```

经过测试发现若在某几层网络之间加入BN层和Dropout层，同时相应调整激活函数但保持原网络框架不变时，可以使神经网络准确率上升。

```python
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.BatchNorm2d(6),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        return self.network(x)
```

3. **运行结果展示**

<div  align="center">    
<img src="https://github.com/toygerse/MinistCNN/blob/master/LeNet-5/Lenet_trainaccuracy.png" alt="LeNet准确率"  align="middle" />  
  
  <div align="center"><I>LeNet训练准确率</I></div>
</div>  
  
  
<div  align="center">    
<img src="https://github.com/toygerse/MinistCNN/blob/master/LeNet-5/Lenet_testaccuracy.png" alt="LeNet准确率"  align="middle" />  
  
  <div align="center"><I>LeNet测试准确率</I></div>
</div>

### 5 ResNet 模型  
  
  1. **模型介绍**  

从经验来看，网络的深度对模型的性能至关重要，当增加网络层数后，网络可以进行更加复杂的特征模式的提取，所以当模型更深时理论上可以取得更好的结果，但是更深的网络其性能一定会更好吗？实验发现深度网络出现了退化问题网络深度增加时，网络准确度出现饱和，甚至出现下降。  
为了解决深层网络中的退化问题，可以人为地让神经网络某些层跳过下一层神经元的连接，隔层相连，弱化每层之间的强联系。这种神经网络被称为 残差网络 (ResNets)。ResNet论文提出了 residual结构（残差结构）来减轻退化问题
<div  align="center">    
<img src="https://pic2.zhimg.com/80/v2-7cb9c03871ab1faa7ca23199ac403bd9_720w.jpg" alt="ResNet"  align="middle" />  
  
  <div align="center"><I>ResNet网络结构图</I></div>
</div>


  <div  align="center">    
<img src="https://pic4.zhimg.com/80/v2-252e6d9979a2a91c2d3033b9b73eb69f_720w.jpg" alt="ResNet"  align="middle" />  
  
  <div align="center"><I>残差学习单元</I></div>
</div>      

residual结构使用了一种shortcut的连接方式，也可理解为捷径。让特征矩阵隔层相加，注意F(X)和X形状要相同，所谓相加是特征矩阵相同位置上的数字进行相加。  

  <div  align="center">    
<img src="https://pic1.zhimg.com/80/v2-1dfd4022d4be28392ff44c49d6b4ed94_720w.jpg" alt="ResNet"  align="middle" />  
  
  <div align="center"><I>不同深度的ResNet</I></div>
</div>  

conv3_x, conv4_x, conv5_x所对应的一系列残差结构的第一层残差结构都是虚线残差结构。因为这一系列残差结构的第一层都有调整输入特征矩阵shape的使命（将特征矩阵的高和宽缩减为原来的一半，将深度channel调整成下一层残差结构所需要的channel）

##### [**参考1**](https://zhuanlan.zhihu.com/p/31852747)   [**参考2**](https://blog.csdn.net/qq_45649076/article/details/120494328)

2. **代码实现**

```python
#定义残差网络块
class Basicblock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(Basicblock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = nn.Sequential()
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        out = self.network(x)+self.downsample(x)
        return out

#定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cov_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16)
        )
        self.block1 = Basicblock(16, 16)
        self.block2 = Basicblock(16, 32, 3)
        self.block3 = Basicblock(32, 32)
        self.block4 = Basicblock(32, 64, 3)
        self.block5 = Basicblock(64, 64)
        self.linear = nn.Sequential(
            nn.Linear(4*4*64,10)
        )
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.08)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,x):
        out = torch.sigmoid(self.cov_1(x))
        out = f.relu(self.block1(out))
        out = torch.sigmoid(self.block2(out))
        out = f.relu(self.block3(out))
        out = torch.sigmoid(self.block4(out))
        out = f.relu(self.block5(out))
        out = out.reshape(x.size(0),-1)
        out = f.dropout(out, 0.1)
        out = self.linear(out)
        return out

```

3. **运行结果展示**

<div  align="center">    
<img src="https://github.com/toygerse/MinistCNN/blob/master/ResNet/resnet_trainaccuracy.png" alt="RseNet训练准确率"  align="middle" />  
  
  <div align="center"><I>RseNet训练准确率</I></div>
</div>  

<div  align="center">    
<img src="https://github.com/toygerse/MinistCNN/blob/master/ResNet/resnet_testaccuracy.png" alt="RseNet测试准确率"  align="middle" />  
  
  <div align="center"><I>RseNet测试准确率</I></div>
</div>  













