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

#unsqueeze(0)为了在卷积中给minist一个通道
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

def count_accuracy(pred_y, label):
    global dataNum, correctNum
    for i in range(len(label)):
        dataNum =dataNum +1
        if pred_y[i] == label[i]:
            correctNum = correctNum +1

def test_existModle(myNet):
    trainData = Mydata(trainRoad)
    trainLoader = DataLoader(trainData, batch_size=150, shuffle=True)
    for input,label in trainLoader:
        output = myNet(input)
        count_accuracy(torch.max(output, 1)[1].numpy(), label.numpy())
    print('accuracy = %.3f%%' % (float(correctNum * 100) / dataNum))

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

