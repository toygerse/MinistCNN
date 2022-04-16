import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

dataNum = 0
correctNum = 0

testRoad = "./MNIST_dataset/test.csv"
trainRoad = "./MNIST_dataset/train.csv"
sampleRoad = "./MNIST_dataset/sample_submission.csv"
predictRoad = "./MNIST_dataset/sample_submission_predict.csv"

def read_csv(fileRoad):
    return np.loadtxt(fileRoad, dtype=str, skiprows=1)

def count_accuracy(pred_y, label):
    global dataNum, correctNum
    for i in range(len(label)):
        dataNum =dataNum +1
        if pred_y[i] == label[i]:
            correctNum = correctNum +1

def test_existModle(myNet):
    trainData = Mydata(trainRoad)
    trainLoader = DataLoader(trainData, batch_size=200, shuffle=True)
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

if __name__ == '__main__':
    try:
        myNet = torch.load('MINIST_model.pkl')
        print("MINIST_model exist and have been loaded")
        print("testing accuracy...")
        test_existModle(myNet)
        predict(myNet)
    except:
        trainData = Mydata(trainRoad)
        trainLoader = DataLoader(trainData,batch_size=200,shuffle=True)
        myNet = Net()
        epochList = []
        lossList = []
        count = 0
        print("can't find MINIST_model, start training...")
        for epoch in range(1,181):
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
