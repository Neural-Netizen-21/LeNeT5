from typing import self
from torch import nn
from torch.nn import Module
class LeNet5(nn.Module) :
    def__init__(self) :
     super(LeNet5,self).__init__()
     self.conv1=nn.Conv2d(1,6,1)
     self.pool=nn.MaxPool2d(2)
     self.conv2=nn.Conv2d(6,16,5)
     self.fc1=nn.Linear(256)
     self.fc2=nn.Linear(120)
     self.fc3=nn.Linear(84)
     self.relu=nn.Relu()

     def forward(self,x) :
        y=self.conv1(x)
        y=self.relu(y)
        y=self.pool(y)
        y=self.conv2(y)
        y=self.relu(y)
        y=self.pool(y)
        y=y.view(y.shape[0],-1)
        y=self.fc1(y)
        y=y.relu(y)
        y=self.fc2(y)
        y=y.relu(y)
        y=self.fc3(y)
        y=y.relu(y)
        return y
