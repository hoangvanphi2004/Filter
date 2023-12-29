import torch;
import matplotlib.pyplot as plt;
import config as configApp;
from Base import model, config;

def loadModel():
    net = model.Model();
    net.load_state_dict(torch.load("resnet-for-face-points-recognize-state-dict.pth"));
    return net;
    
def predictKeypoints(image, net):
    predict = net(torch.tensor(image).unsqueeze(0))[0];
    return predict;