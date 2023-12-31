import torch;
from torch import nn;
import pytorch_lightning as pl;
from torchvision.models import resnet34;
import numpy as np;
from Base.config import IMG_SIZE, PREDICT_SIZE;

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__();
        
        pretrainNet = resnet34(pretrained = True);
        # ### ----- Transfer Learning ----- ###
        # pretrainNet.eval();
        # for param in pretrainNet.parameters():
        #     param.requires_grad = False;
        
        self.net = nn.Sequential(
            *list(pretrainNet.children())[:-2],
            
            # ### ----- vgg11 ----- ###
            # nn.Flatten(),
            # nn.Linear(25088, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(),
            # nn.Linear(4096, 1000),
            # nn.ReLU(),
            # nn.Linear(1000, config.PREDICT_SIZE)
            
            nn.Flatten(),
            nn.Linear(int(IMG_SIZE / 32 * IMG_SIZE / 32) * 512, 1000),
            nn.ReLU(),
            nn.Linear(1000, PREDICT_SIZE)
            
        ); 
        self.loss_fn = nn.L1Loss();
    def forward(self, x):
        return self.net(x.float());
    def training_step(self, batch, batch_idx):
        x, box, y = batch;
        y = torch.cat([y[:, :, 0], y[:, :, 1]], dim = 1);
        scores = self.forward(x);
        loss = self.loss_fn(scores.float(), y.float());
        self.log_dict({'train_loss': loss}, on_epoch = True, prog_bar = True);
        return loss;
    def validation_step(self, batch, batch_idx):
        x, box, y = batch
        y = torch.cat([y[:, :, 0], y[:, :, 1]], dim = 1);
        scores = self.forward(x);
        loss = self.loss_fn(scores.float(), y.float());
        self.log('val_loss', loss);
        return loss;
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = 8e-5);
