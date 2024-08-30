import torch;
import pytorch_lightning as pl;
from Base.config import X_COLS_LEN, IMG_SIZE;
from Base.dataset import DataModule;
from Base.model import Model;
import matplotlib.pyplot as plt;
import matplotlib.patches as patches;
from pytorch_lightning.loggers import TensorBoardLogger;
from sys import argv

def train_model():
    data = DataModule(batch_size = 32, num_workers = 0);
    
    # ### ----- Visualize Data For Testing ----- ###
    # data.setup('fit');
    # # print(next(iter(data.train_dataloader()))[1])
    # #index = 0;
    # for index in range(30):
    #     # print(data.train_dataloader().dataset[index][0].permute(1, 2, 0));
    #     # print(data.train_dataloader().dataset[index][1]);
        
    #     cur = data.train_dataloader().dataset[index];
        
    #     plt.scatter(cur[1]["points"][:, 0], cur[1]["points"][:, 1], c = "r", linewidths = 0.5);
    #     plt.imshow(cur[0].permute(1, 2, 0));
    #     plt.show();
        
    model = Model()
    logger = TensorBoardLogger("tb_logs", name = "FilterLogger");
    trainer = pl.Trainer(logger = logger, profiler = "simple", accelerator = "gpu", devices = 1, min_epochs = 30, max_epochs = 35);
    trainer.fit(model, data);
    trainer.validate(model, data);
    
    # ### ----- Visualize Data For Testing ----- ###
    # for predictIndex in range(10):
    #     cur = data.validateData.dataset[predictIndex];
    #     predict = model(torch.unsqueeze(cur[0], 0))[0];
    #     plt.imshow(cur[0].permute(1, 2, 0));
    #     plt.scatter(predict[:X_COLS_LEN].detach(), predict[X_COLS_LEN:].detach(), c = 'r');
    #     plt.show();
    
    torch.save(model.state_dict(), argv[1]);

train_model()