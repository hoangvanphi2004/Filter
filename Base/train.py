import torch;
import pytorch_lightning as pl;
from .config import X_COLS_LEN;
from .dataset import DataModule;
from .model import Model;
import matplotlib.pyplot as plt;
import matplotlib.patches as patches;

if __name__ == "__main__":
    data = DataModule(batch_size = 32, num_workers = 0);
    
    # ### ----- Visualize Data ----- ###
    # data.setup('fit');
    # # print(next(iter(data.train_dataloader()))[1])
    # #index = 0;
    # for index in range(30):
    #     # print(data.train_dataloader().dataset[index][0].permute(1, 2, 0));
    #     #print(data.train_dataloader().dataset[index][1]);
    #     plt.scatter(data.train_dataloader().dataset[index][1]["points"][:, 0], data.train_dataloader().dataset[index][1]["points"][:, 1], c = "r", linewidths = 0.5);
    #     patch = patches.Rectangle((data.train_dataloader().dataset[index][1]['box'][0],\
    #                     data.train_dataloader().dataset[index][1]['box'][1]),\
    #                     data.train_dataloader().dataset[index][1]['box'][2],\
    #                     data.train_dataloader().dataset[index][1]['box'][3],\
    #                     linewidth = 1,\
    #                     edgecolor='g',\
    #                     facecolor="none"\
    #                 );
    
    #     plt.gca().add_patch(patch);
    #     plt.imshow(data.train_dataloader().dataset[index][0].permute(1, 2, 0));
    #     plt.show();
    
    model = Model()
    trainer = pl.Trainer(profiler = "simple", accelerator = "gpu", devices = [0], min_epochs = 30, max_epochs = 35);
    trainer.fit(model, data);
    trainer.validate(model, data);
    
    for predictIndex in range(10):
        predict = model(torch.unsqueeze(data.validateData.dataset[predictIndex][0], 0))[0];
        plt.imshow(data.validateData.dataset[predictIndex][0].permute(1, 2, 0));
        plt.scatter(predict[:X_COLS_LEN].detach(), predict[X_COLS_LEN:].detach(), c = 'r');
        plt.show();
    
    torch.save(model.state_dict(), 'resnet-for-face-points-recognize-state-dict.pth');