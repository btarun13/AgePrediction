import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import time
import sys
import torch

from PIL import Image
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

rel_path = ""
val_acc_gen = []
val_mae_age = []
train_acc_gen = []
train_mae_age = []


class ShuffleneFull(nn.Module):

    def __init__(self):
        super(ShuffleneFull, self).__init__()
        self.model = models.shufflenet_v2_x1_0(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 3)) #Used to be 3 here

    def forward(self, x):
        return self.model(x)

class TrainModel:

    def __init__(self, model, train_dl, valid_dl, optimizer, certrion, scheduler, num_epochs):

        self.num_epochs = num_epochs
        self.model = model
        self.scheduler = scheduler
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.optimizer = optimizer
        self.certrion = certrion

        self.loss_history = []
        self.best_acc_valid = 0.0
        self.best_wieght = None

        self.training()

    def training(self):

        valid_acc = 0
        for epoch in range(self.num_epochs):

            print('Epoch %2d/%2d' % (epoch + 1, self.num_epochs))
            print('-' * 15)

            t0 = time.time()
            train_acc = self.train_model()
            valid_acc = self.valid_model()
            if self.scheduler:
                self.scheduler.step()

            time_elapsed = time.time() - t0
            print('  Training complete in: %.0fm %.0fs' % (time_elapsed // 60, time_elapsed % 60))
            print('| val_acc_gender | val_l1_loss | acc_gender | l1_loss |')
            print('| %.3f          | %.3f       | %.3f      | %.3f   | \n' % (valid_acc[0], valid_acc[1], train_acc[0], train_acc[1]))

 
            
            val_acc_gen.append(valid_acc[0])
            val_mae_age.append(valid_acc[1])

            train_acc_gen.append(train_acc[0])
            train_mae_age.append(train_acc[1])
            

            if valid_acc[0] > self.best_acc_valid:
                self.best_acc_valid = valid_acc[1]
                self.best_wieght = self.model.state_dict().copy()
        return

    def train_model(self):
        self.model.train()
        N = len(self.train_dl.dataset)
        step = N // self.train_dl.batch_size

        avg_loss = 0.0
        acc_gender = 0.0
        loss_age = 0.0

        for i, (x, y) in enumerate(self.train_dl):
            x, y = x.cuda(), y.cuda()
            # forward
            pred_8 = self.model(x)
            # loss
            loss = self.certrion(pred_8, y)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # statistics of model training
            avg_loss = (avg_loss * i + loss) / (i + 1)
            acc_gender += accuracy_gender(pred_8, y)
            loss_age += l1loss_age(pred_8, y)

            self.loss_history.append(avg_loss)

            # report statistics
            sys.stdout.flush()
            sys.stdout.write("\r  Train_Step: %d/%d | runing_loss: %.4f" % (i + 1, step, avg_loss))

        sys.stdout.flush()
        return torch.tensor([acc_gender, loss_age]) / N

    def valid_model(self):
        print()
        self.model.eval()
        N = len(self.valid_dl.dataset)
        step = N // self.valid_dl.batch_size
        acc_gender = 0.0
        loss_age = 0.0

        with torch.no_grad():
            for i, (x, y) in enumerate(self.valid_dl):
                x, y = x.cuda(), y.cuda()

                score = self.model(x)
                acc_gender += accuracy_gender(score, y)
                loss_age += l1loss_age(score, y)

                sys.stdout.flush()
                sys.stdout.write("\r  Vaild_Step: %d/%d" % (i, step))

        sys.stdout.flush()
        return torch.tensor([acc_gender, loss_age]) / N

def accuracy_gender(input, targs):
    pred = torch.argmax(input[:, :2], dim=1)
    y = targs[:, 0]
    return torch.sum(pred == y)

def l1loss_age(input, targs):
    return F.l1_loss(input[:, -1], targs[:, -1]).mean()       


class MultitaskDataset(Dataset):

    def __init__(self, data, tfms, root=rel_path + 'data/'):
        self.root = root
        self.tfms = tfms
        self.ages = data[:, 3].astype("int")
        self.races = data[:, 2].astype("int")
        self.genders = data[:, 1].astype("int")
        self.imgs = data[:, 0]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.tfms(Image.open(self.root + self.imgs[i])), torch.tensor(
            [self.genders[i], self.races[i], self.ages[i]]).float()

    def __repr__(self):
        return f'{type(self).__name__} of len {len(self)}'
    
    
def multitask_loss(input, target):
    input_gender = input[:, :2]
    input_age = input[:, -1]

    loss_gender = F.cross_entropy(input_gender, target[:, 0].long())
    loss_age = F.l1_loss(input_age, target[:, 2])

    return loss_gender / (.16) + loss_age * 2



def get_plots(x):
    plt.plot(x[:,1], label = "Validation")
    plt.plot(x[:,3], label = "Training")
    plt.xlabel("EPOCHS")
    plt.legend()
    plt.title("MAE (Age)")
    plt.savefig("plots/age.png", dpi=200)

    plt.clf()
    plt.plot(x[:,0], label = "Validation")
    plt.plot(x[:,2], label = "Training")
    plt.xlabel("EPOCHS")
    plt.legend()
    plt.title("Accuracy (Gender)")
    plt.savefig("plots/gender.png", dpi=200)



mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
sz = 112
bs = 64

tf = {'train': transforms.Compose([
    transforms.RandomRotation(degrees=0.2),
    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomGrayscale(p=.2),
    transforms.Resize((sz, sz)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)]),
    'test': transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}


data = np.load(rel_path + "data/labels.npy", allow_pickle=True)
train_data = data[data[:, -1] == "1"]
valid_data = data[data[:, -1] == "0"]

valid_ds = MultitaskDataset(data=valid_data, tfms=tf['test'])
train_ds = MultitaskDataset(data=train_data, tfms=tf['train'])

train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=1)
valid_dl = DataLoader(valid_ds, batch_size=bs, shuffle=True, num_workers=1)



if __name__ == '__main__':
    model = ShuffleneFull().cuda()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    history = TrainModel(model, train_dl, valid_dl, optimizer, multitask_loss, scheduler, 100)
    
    res = np.array([val_acc_gen, val_mae_age, train_acc_gen, train_mae_age]).T
    get_plots(res)
    torch.save(model.state_dict(), "model/saved_model1.pth")
    np.save("plots/results.npy", res)
