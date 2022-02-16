import math

import numpy as np
import torch
from torch import nn
from torchvision import models
import os
import torch, clip
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from data_handler import CustomCLEVRImageDataset
from torchmetrics import Accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device ", device)

DBG_MODE = not torch.cuda.is_available()
# print(f'dbg mod ={DBG_MODE}')

DATA_SIZE = 15 if DBG_MODE else 5000
NUM_WORKERS = 1 if DBG_MODE else 1
GPUS = 0 if DBG_MODE else 1

USE_CLIP = True
amount_of_options_per_relation = [3, 2, 2, 4, 5, 2]

BATCH_SIZE = 3 if DBG_MODE else 8
mlp_width = int(math.pow(2, 10 if not DBG_MODE else 10))
amount_of_relations_types, amount_of_options_for_relations_type = 6, 5
visual_features_size = 512 if USE_CLIP else 128

OUTPUT_MODE = 6  # 0-5 IS FOR SINGLE RELATION, 6 IS FOR ALL REL TOGETHER.
all_rels_together = OUTPUT_MODE == 6
relation_to_train = OUTPUT_MODE

if all_rels_together:
    train_total_rels, train_total_options = amount_of_relations_types, amount_of_options_for_relations_type
else:
    train_total_rels, train_total_options = 1, amount_of_options_per_relation[relation_to_train]
gt_relations_amount = train_total_rels * train_total_options

losses = [nn.CrossEntropyLoss().to(device) for i in range(amount_of_relations_types)]
accuracies = [0 for _ in range(amount_of_relations_types)]


class FeatureExtractor(nn.Module):
    def __init__(self, out_size, freeze=True, use_clip=USE_CLIP):
        super(FeatureExtractor, self).__init__()
        self.use_clip = use_clip
        if use_clip:
            self.clip_model, self.preprocess = clip.load("ViT-B/32", device=device)  # try "RN50"
            for param in self.clip_model.parameters():
                param.requires_grad = not freeze
        else:
            inner_model = models.vgg16(pretrained=True).to(device)
            # Extract VGG-16 Feature Layers
            self.features = list(inner_model.features)
            self.features = nn.Sequential(*self.features)
            # Extract VGG-16 Average Pooling Layer
            self.pooling = inner_model.avgpool
            # Convert the image into one-dimensional vector
            self.flatten = nn.Flatten()
            # Extract the first part of fully-connected layer from VGG16
            self.fc = nn.Linear(25088, out_size)

            for param in self.features.parameters():
                param.requires_grad = not freeze
        self.to(device)

    def forward(self, x):
        self.to(device)
        if self.use_clip:
            out = self.clip_model.encode_image(x)
        else:
            out = self.features(x)
            out = self.pooling(out)
            out = self.flatten(out)
            out = self.fc(out)
        return out


class MLP_on_features(pl.LightningModule):
    def __init__(self, input_dim=(visual_features_size + (2 * 24) + 1 + 4), freeze_feat_extract=True):
        super().__init__()
        self.mlp_layers = nn.Sequential(
            nn.Linear(input_dim, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, mlp_width),
            nn.ReLU(),
            nn.Linear(mlp_width, gt_relations_amount)
        ).to(device)
        self.feat_extract = FeatureExtractor(out_size=visual_features_size, freeze=freeze_feat_extract).to(device)
        self.feat_extract.to(device)
        self.to(device)

    def forward(self, x):
        img, ndxs = x
        feats = self.feat_extract(img.to(device))
        mlp_in = torch.cat((feats, ndxs.to(device)), 1).to(device)
        self.to(device)
        mlp_out = self.mlp_layers(mlp_in)
        return torch.reshape(mlp_out, (-1, train_total_rels, train_total_options))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32).to(device)
        y_hat = self(x)
        total_loss, accuracy = 0.0, 0.0
        if all_rels_together:
            for i in range(train_total_rels):
                total_loss += losses[i](y_hat[:, i, :], y[:, i].long())
                acc_of_rel_i = ((np.argmax(y_hat[:, i, :].cpu().detach().numpy(), axis=1) == y[:, i].long().cpu().numpy()) * 1).sum() / BATCH_SIZE
                accuracy += acc_of_rel_i
                self.log(f"{i}_train_acc", acc_of_rel_i, prog_bar=False)
            accuracy /= amount_of_relations_types
        else:
            total_loss += losses[0](y_hat[:, 0, :], y[:, relation_to_train].long())
            accuracy += ((np.argmax(y_hat[:, 0, :].cpu().detach().numpy(), axis=1) == y[:, relation_to_train].long().cpu().numpy()) * 1).sum() / BATCH_SIZE

        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.to(torch.float32).to(device)
        y_hat = self(x)

        total_loss, accuracy = 0.0, 0.0
        if all_rels_together:
            for i in range(train_total_rels):
                total_loss += losses[i](y_hat[:, i, :], y[:, i].long())
                acc_of_rel_i = ((np.argmax(y_hat[:, i, :].cpu().detach().numpy(), axis=1) == y[:, i].long().cpu().numpy()) * 1).sum() / BATCH_SIZE
                accuracy += acc_of_rel_i
                self.log(f"{i}_val_acc", acc_of_rel_i, prog_bar=False)

            accuracy /= amount_of_relations_types
        else:
            total_loss += losses[0](y_hat[:, 0, :], y[:, relation_to_train].long())
            accuracy += ((np.argmax(y_hat[:, 0, :].cpu().detach().numpy(), axis=1) == y[:, relation_to_train].long().cpu().numpy()) * 1).sum() / BATCH_SIZE

        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


def regular_training():
    my_model = MLP_on_features()
    my_model = my_model.to(device)
    my_model.train()
    dataset = CustomCLEVRImageDataset(DATA_SIZE)
    TEST_SIZE = int(0.2 * DATA_SIZE)
    train, val = random_split(dataset, [DATA_SIZE - TEST_SIZE, TEST_SIZE])
    trainloader, val = DataLoader(train, batch_size=BATCH_SIZE), DataLoader(val, batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-4)

    for epoch in range(100):  # loop over the dataset multiple times

        accuracies = [0 for i in range(amount_of_relations_types)]
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            x, y = data
            y = y.to(torch.float32)
            # x = x.view(x.size(0), -1)
            y_hat = my_model(x)
            # loss = F.mse_loss(y_hat, y.to(torch.float32))
            # todo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # todo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            total_loss = 0
            for k in range(amount_of_relations_types):
                total_loss += losses[k](y_hat[:, k, :], y[:, k].long())
                accuracies[k] += int(np.argmax(y_hat[:, k, :].detach().numpy()) == y[:, k].long()) / BATCH_SIZE
            print(total_loss.item(), "      ", np.array(accuracies) / (i + 1))

            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')


def run_exp():
    dataset = CustomCLEVRImageDataset(DATA_SIZE)
    test_size = int(DATA_SIZE * 0.2)
    train, val = random_split(dataset, [DATA_SIZE - test_size, test_size])
    my_model = MLP_on_features().to(device)
    my_model.feat_extract.to(device)
    trainer = pl.Trainer(gpus=GPUS, max_epochs=50, progress_bar_refresh_rate=20)
    trainer.fit(my_model, DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS),
                DataLoader(val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS))


if __name__ == '__main__':
    # regular_training()
    # exit()
    dataset = CustomCLEVRImageDataset(DATA_SIZE)
    test_size = int(DATA_SIZE * 0.2)
    train, val = random_split(dataset, [DATA_SIZE - test_size, test_size])
    my_model = MLP_on_features().to(device)
    my_model.feat_extract.to(device)
    trainer = pl.Trainer(gpus=GPUS, max_epochs=50, progress_bar_refresh_rate=20)
    trainer.fit(my_model, DataLoader(train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS), DataLoader(val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS))
