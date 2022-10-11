import os

import numpy as np
import torch.backends.mps

from encoders import *
from dataset import NLSTDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Runner:

    MANIFEST = 1663396252954

    def __init__(self, model: BaseVAE):
        print(f"-------------------------")
        self.learning_rate = 5e-5
        self.batch_size = 8
        self.epochs = 10

        # Note: MPS currently doesn't support Conv3D
        if torch.cuda.is_available():
            self.device = torch.cuda.device
        else:
            self.device = torch.device("cpu")

        print(f"Device is {self.device}")

        self.model = model.to(self.device)
        self.kld_weight = 3e-6
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

    def train_loop(self, dataloader):
        size = len(dataloader.dataset)
        for batch, X in enumerate(dataloader):
            loss_dict = self.model.loss_function(
                *self.model(X.float().to(self.device)),
                M_N=self.kld_weight,
            )
            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            self.optimizer.step()

            print(f"[{self.batch_size * batch + len(X)}/{size}] Training loss: {loss_dict['loss'].item()}")

    def test_loop(self, dataloader):
        with torch.no_grad():
            recon_loss = []
            for X in dataloader:
                loss_dict = self.model.loss_function(
                    *self.model(X.float().to(self.device)),
                    M_N=self.kld_weight,
                )
                recon_loss.append(loss_dict["loss"].item())
            print(f"Average test loss: {np.mean(recon_loss)}")

    def model_train(self):
        print("Preparing training dataloader")
        train_dataloader = DataLoader(
            NLSTDataset(self.MANIFEST, train=True),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        print("Preparing testing dataloader")
        test_dataloader = DataLoader(
            NLSTDataset(self.MANIFEST, train=False),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
        )

        for t in range(self.epochs):
            print(f"-------- Epoch {t + 1} --------")
            self.train_loop(train_dataloader)
            self.test_loop(test_dataloader)


if __name__ == '__main__':
    runner = Runner(
        model=VanillaVAE(latent_dimensions=32),
    )
    runner.model_train()
