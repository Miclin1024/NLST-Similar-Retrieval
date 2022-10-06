from encoders import *
from dataset import NLSTDataset
from torch.utils.data import Dataset


class Runner:
    def __init__(self, model: BaseVAE, dataset: Dataset):
        self.learning_rate = 5e-3
        self.batch_size = 64
        self.epochs = 10

        self.model = model
        self.dataset = dataset
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

    def train_loop(self, dataloader):
        for batch, X in enumerate(dataloader):
            loss_dict = self.model.loss_function(self.model(X))
            self.optimizer.zero_grad()
            loss_dict["loss"].backward()
            self.optimizer.step()

    def test_loop(self, dataloader):
        with torch.no_grad():
            recon_loss = []
            for X in dataloader:
                loss_dict = self.model.loss_function(self.model(X))
                recon_loss.append(loss_dict["Reconstruction_Loss"].item())

