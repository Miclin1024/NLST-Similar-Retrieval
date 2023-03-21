import os
import torch        
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from definitions import *
import sklearn.metrics as metrics
from torch.utils.data import Dataset, DataLoader
from evaluations.visualize_embeddings import UnderstandModel


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Linear(400, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = torch.sigmoid(out)
        return out

class MLP_Regression(nn.Module):
    def __init__(self):
        super(MLP_Regression, self).__init__()
        self.fc = nn.Linear(400, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out
    
class Embedding_Attribute(Dataset):

    attribute_dict: dict

    def __init__(self, attribute_dict):
        self.attribute_dict = attribute_dict

    def __len__(self):
        return len(self.attribute_dict)

    def __getitem__(self, index):
        seriesID = list(self.attribute_dict.keys())[index]

        _, attribute, embedding = self.attribute_dict[seriesID]
        
        return embedding, attribute


class Probing(UnderstandModel):

    prober: nn.Module
    learning_rate: float
    prober_num_epoch: int
    attribute: str
    train_test_split: float

    binary_markers: list

    def __init__(self, 
                 attribute,
                 prober, 
                 lr, 
                 prober_num_epoch,
                 log_loc, 
                 top_k,
                 model_version, 
                 epoch, 
                 train_test_split=0.7,
                 metadata=os.path.join(ROOT_DIR, "metadata", "nlst_297_prsn_20170404.csv"),
                 save_embedding_rootdir=os.path.join(ROOT_DIR, "logs", "embeddings")):

        super().__init__(log_loc, top_k, model_version, epoch, metadata, save_embedding_rootdir)

        assert attribute in self.metadata.columns, "attribute doesn't exist"

        self.attribute = attribute
        self.prober = prober.to("cuda")
        self.learning_rate = lr
        self.prober_num_epoch = prober_num_epoch
        self.train_test_split = train_test_split
    
    def removing_nan_values(self):
        # for all query scans in validation dataset, return pid | attribute 
        # take out any pids where attribute = nan 
        all_pids = self.log["true_patient_id"]
        pids_attributes = self.metadata.loc[all_pids, self.attribute]
        
        # report nan values 
        no_nan = pids_attributes[~pids_attributes.isna()]

        print(f"""Original dataset length is {pids_attributes.shape[0]}; 
        There are {pids_attributes.shape[0] - no_nan.shape[0]} nan values; 
        After eliminating nan values, we have {no_nan.shape[0]} values left.""")
        return no_nan
    
    def make_dataloaders(self, series_dict):
        # split into train vs test
        random.seed(123)
        keys = list(series_dict.keys())
        random.shuffle(keys)
        split_index = int(len(keys) * self.train_test_split)
        train_dict = {k: series_dict[k] for k in keys[:split_index]}
        test_dict = {k: series_dict[k] for k in keys[split_index:]}

        train_dataset = Embedding_Attribute(train_dict)
        self.train_dataloader = DataLoader(train_dataset, batch_size=8)

        test_dataset = Embedding_Attribute(test_dict)
        self.test_dataloader = DataLoader(test_dataset, batch_size=8)
        
        print(f"Train dataset size: {len(train_dataset)}; Test dataset size: {len(test_dataset)}")
    
    def train_prober(self, optimizer, loss_fn):
        
        for epoch in tqdm(range(self.prober_num_epoch), desc=f"training linear prob..."):
            running_loss = 0.0
            for i, (x, y) in enumerate(self.train_dataloader):

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = self.prober(x.to("cuda"))
        
                loss = loss_fn(torch.squeeze(outputs).type(torch.float), y.to("cuda").type(torch.float))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
    
    
class Probing_Regressor(Probing):

    def __init__(self, attribute, prober, lr, prober_num_epoch, log_loc, top_k, model_version, epoch, train_test_split=0.7, metadata=os.path.join(ROOT_DIR, "metadata", "nlst_297_prsn_20170404.csv"), save_embedding_rootdir=os.path.join(ROOT_DIR, "logs", "embeddings")):
        super().__init__(attribute, prober, lr, prober_num_epoch, log_loc, top_k, model_version, epoch, train_test_split, metadata, save_embedding_rootdir)

        # check that attribute is not a binary attribute
        values = np.unique(self.metadata[attribute])
        assert len(values[~np.isnan(values)]) > 2, "attribute is binary, use Probing_Classifier instead"
    
    def filter_to_earliest_timestamp(self, manifest_no_nan):

        '''
        The NLST dataset only records the weight for the first scan, so we'll only use 
        scans recorded as the first time stamp for linear probing.
        '''

        yr = manifest_no_nan["Study Date"].apply(lambda date: date.split("-")[-1])
        # 1999 is the earliest time stamp
        return manifest_no_nan.loc[(yr == "1999")]
    
    def setup_dataloader(self):

        no_nan = self.removing_nan_values()

        if self.attribute == "weight":
            manifest_no_nan = self.manifest[self.manifest["Subject ID"].isin(no_nan.index)]
            no_nan = self.filter_to_earliest_timestamp(manifest_no_nan=manifest_no_nan)
            print(f"""Dataset only records weight for the scan at the first timestep.
            Using only scans from the first timestep, we have {no_nan.shape[0]} records.""")

            # create DataLoader 
            series_dict = {}
            for seriesID in no_nan.index:
                
                pid = self.manifest.loc[seriesID]["Subject ID"]
                attribute_of_pid = self.metadata.loc[pid, self.attribute]

                if f"{seriesID}.npy" in os.listdir(self.SAVE_EMBEDDING_DIR):
                    embedding = np.load(os.path.join(self.SAVE_EMBEDDING_DIR, f"{seriesID}.npy"), allow_pickle=True)
                    series_dict[seriesID] = (pid, attribute_of_pid, embedding)

        else:

            # create DataLoader 
            series_dict = {}
            for pid in no_nan.index:
            
                seriesIDs = self.manifest[self.manifest["Subject ID"] == pid].index
                attribute_of_pid = self.metadata.loc[pid, self.attribute]
            
                for seriesID in seriesIDs:
                    if f"{seriesID}.npy" in os.listdir(self.SAVE_EMBEDDING_DIR):
                        embedding = np.load(os.path.join(self.SAVE_EMBEDDING_DIR, f"{seriesID}.npy"), allow_pickle=True)
                        series_dict[seriesID] = (pid, attribute_of_pid, embedding)

        self.make_dataloaders(series_dict)
    
    def test_prober(self):

        predictions = []
        true_values = []
        
        self.prober.eval()
        with torch.no_grad():

            for x, y in tqdm(self.test_dataloader, desc="Testing linear prob..."):
                outputs = self.prober(x.to("cuda"))
                print(outputs, y)
                predictions.append(outputs.cpu())
                true_values.append(y)

            print(torch.cat(predictions))
            print("True values")
            print(torch.cat(true_values))
            # calculate RMSE 
            loss = nn.MSELoss()
            loss_value = loss(torch.cat(predictions), torch.cat(true_values))
            print('RMSE: {:.4f}'.format(loss_value**(1/2)))

class Probing_Classifier(Probing):

    train_dataloader: DataLoader
    test_dataloader: DataLoader

    def __init__(self, attribute, prober, lr, prober_num_epoch, log_loc, top_k, model_version, epoch, train_test_split=0.7, metadata=os.path.join(ROOT_DIR, "metadata", "nlst_297_prsn_20170404.csv"), save_embedding_rootdir=os.path.join(ROOT_DIR, "logs", "embeddings")):
        super().__init__(attribute, prober, lr, prober_num_epoch, log_loc, top_k, model_version, epoch, train_test_split, metadata, save_embedding_rootdir)

        # check that attribute is a binary attribute
        binary_markers = np.unique(self.metadata[attribute])
        self.binary_markers = binary_markers[~np.isnan(binary_markers)]
        assert len(binary_markers) == 2, f"{attribute} is not binary attribute, cannot use classifier on this attribute"

    
    def setup_dataloader(self):
        
        # remove nan values 
        no_nan = self.removing_nan_values()
        counts = np.vstack(np.unique(no_nan, return_counts=True))
        
        # report counts for binary markers 
        print(f"""Count of attribute {self.attribute}: 
        {counts}""")
        print(f"Binary markers are {self.binary_markers[0]}->0, {self.binary_markers[1]}->1")
        

        # create DataLoader 
        series_dict = {}
        for pid in no_nan.index:
            
            seriesIDs = self.manifest[self.manifest["Subject ID"] == pid].index
            attribute_of_pid = self.metadata.loc[pid, self.attribute]
            # convert binary markers to 0 and 1 
            attribute_of_pid = 0 if attribute_of_pid == self.binary_markers[0] else 1
            for seriesID in seriesIDs:
                if f"{seriesID}.npy" in os.listdir(self.SAVE_EMBEDDING_DIR):
                    embedding = np.load(os.path.join(self.SAVE_EMBEDDING_DIR, f"{seriesID}.npy"), allow_pickle=True)
                    series_dict[seriesID] = (pid, attribute_of_pid, embedding)

        self.make_dataloaders(series_dict)

    def test_prober(self, threshold=0.5):

        total = 0
        correct = 0
        pred_probs = []
        true_labels = []
        
        self.prober.eval()

        for x, y in tqdm(self.test_dataloader, desc="Testing linear prob..."):
            outputs = self.prober(x.to("cuda"))
            predicted = [1 if prob > 0.5 else 0 for prob in outputs.data]  
            total += y.size(0)
            correct += (predicted == y.numpy()).sum().item()
            pred_probs.extend(outputs.tolist())
            true_labels.extend(y.tolist())

        # calculate the test accuracy and loss
        test_accuracy = 100 * correct / total
        print('Test Accuracy: {:.2f}%'.format(test_accuracy))

        # calculate the AUC score
        fpr, tpr, thresholds = metrics.roc_curve(true_labels, [p[0] for p in pred_probs], pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        print('AUC Score: {:.4f}'.format(auc_score))

if __name__ == '__main__':

    # prober_params = {
    #     "attribute": "gender",
    #     "prober": MLP(),
    #     "lr": 0.01,
    #     "prober_num_epoch": 100,
    #     "top_k": 7,
    #     "model_version": "r3d_18_moco_v66",
    #     "epoch": 99,
    #     "loss_fn": nn.BCELoss()
    # }

    # prober_params["log_loc"] = os.path.join(ROOT_DIR, "logs", "same_patient", 
    #                                         prober_params["model_version"], 
    #                                         f"epoch{prober_params['epoch']}.csv")
    # prober_params["optimizer"] = torch.optim.SGD(prober_params["prober"].parameters(),lr=prober_params["lr"])

    # prober = Probing_Classifier(attribute=prober_params["attribute"],
    #                            prober=prober_params["prober"], 
    #                            lr=prober_params["lr"], 
    #                            prober_num_epoch=prober_params["prober_num_epoch"],
    #                            log_loc=prober_params["log_loc"], 
    #                            top_k=prober_params["top_k"],
    #                            model_version=prober_params["model_version"],
    #                            epoch=prober_params["epoch"])
    
    # prober.setup_dataloader()
    # prober.train_prober(optimizer=prober_params["optimizer"], loss_fn=prober_params["loss_fn"])
    # prober.test_prober()


    prober_reg_params = {
        "attribute": "weight",
        "prober": MLP_Regression(),
        "lr": 0.01,
        "prober_num_epoch": 100,
        "top_k": 7,
        "model_version": "r3d_18_moco_v66",
        "epoch": 99,
        "loss_fn": nn.MSELoss()
    }

    prober_reg_params["log_loc"] = os.path.join(ROOT_DIR, "logs", "same_patient", 
                                            prober_reg_params["model_version"], 
                                            f"epoch{prober_reg_params['epoch']}.csv")
    prober_reg_params["optimizer"] = torch.optim.SGD(prober_reg_params["prober"].parameters(),lr=prober_reg_params["lr"])

    prober = Probing_Regressor(attribute=prober_reg_params["attribute"],
                               prober=prober_reg_params["prober"], 
                               lr=prober_reg_params["lr"], 
                               prober_num_epoch=prober_reg_params["prober_num_epoch"],
                               log_loc=prober_reg_params["log_loc"], 
                               top_k=prober_reg_params["top_k"],
                               model_version=prober_reg_params["model_version"],
                               epoch=prober_reg_params["epoch"])
    
    prober.setup_dataloader()
    prober.train_prober(optimizer=prober_reg_params["optimizer"], loss_fn=prober_reg_params["loss_fn"])
    prober.test_prober()

    #TODO: implement weight 
    #TODO: control task / baseline and sensitivity 
