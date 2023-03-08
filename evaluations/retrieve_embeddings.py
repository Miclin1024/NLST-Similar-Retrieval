import os
import torch
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
from definitions import *
from moco_lightning.builder import MLP
from moco_lightning.params import ModelParams
from torch.utils.data import Dataset, DataLoader
from data.dataset import DatasetManager, NLSTDataset


# process log to shorten 
class ShortenLog():
    similarity_log: pd.DataFrame
    k: int

    def __init__(self, log_loc, top_k=7):
        '''
        Shorten log to only keep the k most similar and disimilar scans.
        Recommend 7 scans (at most 2 of the most similar scans may come 
        from the same patient). 
        '''
        self.similarity_log = pd.read_csv(log_loc)
        self.k = top_k

        def shorten_log(array: np.array):
            '''Assume array to be structured as follows: from most similar to disimilar.'''
            lst_IDs = array.split(", ")
            return np.concatenate((lst_IDs[:top_k], lst_IDs[-top_k:]))

        columns_to_shorten = ["observed_patient_ids", 
                              "observed_scans", 
                              "observed_scores"]
        
        for col in columns_to_shorten:
            self.similarity_log[col] = self.similarity_log[col].apply(shorten_log)

    def get_similarity_log(self):
        return self.similarity_log


# load model checkpoints and pass in Cached data and save embeddings 
class CalculateEmbeddings():
    
    similarity_log: pd.DataFrame 
    CHECKPOINT_DIR: str
    CACHE_DIR: str
    SAVE_EMBEDDING_DIR: str

    model_version: str
    epoch_loaded: int
    model: torch.nn.Module
    projection_layer: torch.nn.Module
    prediction_layer: torch.nn.Module
    config: ModelParams

    def __init__(self, 
                 model_version, # model name and version
                 epoch,
                 model_params, # parameter of the model
                 top_k=7, # number of top k similar or disimilar scans to record
                 log_rootdir=os.path.join(ROOT_DIR, "logs", "same_patient"),
                 checkpoint_rootdir=os.path.join(ROOT_DIR, "logs", "models"), 
                 cache_dir=os.path.join(ROOT_DIR, "data", "cache"), 
                 save_embedding_rootdir=os.path.join(ROOT_DIR, "logs", "embeddings")): 

        self.CHECKPOINT_DIR = os.path.join(checkpoint_rootdir, model_version)
        self.CACHE_DIR = cache_dir
        self.SAVE_EMBEDDING_DIR = os.path.join(save_embedding_rootdir, model_version)
        if not os.path.exists(self.SAVE_EMBEDDING_DIR):
            os.mkdir(self.SAVE_EMBEDDING_DIR)
        self.model_version = model_version
        self.epoch_loaded = epoch

        self.config = model_params 

        log_loc = os.path.join(log_rootdir, model_version, f"epoch{epoch}.csv")
        shortener = ShortenLog(log_loc, top_k)
        self.similarity_log = shortener.get_similarity_log()

    def load_model_checkpoint(self):

        print("Initializing model...")
        self.model = self.config.encoder.to("cuda")
        self.projection_layer = MLP(self.config.embedding_dim, 
                                    self.config.dim, 
                                    self.config.mlp_hidden_dim,
                                    num_layers=self.config.projection_mlp_layers,
                                    normalization=MLP.get_normalization(self.config)).to("cuda")
        
        self.prediction_layer = MLP(self.config.dim, 
                                    self.config.dim, 
                                    self.config.mlp_hidden_dim, 
                                    num_layers=self.config.prediction_mlp_layers,
                                    normalization=MLP.get_normalization(self.config, prediction=True)).to("cuda")
        
        print(f"Loading in checkpoint of epoch {self.epoch_loaded}...")
        checkpoint = torch.load(os.path.join(self.CHECKPOINT_DIR, f"epoch{self.epoch_loaded}.pt"))  
        self.epoch_loaded = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model_state_dict"]) 
        self.model.eval()
        self.projection_layer.load_state_dict(checkpoint["projection_model_state_dict"])
        self.prediction_layer.load_state_dict(checkpoint["prediction_model_state_dict"])

    
    def load_and_save_all_embeddings(self):
        '''
        Calculate and save embeddings present in logs. 
        '''
        
        # create folder to store embeddings
        save_to_folder_name = os.path.join(self.SAVE_EMBEDDING_DIR, 
                                           f"epoch{self.epoch_loaded}")
        if not os.path.exists(save_to_folder_name):
            os.mkdir(save_to_folder_name)

        # retrieve embeddings
        manager = DatasetManager(
            list(map(lambda elem: int(elem), os.environ.get("MANIFEST_ID").split(","))), 
            default_access_mode="cached"
        )
        
        seriesIDs_need_embeddings = []

        def check_saved_embeddings(row):

            seriesIDs = np.append([row["scan_id"]], row["observed_scans"])
            for seriesID in seriesIDs:
                save_path = os.path.join(save_to_folder_name, f"{seriesID}.npy")
                if os.path.exists(save_path):
                    continue
                else:
                    seriesIDs_need_embeddings.append(seriesID)

        self.similarity_log.apply(check_saved_embeddings, axis=1)

        print("len", len(seriesIDs_need_embeddings))
        if len(seriesIDs_need_embeddings) > 0:
            emb_dataset = NLSTDataset(manager.reader, seriesIDs_need_embeddings, train=False)
            dataloader = DataLoader(emb_dataset, batch_size=self.config.eval_batch_size)
            with torch.no_grad():
                self.model.eval()

                cnt = 0
                for x, y in tqdm(dataloader, desc=f"calculating embeddings for {len(emb_dataset)} scans..."):
                    emb = self.model(x.to("cuda"))
                    emb_projection = self.projection_layer(emb)
                    emb_prediction = self.prediction_layer(emb_projection)
                    emb_prediction = emb_projection.cpu().numpy()

                    for i, emb in enumerate(emb_prediction):
                        cnt += 1
                        seriesID = y["series_id"][i]
                        save_path = os.path.join(save_to_folder_name, f"{seriesID}.npy")
                        np.save(save_path, emb)
                    print(cnt)



if __name__ == '__main__':

    #TODO: modify code such that the configuration is saved in checkpoint 
    #TODO: modify code for transformation (?)

    # parameters
    encoder_name = "r3d_18"
    batch_size = 8
    eval_batch_size = 16
    lr = 0.001
    m = 0.996
    K = 2048
    mlp_embedding_dim = 400
    mlp_output_dim = 400
    mlp_hidden_dim = 400

    encoder = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3),
            torchvision.models.video.r3d_18(
                weights=torchvision.models.video.R3D_18_Weights.DEFAULT
            )
        )
    
    model_params = ModelParams(
                encoder_name=encoder_name,
                encoder=encoder,
                embedding_dim=mlp_embedding_dim,
                dim=mlp_output_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                mlp_normalization="bn",
                lr=lr,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                m=m,
                K=K,
                max_epochs=100,
                prediction_mlp_layers=2,
            )
    
    model_version = "r3d_18_moco_v66"
    TOP_K = 7
    for e in [25, 50, 75, 99]:
        emb_calculator = CalculateEmbeddings(model_version=model_version,
                                            epoch=e,
                                            model_params=model_params,
                                            top_k=TOP_K)
        
        emb_calculator.load_model_checkpoint()
        emb_calculator.load_and_save_all_embeddings()
    