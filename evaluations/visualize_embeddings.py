import os
import numpy as np
import pandas as pd
from definitions import *
from typing import Optional
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from data.reader import NLSTDataReader
from evaluations.retrieve_embeddings import CalculateEmbeddings, ShortenLog


class DataDict():
    """Provides information about metadata."""
    
    data_dict = pd.DataFrame
    metadata = pd.DataFrame    
    
    def __init__(self, data_dict_loc="Data dictionary.csv", 
                 metadata_loc="nlst_297_prsn_20170404.csv"):
        
        self.data_dict = pd.read_csv(data_dict_loc)
        self.metadata = pd.read_csv(metadata_loc)
        all_categories = np.unique(self.data_dict["Category"])
        
        
        by_categories = {}

        for category in all_categories:
            col_by_categories = self.data_dict[self.data_dict["Category"] == category]["Variable"].values
            col_by_categories = [col for col in col_by_categories if col in self.metadata.columns]
            by_categories[category] = self.metadata.loc[:, col_by_categories]
            
        self.by_categories = by_categories

    
    def __str__(self):
        return f"All categories: {np.unique(self.data_dict['Category'])}"
        
    def return_category(self, category):
        '''Returns the columns in the metadata belonging to the given category.'''
        
        assert category in self.by_categories, f"category {category} doesn't exist"
        
        return self.by_categories[category]
    
    def category_info(self, category):
        '''Returns information about a specific category.'''
        
        assert category in self.by_categories, f"category {category} doesn't exist"
        
        return self.data_dict[self.data_dict["Category"] == category]
    
    def column_desc(self, column_name):
        '''Returns description about a column name from the metadata table.'''
        
        assert column_name in self.metadata.columns, "column not found"
        
        desc = self.data_dict[self.data_dict["Variable"] == column_name] 
        print(f"Label: {desc['Label'].values}")
        print(f"Description: {desc['Description'].values}")
        return desc

class UnderstandModel():

# class Visualize():

    log: pd.DataFrame
    metadata: pd.DataFrame 
    manifest: pd.DataFrame
    model_version: str
    epoch: int
    top_k: int
    SAVE_EMBEDDING_DIR: str

    # cache
    TSNE_projections: np.ndarray
    

#TODO: make top_k an environment variable to be imported from visualize data
    def __init__(self, 
                 log_loc, 
                 top_k,
                 model_version, 
                 epoch, 
                 metadata=os.path.join(ROOT_DIR, "metadata", "nlst_297_prsn_20170404.csv"),
                 save_embedding_rootdir=os.path.join(ROOT_DIR, "logs", "embeddings")):

        self.metadata = pd.read_csv(metadata).set_index("pid")

        shortener = ShortenLog(log_loc, top_k)
        log = shortener.get_similarity_log()
        log["true_patient_id"] = log["true_patient_id"].astype(int)
        log["observed_patient_ids"] = log["observed_patient_ids"].apply(lambda arr: arr.astype(float).astype(int))
        self.log = log

        self.model_version = model_version
        self.epoch = epoch
        self.SAVE_EMBEDDING_DIR = os.path.join(save_embedding_rootdir, model_version, f"epoch{epoch}")
        self.top_k = top_k
        manifests = list(map(lambda elem: int(elem), os.environ.get("MANIFEST_ID").split(",")))
        reader = NLSTDataReader(manifests, default_access_mode="cache")
        self.manifest = reader.manifest


    def retrieve_embeddings(self):

        embeddings = []
        seriesIDs = []
        for scan in os.listdir(self.SAVE_EMBEDDING_DIR):
            
            path_to_scan = os.path.join(self.SAVE_EMBEDDING_DIR, scan)
            embeddings.append(np.load(path_to_scan, allow_pickle=True))
            seriesIDs.append(scan[:-4])

        return embeddings, seriesIDs

    # def check_folder(self):

    #     print(f"embedding folder: {len(os.listdir(self.SAVE_EMBEDDING_DIR))}")
    #     print(f"number of unique scans: {len(np.unique(self.get_all_seriesIDs()))}")


    def get_pids_similar(self):
        '''Get a 2D array with just pids. 
        An element of the array looks like the following:
        [PID of query scan, PID of most similar scan, PID of 2nd similar scan, ...]
        '''        
        top_k_similar = self.log["observed_patient_ids"].apply(lambda arr: arr[:self.top_k])
        top_k_similar = np.array([list(lst) for lst in top_k_similar.values])
        query_patient_ids = self.log["true_patient_id"].values[:,np.newaxis]
        return np.hstack((query_patient_ids, top_k_similar))
                
    def get_pids_disimilar(self):

        top_k_disimilar = self.log["observed_patient_ids"].apply(lambda arr: arr[:self.top_k])
        top_k_disimilar = np.array([list(lst) for lst in top_k_disimilar.values])
        query_patient_ids = self.log["true_patient_id"].values[:,np.newaxis]
        return np.hstack((query_patient_ids, top_k_disimilar))
    
    def get_all_pids(self):
        
        similar_disimilar_scans = np.array([list(lst) for lst in self.log["observed_patient_ids"].values])
        query_patient_ids = self.log["true_patient_id"].values[:,np.newaxis]
        return np.hstack((query_patient_ids, similar_disimilar_scans))

    
class Visualize(UnderstandModel):

    def __init__(self, 
                 log_loc, 
                 top_k, 
                 model_version, 
                 epoch, 
                 metadata=os.path.join(ROOT_DIR, "metadata", "nlst_297_prsn_20170404.csv"), 
                 save_embedding_rootdir=os.path.join(ROOT_DIR, "logs", "embeddings")):
        
        super().__init__(log_loc, top_k, model_version, epoch, metadata, save_embedding_rootdir)
    
    
    def tsne(self, attribute: Optional[str]):
        
        assert attribute in self.metadata.columns, "attribute doesn't exist"

        embeddings, seriesIDs = self.retrieve_embeddings()
        PIDs = self.manifest.loc[seriesIDs, "Subject ID"]
        tsne = TSNE(n_components=3, random_state=0)
        # TODO: formating changes for embeddings
        if not self.TSNE_projections: 
            projections = tsne.fit_transform(np.array(embeddings))
        else: 
            projections = self.TSNE_projections

        if attribute: 
            fig = px.scatter_3d(
                projections, x=0, y=1, z=2,
                color=self.metadata.loc[PIDs, :][attribute].values, labels={'color': attribute}            
            )
        else:
            fig = px.scatter_3d(
                projections, x=0, y=1, z=2
            )

        fig.update_traces(marker_size=5)
        fig.show()
    
    def calculate_std_(self, attribute: str):
        '''
        Each entry in the similarity log has a query scan + top k most similar scans. 
        For a quantitative attribute (ex: height, weight), calculate cthe standard deviation
        of that quantiative attribute for all entries in the simlarity log. 
        Graph a histogram of all of the standard deviations. 
        '''
        assert attribute in self.metadata.columns, "attribute doesn't exist"

        stdeviations = []
        pids = self.get_pids_similar()

        for i,_ in enumerate(pids):

            std = np.std(self.metadata.loc[pids[i,:],[attribute]].values)
            stdeviations.append(std)
            
        fig = px.histogram(stdeviations)
        fig.update_layout(
            xaxis_title=f"Standard deviations in {attribute} among the top {self.top_k} scans")
        
        fig.show()

    def calculate_prop_(self, attribute: str):
        '''


        Each entry in the similarity log has a query scan + top k most similar scans. 
        Let an binary attribute (ex: gender, having a diagnosis) be 0 or 1. 
        Calculate the proportion of the top-k most similar scans in an entry with binary attribute = 0.
        Graph a histogram of all of the proportions calculated for each entry. 
        '''
        assert attribute in self.metadata.columns, "attribute doesn't exist"

        binary_markers = np.unique(self.metadata[attribute])
        binary_markers = binary_markers[~np.isnan(binary_markers)]

        assert len(binary_markers) == 2, f"{attribute} is not binary attribute, try calculate_std_ instead"

        props = []
        attributes = []
        pids = self.get_pids_similar()
        for j in range(len(pids)):
            
            query_scan_attribute = self.metadata.loc[pids[j][0],:][attribute]
            attributes.append(query_scan_attribute)

            data = self.metadata.loc[pids[j][1:],:][attribute].values
            prop = np.sum(data == query_scan_attribute) / self.top_k
            props.append(prop)
        
        fig = px.histogram(props, color=attributes)
        #histnorm="percent"

        print(f"Count of {attribute} in entire dataset is: {np.unique(self.metadata[attribute], return_counts=True)}")
        fig.update_layout(
            xaxis_title=f"Proportion of top k scans have attribute {attribute} similar to query scan",
            yaxis_title=f"Count of entries", 
            barmode="group")
        
        fig.show()
    

    
    # def calculate_accuracy(self):
        
    #     PIDS = self.get_all_pids()
    #     accuracy = np.sum(PIDS[:, 0] == PIDS[:, 1]) / len(PIDS)
    #     print(f"Accuracy of {self.model_version}, epoch {self.epoch} is {accuracy:.3%}")
    #     return accuracy
    
if __name__ == '__main__':

    data_dict_loc = os.path.join(ROOT_DIR, "metadata", "Data dictionary.csv")
    metadata_loc = os.path.join(ROOT_DIR, "metadata/nlst_297_prsn_20170404.csv")
    dd = DataDict(data_dict_loc=data_dict_loc, metadata_loc=metadata_loc)

    demographic = ["age", "height", "weight", "gender"]
    diagnoses = dd.category_info("Disease history")[16:]["Variable"].values
    smoking = dd.category_info("Smoking")["Variable"].values

    # model_version = "r3d_18_moco_v66"
    # epoch = 99

    # for epoch in [0, 25, 50, 75, 99]:

    #     visualizer = Visualize(log_loc=os.path.join(ROOT_DIR, "logs", "same_patient", model_version, f"epoch{epoch}.csv"), 
    #                         top_k=7,
    #                         model_version=model_version,
    #                         epoch=epoch,
    #                         )
        # visualizer.calculate_accuracy()
        # print(f"epoch: {epoch}")
        # visualizer.check_folder()