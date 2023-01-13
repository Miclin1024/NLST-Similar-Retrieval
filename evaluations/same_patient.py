import torch
import attrs
import numpy as np
from tqdm import tqdm
from definitions import *
from data import NLSTDataReader
from torch.nn import CosineSimilarity

TOP_NO_SIMILAR: int = 10
EVAL_BATCH_SIZE = 32

@attrs.define(init=False)
class SamePatientEvaluator:
    
    encoder: torch.nn.Module
    reader: NLSTDataReader
    log_predictions: dict[SeriesID, list[SeriesID]]
    series_correct_pred: list[SeriesID]
    
    def __init__(self, encoder: torch.nn.Module, reader: NLSTDataReader) -> None:
        self.encoder = encoder
        self.reader = reader
    
    @torch.no_grad()
    def score(self, series_ids: list[SeriesID], log_similar_scans = False) -> float:
        """
        Evaluate the same patient prediction performance of the encoder. The encoder will be given 
        single 4D tensors (1, C, W, H, D).
        """

        n = len(series_ids)
        cos = CosineSimilarity(dim=1)
        
        similarity_matrix = np.zeros((n, n))
        patient_ids_embeddings: list[(PatientID, torch.Tensor)] = [None] * n
        def load_encode_series(index: int):
            if patient_ids_embeddings[index] is None:
                image, metadata = self.reader.read_series(series_ids[index])
                image = torch.unsqueeze(image.tensor, 0).to("cuda")
                pid = metadata["pid"]
                embedding = self.encoder(image).view(1, -1)
                patient_ids_embeddings[index] = (pid, embedding)
            
            return patient_ids_embeddings[index]
        
        # def load_encode_series_batch(batch_number: int):
        for batch_num in tqdm(range(n // EVAL_BATCH_SIZE + 1), 
                              desc="Encoding raw data for cosine similarity comparison"):
            idx_start = batch_num * EVAL_BATCH_SIZE
            idx_end = min(idx_start + EVAL_BATCH_SIZE, n)
            
            effective_batch_size = idx_end - idx_start
            
            input_images, input_pids = [], []
            for i in range(idx_start, idx_end):
                image, metadata = self.reader.read_series(series_ids[i])
                input_images.append(image.data)
                input_pids.append(metadata["pid"])
            
            input_batch = torch.stack(input_images, dim=0).to("cuda")
            embeddings = self.encoder(input_batch).view(effective_batch_size, -1)
            
            for i, pid in enumerate(input_pids):
                patient_ids_embeddings[i + idx_start] = (pid, torch.unsqueeze(embeddings[i], 0))
                
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                similarity = cos(
                    patient_ids_embeddings[i][1], 
                    patient_ids_embeddings[j][1]
                ).cpu().numpy()[0]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        correct_count = 0
        
        # if log_similar_scans:
        for i in range(n):
            # sorted_by_similarity = np.argsort(similarity_matrix[i])[::-1]
            
            # index of most similar scan
            # pred_idx = sorted_by_similarity[0] 
            pred_idx = np.argmax(similarity_matrix[i])
            if patient_ids_embeddings[pred_idx][0] == patient_ids_embeddings[i][0]:
                correct_count += 1
                # self.series_correct_pred.append(series_ids[i])

            # for each scan, log the top 10 most similar scans
            # curr_series = series_ids[i]            
            # self.log_predictions[curr_series] = series_ids[sorted_by_similarity[:TOP_NO_SIMILAR]]

        accuracy = round(correct_count / n, 5)
               
        np.set_printoptions(precision=3) 
        print(f"Similarity matrix: \n{similarity_matrix}")
        print(f"Same patient prediction accuracy (n={n}): {accuracy * 100}%")
                
        return accuracy
    
    # TODO: Reverted similar scans retrieval due to errors

    # def get_most_similar_scan(self, series_id: SeriesID) -> SeriesID:
    #     '''Return series id of the top 1 similar scan.'''
        
    #     if series_id in self.log_predictions:
    #         return self.log_predictions[series_id][0]
        
    #     else:
    #         raise ValueError(f"No predictions available for series ID {series_id}.")

    # def get_top_N_similar_scans(self, series_id: SeriesID, N: int = 10) -> list[SeriesID]:
    #     '''Return a list top N similar scans.'''
        
    #     if series_id in self.log_predictions:
    #         if N > TOP_NO_SIMILAR:
    #             print(f"This log currently only records the top {TOP_NO_SIMILAR} similar scans.")
    #         elif N < 1:
    #             raise ValueError(f"N must be at least 1.")
    #         else:
    #             return self.log_predictions[series_id][:N]

    #     else:
    #         raise ValueError(f"No predictions available for series ID {series_id}.")
    
    # # TODO: get loggers to log after each epoch, what series ids were correctly identified. 
