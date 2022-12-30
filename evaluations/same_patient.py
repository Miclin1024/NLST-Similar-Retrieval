import torch
import attrs
import numpy as np
from tqdm import tqdm
from definitions import *
from data import NLSTDataReader
from torch.nn import CosineSimilarity


@attrs.define()
class SamePatientEvaluator:
    
    encoder: torch.nn.Module
    reader: NLSTDataReader
    
    @torch.no_grad()
    def score(self, series_ids: list[SeriesID]):
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
        
        for i in tqdm(range(n), desc="Encoding raw data for cosine similarity comparison"):
            load_encode_series(i)
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                similarity = cos(
                    patient_ids_embeddings[i][1], 
                    patient_ids_embeddings[j][1]
                ).cpu().numpy()[0]
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
        
        correct_count = 0
        for i in range(n):
            pred = np.argmax(similarity_matrix[i])
            pred = patient_ids_embeddings[pred][0]
            if pred == patient_ids_embeddings[i][0]:
                correct_count += 1
                
        accuracy = round(correct_count / n, 5)
               
        np.set_printoptions(precision=3) 
        print(f"Similarity matrix: {similarity_matrix}")
        print(f"Same patient prediction accuracy (n={n}): {accuracy * 100}%")
                
        return accuracy
