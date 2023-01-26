import os
import torch
import attrs
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from definitions import *
from io import TextIOWrapper
from data import NLSTDataReader
from typing import TextIO, Optional
from torch.nn import CosineSimilarity
from sklearn.metrics import roc_auc_score

EVAL_BATCH_SIZE = 12
SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE = 5
SIMILAR_LOG_SAMPLE_SIZE = 300

_LOG_DF_COLUMN_NAMES = ["PID", "SID", "CORRECT"]
for i in range(SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE):
    _LOG_DF_COLUMN_NAMES.extend([f"SMLR_{i}", f"SMLR_{i}_SCORE", f"SMLR_{i}_PID"])


@attrs.define()
class SamePatientEvaluator:
    encoder: torch.nn.Module
    reader: NLSTDataReader

    @staticmethod
    def clear_log_folder(model_name: str, version: int):
        log_file_dir = os.path.join(LOG_DIR, "same_patient", f"{model_name}_v{version}")
        if not os.path.exists(log_file_dir):
            return
        # Delete everything in the folder
        for file in os.listdir(log_file_dir):
            path = os.path.join(log_file_dir, file)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(e)

    @staticmethod
    def create_log_file(model_name: str, version: int, epoch: int) -> TextIO:
        log_file_dir = os.path.join(LOG_DIR, "same_patient", f"{model_name}_v{version}")
        os.makedirs(log_file_dir, exist_ok=True)

        log_file_path = os.path.join(log_file_dir, f"{epoch}.csv")
        file = open(log_file_path, "w")
        return file

    @torch.no_grad()
    def score(self, series_ids: list[SeriesID], log_file: Optional[TextIO] = None) -> float:
        """
        Evaluate the same patient prediction performance of the encoder. The encoder will be given 
        single 4D tensors (1, C, W, H, D).
        """

        n = len(series_ids)
        cos = CosineSimilarity(dim=1)

        similarity_matrix = np.zeros((n, n))
        true_similarity_matrix = np.zeros((n, n))
        patient_ids_embeddings: list[(PatientID, torch.Tensor)] = [None] * n

        # load in data series in batches
        for batch_num in tqdm(range(n // EVAL_BATCH_SIZE + 1),
                              desc="Encoding raw data for cosine similarity comparison"):
            idx_start = batch_num * EVAL_BATCH_SIZE
            idx_end = min(idx_start + EVAL_BATCH_SIZE, n)
            if idx_start == idx_end:
                continue

            effective_batch_size = idx_end - idx_start

            input_images, input_pids = [], []
            for i in range(idx_start, idx_end):
                image, metadata = self.reader.read_series(series_ids[i])
                input_images.append(image.data)
                input_pids.append(metadata["pid"])

            input_batch = torch.stack(input_images, dim=0).to("cuda").to(torch.float)
            embeddings = self.encoder(input_batch).view(effective_batch_size, -1)

            for i, pid in enumerate(input_pids):
                patient_ids_embeddings[i + idx_start] = (pid, torch.unsqueeze(embeddings[i], 0))

        auc_scores, correct_count = [], 0
        if log_file is not None:
            log_result_df = pd.DataFrame(columns=_LOG_DF_COLUMN_NAMES)
            log_result_sample_index_set = np.random.choice(
                np.arange(0, n),
                size=min(SIMILAR_LOG_SAMPLE_SIZE, n),
                replace=False
            )
            log_result_sample_index_set = set(log_result_sample_index_set)
        else:
            log_result_df = None
            log_result_sample_index_set = set()
        # create similarity matrix and true similarity matrix
        # (0 if two scans are not from the same patient, else 1)
        for i in range(n):
            for j in range(i + 1, n):
                similarity = abs(cos(
                    patient_ids_embeddings[i][1],
                    patient_ids_embeddings[j][1]
                ).cpu().numpy()[0])
                true_similarity = 1 if patient_ids_embeddings[i][0] == patient_ids_embeddings[j][0] else 0

                similarity_matrix[i, j], similarity_matrix[j, i] = similarity, similarity
                true_similarity_matrix[i, j], true_similarity_matrix[j, i] = true_similarity, true_similarity

            if i in log_result_sample_index_set:
                top_n_sorted_idx = np.argpartition(
                    similarity_matrix[i], -SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE)[-SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE:]
                # Negate the array to sort by descending order
                top_n_sorted_idx = top_n_sorted_idx[
                    np.argsort(-similarity_matrix[i, top_n_sorted_idx])
                ]
                pred_idx = top_n_sorted_idx[-1]
                correct = patient_ids_embeddings[pred_idx][0] == patient_ids_embeddings[i][0]
                if correct:
                    correct_count += 1
                row_data = [
                    patient_ids_embeddings[i][0],
                    series_ids[i],
                    1 if correct else 0
                ]
                for idx in top_n_sorted_idx:
                    row_data.extend([
                        series_ids[idx],
                        similarity_matrix[i, idx],
                        patient_ids_embeddings[idx][0]
                    ])
                row_df = pd.DataFrame([row_data], columns=_LOG_DF_COLUMN_NAMES)
                log_result_df = pd.concat([log_result_df, row_df])

            else:
                pred_idx = np.argmax(similarity_matrix[i])
                if patient_ids_embeddings[pred_idx][0] == patient_ids_embeddings[i][0]:
                    correct_count += 1

            # Note: There exist patients in the NLST dataset with only one scan
            # during the study period (for example, patient 100518). For those,
            # we need to ignore their rows in the AUC calculation.
            if len(np.unique(true_similarity_matrix[i])) > 1:
                auc_scores.append(
                    roc_auc_score(
                        true_similarity_matrix[i], similarity_matrix[i]
                    )
                )

        accuracy = round(correct_count / n, 5)
        average_auc = np.mean(auc_scores)

        np.set_printoptions(precision=4)
        print(f"Similarity matrix: \n{similarity_matrix}")
        print(f"Same patient prediction accuracy: {accuracy * 100}%")
        print(f"Average AUC: {average_auc}")

        if log_result_df is not None:
            log_result_df.to_csv(log_file, index=False)
            file_name = log_file.name
            file_path = os.path.abspath(file_name)
            print(f"Snippet of the similarity results logged to {file_path}")

        return accuracy
