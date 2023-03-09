import os

import comet_ml
import attrs
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from definitions import *
import torch.nn.functional
import pytorch_lightning as pl
from data import NLSTDataReader
from torch.nn import CosineSimilarity
from sklearn.metrics import roc_auc_score
from typing import TextIO, Optional, Union

SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE = 5
SIMILAR_LOG_SAMPLE_SIZE = 300

_LOG_DF_COLUMN_NAMES = ["PID", "SID", "CORRECT"]
for i in range(SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE):
    _LOG_DF_COLUMN_NAMES.extend([f"SMLR_{i}", f"SMLR_{i}_SCORE", f"SMLR_{i}_PID"])


@attrs.define()
class SamePatientEvaluator:
    experiment_name: str
    encoder: Union[pl.LightningModule, torch.nn.Module]
    reader: NLSTDataReader
    batch_size: int

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
    def score(self, series_ids: list[SeriesID], log_file: Optional[TextIO] = None) -> (float, float, float):
        """
        Evaluate the same patient prediction performance of the encoder. The encoder will be given 
        single 4D tensors (1, C, W, H, D).

        Return the top 1 and top 5 same patient accuracy and the average AUC score.
        """

        n = len(series_ids)
        cos = CosineSimilarity(dim=0)

        similarity_matrix = np.zeros((n, n))
        true_similarity_matrix = np.zeros((n, n))
        patient_ids: np.ndarray = np.zeros(n)
        series_ids = np.array(series_ids)
        embeddings: torch.Tensor = torch.zeros(0, 0)
        embeddings_output: list[torch.Tensor] = [torch.zeros(0, 0).to("cuda")] * n

        # load in data series in batches
        for batch_num in tqdm(range(n // self.batch_size + 1),
                              desc="Encoding raw data for cosine similarity comparison"):
            idx_start = batch_num * self.batch_size
            idx_end = min(idx_start + self.batch_size, n)
            if idx_start == idx_end:
                continue

            effective_batch_size = idx_end - idx_start

            input_images, input_pids = [], []
            for i in range(idx_start, idx_end):
                image, metadata = self.reader.read_series(series_ids[i])
                input_images.append(image.data)
                input_pids.append(metadata["pid"])

            input_batch = torch.stack(input_images, dim=0).to("cuda").to(torch.float)
            if isinstance(self.encoder, torch.nn.Module):
                emb = self.encoder(input_batch)
            elif isinstance(self.encoder, pl.LightningModule):
                emb = self.encoder.model(input_batch)
                emb = self.encoder.prediction_model(embeddings)
            else:
                raise ValueError(f"Unrecognized encoder, must be either a lighting module or a torch module")
            emb = emb.view(effective_batch_size, -1)
            emb = torch.nn.functional.normalize(emb, dim=1)
            for i, pid in enumerate(input_pids):
                patient_ids[i + idx_start] = pid
                embeddings_output[i + idx_start] = torch.unsqueeze(emb[i], dim=0)

        embeddings = torch.cat(embeddings_output, dim=0)

        auc_scores, top_1_correct_count, top_5_correct_count = [], 0, 0
        top_1percent_correct_count = 0
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

        # # create similarity matrix and true similarity matrix
        # # (0 if two scans are not from the same patient, else 1)
        similarity_matrix = torch.mm(embeddings, embeddings.T)

        for i in range(n):
            for j in range(i + 1, n):
                true_similarity = 1 if patient_ids[i] == patient_ids[j] else 0
                true_similarity_matrix[i, j], true_similarity_matrix[j, i] = true_similarity, true_similarity

        similarity_matrix = similarity_matrix.fill_diagonal_(0).cpu().numpy()
        np.fill_diagonal(true_similarity_matrix, 0)

        log_rows = []
        for i in range(n):
            # Note: There exist patients in the NLST dataset with only one scan
            # during the study period (for example, patient 100518). For those,
            # we need to ignore their rows in the calculation.
            if len(np.unique(true_similarity_matrix[i])) <= 1:
                top_5_correct_count += 1
                top_1_correct_count += 1
                top_1percent_correct_count += 1
                continue

            sorted_idx = np.argsort(-similarity_matrix[i])
            for j in range(5):
                idx = sorted_idx[j]
                if patient_ids[idx] == patient_ids[i]:
                    top_5_correct_count += 1
                    break
            for j in range(int(n * .01)):
                idx = sorted_idx[j]
                if patient_ids[idx] == patient_ids[i]:
                    top_1percent_correct_count += 1
                    break
            pred_idx = sorted_idx[0]
            if patient_ids[pred_idx] == patient_ids[i]:
                top_1_correct_count += 1

            head, tail = 200, 200
            observed_patient_ids = [str(val) for val in patient_ids[sorted_idx]]
            observed_patient_ids = observed_patient_ids[:head] + observed_patient_ids[-tail:]
            observed_scans = series_ids[sorted_idx]
            observed_scans = np.concatenate((observed_scans[:head], observed_scans[-tail:]))
            observed_scores = [str(val) for val in similarity_matrix[i][sorted_idx]]
            observed_scores = observed_scores[:head] + observed_scores[-tail:]

            # Scan ID | True Patient ID | Observed Patient IDs | Observed Scans | Observed Scores
            log_rows.append({
                "scan_id": series_ids[i],
                "true_patient_id": patient_ids[i],
                "observed_patient_ids": ", ".join(observed_patient_ids),
                "observed_scans": ", ".join(observed_scans),
                "observed_scores": ", ".join(observed_scores),
                "head": head,
                "tail": tail,
            })

            auc_scores.append(
                roc_auc_score(
                    true_similarity_matrix[i], similarity_matrix[i]
                )
            )

        log_df = pd.concat([pd.DataFrame(log_rows)], ignore_index=True)
        log_df = log_df.set_index("scan_id")

        log_file_dir = os.path.join(LOG_DIR, "same_patient", self.experiment_name)
        os.makedirs(log_file_dir, exist_ok=True)
        if hasattr(self.encoder, "current_epoch"):
            log_file_path = os.path.join(log_file_dir, f"epoch{self.encoder.current_epoch}.csv")
        else:
            log_file_path = os.path.join(log_file_dir, f"baseline.csv")
        log_df.to_csv(log_file_path)
        print(f"Raw evaluation results saved to {log_file_path}")

        top_1_accuracy = round(top_1_correct_count / n, 5)
        top_5_accuracy = round(top_5_correct_count / n, 5)
        top_1percent_accuracy = round(top_1percent_correct_count / n, 5)
        average_auc = np.mean(auc_scores)

        np.set_printoptions(precision=4)
        print(f"Similarity matrix: \n{similarity_matrix}")
        print(f"* Same patient top 1 accuracy: {top_1_accuracy * 100}%, top 5 accuracy: {top_5_accuracy * 100}%")
        print(f"* Same patient top 1% accuracy: {top_1percent_accuracy * 100}%")
        print(f"* Average AUC: {average_auc}")

        if log_result_df is not None:
            log_result_df.to_csv(log_file, index=False)
            file_name = log_file.name
            file_path = os.path.abspath(file_name)
            print(f"Snippet of the similarity results logged to {file_path}")

        return {
            "top1_accuracy": top_1_accuracy,
            "top5_accuracy": top_5_accuracy,
            "top1percent_accuracy": top_1percent_accuracy,
            "average_auc": average_auc,
        }
