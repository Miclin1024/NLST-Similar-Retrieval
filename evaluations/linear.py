import attrs
import torch
from typing import Union
from sklearn import metrics
import pytorch_lightning as pl
from evaluations.base import *
from data import NLSTDataReader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression


@attrs.define()
class LinearEvaluator(Evaluator):

    target_key: str = attrs.field(default="weight")
    train_split_ratio: float = attrs.field(init=False, default=0.7)
    regression: bool = False

    @classmethod
    def from_pl_checkpoint(cls: Type[TEvaluator], hparams: ModelParams,
                           experiment: str, epoch: int) -> TEvaluator:
        return super().from_pl_checkpoint(
            hparams=hparams,
            experiment=experiment,
            epoch=epoch
        )

    def score(self, series_ids: list[SeriesID], log_file: Optional[TextIO] = None) -> dict:
        result = super().score(series_ids, log_file)

        n = len(series_ids)
        patient_ids = self._get_patient_ids(series_ids)
        patient_ids_set = set(patient_ids)
        exclude_set = set()
        for pid in patient_ids:
            metadata_row = self.metadata.loc[pid].to_dict()
            target = metadata_row[self.target_key]
            if np.isnan(target):
                exclude_set.add(pid)

        patient_ids = list(patient_ids_set.difference(exclude_set))

        # New series ids with only the earliest scan from a patient
        series_ids = [self.reader.patient_series_index[pid][0] for pid in patient_ids]
        n_updated = len(series_ids)
        print(f"{n - n_updated} scans ignored due to incomplete metadata")

        embeddings: np.ndarray = self._get_embeddings(series_ids).cpu().numpy()
        labels: np.ndarray = np.array([self.metadata.loc[pid].to_dict()[self.target_key] for pid in patient_ids])

        split = int(n_updated * self.train_split_ratio)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(embeddings[:split])
        x_test = scaler.transform(embeddings[split:])

        y_train = labels[:split]
        y_test = labels[split:]

        if self.regression:
            regressor = LinearRegression()
            regressor.fit(x_train, y_train)

            y_test_pred = regressor.predict(x_test)
            mae = metrics.mean_absolute_error(y_test, y_test_pred)
            r2 = metrics.r2_score(y_test, y_test_pred)
            np.set_printoptions(precision=4)
            print(f"* Linear model MSE for metadata key \"{self.target_key}\": {mse}")
            print(f"* Model R2: {r2}")
            return {
                "mae": mae,
                "r2": r2,
                **result,
            }
        else:
            classifier = LogisticRegression()
            classifier.fit(x_train, y_train)

            y_test_pred = classifier.predict(x_test)
            y_test_pred_prob = classifier.predict_proba(x_test)[:, 1]

            accuracy = metrics.accuracy_score(y_test, y_test_pred)
            auc_score = metrics.roc_auc_score(y_test, y_test_pred_prob)

            np.set_printoptions(precision=4)
            print(f"* Linear model accuracy for metadata key \"{self.target_key}\": {accuracy * 100}%")
            print(f"* Model AUC: {auc_score}")
            return {
                "accuracy": accuracy,
                "auc": auc_score,
                **result,
            }
