import attrs
import numpy as np
import torch
from typing import Union
from sklearn import metrics
import pytorch_lightning as pl
from evaluations.base import *
from data import NLSTDataReader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression, LinearRegression


@attrs.define()
class LinearEvaluator(Evaluator):

    target_key: str = attrs.field(default="weight")
    train_split_ratio: float = attrs.field(init=False, default=0.7)

    @classmethod
    def from_pl_checkpoint(cls: Type[TEvaluator], hparams: ModelParams,
                           experiment: str, epoch: int) -> TEvaluator:
        return super().from_pl_checkpoint(
            hparams=hparams,
            experiment=experiment,
            epoch=epoch
        )

    def _filter_pids_with_nans(self, patient_ids: np.ndarray) -> np.ndarray:
        patient_ids_set = set(patient_ids)
        exclude_set = set()
        for pid in patient_ids:
            metadata_row = self.metadata.loc[pid].to_dict()
            target = metadata_row[self.target_key]
            if np.isnan(target):
                exclude_set.add(pid)
                continue

        patient_ids = list(patient_ids_set.difference(exclude_set))
        print(f"{len(exclude_set)} patients ignored due to empty target label values")

        return np.array(patient_ids)

    def _embeddings_for_pid(self, patient_ids: np.ndarray) -> torch.Tensor:
        # New series ids with only the earliest scan from a patient
        series_ids = [self.reader.patient_series_index[pid][0] for pid in patient_ids]
        n_updated = len(series_ids)
        print(f"Using first scan only, {n_updated} scans evaluated")

        return self._get_embeddings(series_ids)


@attrs.define()
class ClassificationEvaluator(LinearEvaluator):

    ignore_nan: bool = True

    def score(self, series_ids: list[SeriesID], log_file: Optional[TextIO] = None) -> dict:
        result = super().score(series_ids, log_file)

        n = len(series_ids)
        patient_ids = self._get_patient_ids(series_ids)
        if self.ignore_nan:
            patient_ids = self._filter_pids_with_nans(patient_ids)
        embeddings: np.ndarray = self._embeddings_for_pid(patient_ids).cpu().numpy()

        labels = []
        for pid in patient_ids:
            value = self.metadata.loc[pid].to_dict()[self.target_key]
            if value is None:
                assert not self.ignore_nan, "Something is wrong, ignore nan enabled but " \
                                            "still encountered empty value"
                labels.append("missing_value")
            else:
                labels.append(f"{value}")
        labels = np.array(labels)

        split = int(len(patient_ids) * self.train_split_ratio)
        y_train, y_test = labels[:split], labels[split:]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(embeddings[:split])
        x_test = scaler.transform(embeddings[split:])

        num_class = len(np.unique(labels))
        assert num_class > 1, "Only one class in labels"
        is_multi_class: bool = num_class > 2

        if is_multi_class:
            train_classes = np.unique(y_train)
            for i, value in enumerate(y_test):
                if value not in train_classes:
                    y_test[i] = "missing_value"

            binarizer = LabelBinarizer()
            binarizer.fit(y_train)
            y_test_binarize = binarizer.transform(y_test)

            classifier = LogisticRegression(multi_class="ovr")
            classifier.fit(x_train, y_train)

            y_test_pred = classifier.predict(x_test)
            y_test_pred_prob = classifier.predict_proba(x_test)

            accuracy = metrics.accuracy_score(y_test, y_test_pred)
            auc_scores = []
            for i in range(len(binarizer.classes_)):
                y_true = y_test_binarize[:, i]
                if len(np.unique(y_true)) > 1:
                    auc_scores.append(metrics.roc_auc_score(y_true, y_test_pred_prob[:, i]))

            weights = np.bincount(y_test) / len(y_test)
            auc_score = np.dot(auc_scores, weights)

            np.set_printoptions(precision=4)
            print(f"* Linear classifier accuracy for multi-class metadata key \"{self.target_key}\": {accuracy * 100}%")
            print(f"* Weighted-average AUC for {len(auc_scores)} out of {num_class} classes: {auc_score}")

        else:
            classifier = LogisticRegression()
            classifier.fit(x_train, y_train)

            y_test_pred = classifier.predict(x_test)
            y_test_pred_prob = classifier.predict_proba(x_test)[:, 1]

            accuracy = metrics.accuracy_score(y_test, y_test_pred)
            auc_score = metrics.roc_auc_score(y_test, y_test_pred_prob)

            np.set_printoptions(precision=4)
            print(f"* Linear classifier accuracy for binary metadata key \"{self.target_key}\": {accuracy * 100}%")
            print(f"* AUC: {auc_score}")

        return {
            "accuracy": accuracy,
            "auc": auc_score,
            **result,
        }


@attrs.define()
class RegressionEvaluator(LinearEvaluator):

    def score(self, series_ids: list[SeriesID], log_file: Optional[TextIO] = None) -> dict:
        result = super().score(series_ids, log_file)

        n = len(series_ids)
        patient_ids = self._get_patient_ids(series_ids)
        patient_ids = self._filter_pids_with_nans(patient_ids)
        embeddings = self._embeddings_for_pid(patient_ids).cpu().numpy()

        labels = np.array([self.metadata.loc[pid].to_dict()[self.target_key] for pid in patient_ids])

        split = int(len(patient_ids) * self.train_split_ratio)
        y_train, y_test = labels[:split], labels[split:]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(embeddings[:split])
        x_test = scaler.transform(embeddings[split:])

        regressor = LinearRegression()
        regressor.fit(x_train, y_train)

        y_test_pred = regressor.predict(x_test)
        mae = metrics.mean_absolute_error(y_test, y_test_pred)
        r2 = metrics.r2_score(y_test, y_test_pred)
        np.set_printoptions(precision=4)
        print(f"* Linear model MAE for metadata key \"{self.target_key}\": {mae}")
        print(f"* Model R2: {r2}")
        return {
            "mae": mae,
            "r2": r2,
            **result,
        }
