import os
from evaluations import *
from data import NLSTDataReader
from . import ResNetSliceWiseEncoder, ResNetVideoEncoder

print(f"---------------------------------------------")
reader = NLSTDataReader(
    manifests=[1632928843386], head=200
)
print(f"Running baseline analysis (n={len(reader.patient_series_index)})")

encoders = [
    ResNetSliceWiseEncoder(pretrained=False),
    ResNetSliceWiseEncoder(pretrained=True),
    ResNetVideoEncoder(pretrained=False),
    ResNetVideoEncoder(pretrained=True),
]

for encoder in encoders:
    print(f"---------------------------------------------")
    print(f"Begin evaluating model: {encoder.description}")

    cancer_evaluator = ClassificationEvaluator(
        experiment_name=f"baseline-{encoder.name}",
        batch_size=8, encoder=encoder, reader=reader, target_key="confirmed_icd_topog1", ignore_nan=False
    )
    cancer_evaluator.score(reader.series_list)

    gender_evaluator = ClassificationEvaluator(
        experiment_name=f"baseline-{encoder.name}",
        batch_size=8, encoder=encoder, reader=reader, target_key="gender", ignore_nan=True
    )
    gender_evaluator.score(reader.series_list)

    weight_evaluator = RegressionEvaluator(
        experiment_name=f"baseline-{encoder.name}", batch_size=8,
        encoder=encoder, reader=reader, target_key="weight"
    )
    weight_evaluator.score(reader.series_list)

    sp_evaluator = SamePatientEvaluator(experiment_name=f"baseline-{encoder.name}",
                                        batch_size=8, encoder=encoder, reader=reader)
    sp_evaluator.score(reader.series_list)
