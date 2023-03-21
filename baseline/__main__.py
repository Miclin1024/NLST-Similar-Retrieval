import os
from data import NLSTDataReader
from . import ResNetSliceWiseEncoder, ResNetVideoEncoder
from evaluations import SamePatientEvaluator, LinearEvaluator

print(f"---------------------------------------------")
reader = NLSTDataReader(
    manifests=[1632928843386]
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
    gender_evaluator = LinearEvaluator(experiment_name=f"baseline-{encoder.name}",
                                       batch_size=8, encoder=encoder, reader=reader)
    gender_evaluator.target_key = "gender"
    gender_evaluator.score(reader.series_list)

    sp_evaluator = SamePatientEvaluator(experiment_name=f"baseline-{encoder.name}",
                                        batch_size=8, encoder=encoder, reader=reader)
    sp_evaluator.score(reader.series_list)
