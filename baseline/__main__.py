import os
from data import NLSTDataReader
from . import ResNetSliceWiseEncoder, ResNetVideoEncoder
from evaluations import SamePatientEvaluator

print(f"---------------------------------------------")
reader = NLSTDataReader(
    manifests=[1632928843386],
    head=30
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
    evaluator = SamePatientEvaluator(experiment_name=f"baseline-{encoder.name}", encoder=encoder,
                                     reader=reader, batch_size=8)
    evaluator.score(reader.series_list)
