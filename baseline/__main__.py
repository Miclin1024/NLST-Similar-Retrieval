import os
from data import NLSTDataReader
from . import ResNetSliceWiseEncoder, ResNetVideoEncoder
from evaluations import SamePatientEvaluator

print(f"---------------------------------------------")
reader = NLSTDataReader(
    manifest=int(os.environ.get("MANIFEST_ID")), 
    test_mode=True
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
    evaluator = SamePatientEvaluator(encoder=encoder, reader=reader)
    evaluator.score(reader.series_list)
    
print(f"---------------------------------------------")
reader = NLSTDataReader(
    manifest=int(os.environ.get("MANIFEST_ID")), 
    test_mode=False
)
print(f"Running baseline analysis (n={len(reader.patient_series_index)})")

for encoder in encoders:
    print(f"---------------------------------------------")
    print(f"Begin evaluating model: {encoder.description}")
    evaluator = SamePatientEvaluator(encoder=encoder, reader=reader)
    evaluator.score(reader.series_list)