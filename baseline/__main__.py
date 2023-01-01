import os
from data import NLSTDataReader
from . import ResNetSliceWiseEncoder
from evaluations import SamePatientEvaluator

reader = NLSTDataReader(
    manifest=int(os.environ.get("MANIFEST_ID")), 
    test_mode=True
)

encoder = ResNetSliceWiseEncoder()
evaluator = SamePatientEvaluator(encoder=encoder, reader=reader)
evaluator.score(reader.series_list)