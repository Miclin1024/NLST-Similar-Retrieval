import os
import dotenv
from . import NLSTDataReader

if __name__ == '__main__':
    dotenv.load_dotenv()
    reader = NLSTDataReader(
        manifest=int(os.environ.get("MANIFEST_ID")),
        test_mode=False
    )
    reader.perform_preprocessing()
