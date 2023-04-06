import os
import dotenv
from . import NLSTDataReader

if __name__ == '__main__':
    dotenv.load_dotenv()
    reader = NLSTDataReader(
        manifests=[1632928843386, 1632927888500, 1632929488567, 1632930131404, 1632933130941,
                   1632940796775, 1632960932512, 1632961568183],
        # manifests=[1632928843386, 1632927888500, 1632929488567, 1632930131404, 1632933130941],
        default_access_mode="direct",
    )
    reader.perform_preprocessing()
