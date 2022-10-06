from data import NLSTDataReader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class NLSTDataset(Dataset):
    def __init__(self, manifest: int, train: bool):
        self.reader = NLSTDataReader(manifest)
        self.key_list: list[str] = []
        for _, v in self.reader.read_all_patients().items():
            self.key_list.extend(v.values())

        split_result = train_test_split(self.key_list, test_size=0.33, random_state=1)
        self.key_list = split_result[0] if train else split_result[1]

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, item):
        uid = self.key_list[item]
        return self.reader.read_uid(uid)
