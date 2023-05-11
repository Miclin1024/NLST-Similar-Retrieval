from tqdm import tqdm
from data.reader import *
from inference.base import _AdapterBase

index_dir = os.path.join(ROOT_DIR, "inference", "index")


class _IndexBuilderMixin(_AdapterBase):

    index: dict[SeriesID, np.ndarray] = attrs.field(init=False, default={})

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def model_index_dir(self) -> str:
        path = os.path.join(index_dir, self.name)
        os.makedirs(path, exist_ok=True)
        return path

    def load_index(self):
        self.index = {}
        model_index_dir = self.model_index_dir
        for file in os.listdir(model_index_dir):
            path = os.path.join(model_index_dir, file)
            index = np.load(path, allow_pickle=True).item()
            assert isinstance(index, dict)
            self.index.update(index)
        assert len(self.index) == len(self.reader.series_list)
        print(f"Index loaded, {len(self.index)} series found.")

    def build_index_restore_point(self) -> int:
        """
        The index saved in persistent storage is divided into frames. This allows us to quickly
        recover when `build_index()` is interrupted. Consequently, we need a way of figuring out
        where to restart. This method is used by `build_index()` and returns the last idx of the
        consecutive index cache from the beginning. Also, we choose to also have it clean up all
        the non-consecutive index pieces afterwards for simplicity.
        """
        model_index_dir = self.model_index_dir
        idx_mapping = {}
        for file_path in os.listdir(model_index_dir):
            file_name: str = os.path.basename(file_path)
            file_name, _ = os.path.splitext(file_name)
            idx_mapping[int(file_name.split("-")[0])] = int(file_name.split("-")[1])

        start = 0
        while start in idx_mapping.keys():
            start = idx_mapping[start]

        path_to_remove = []
        for file_path in os.listdir(model_index_dir):
            file_name: str = os.path.basename(file_path)
            file_name, _ = os.path.splitext(file_name)
            if int(file_name.split("-")[0]) > start:
                path_to_remove.append(file_path)

        for path in path_to_remove:
            os.unlink(path)

        return start

    def build_index(self, batch_size: int = 32, flush_interval: int = 256):
        assert flush_interval % batch_size == 0

        series_list = self.reader.series_list
        n = len(series_list)

        index: dict[SeriesID, np.ndarray] = {}
        model_index_dir = self.model_index_dir
        starting_point = self.build_index_restore_point()
        if starting_point != 0:
            print(f"Restarting from index #{starting_point}")

        current_frame_start = 0
        for batch_num in tqdm(range(n // batch_size + 1), desc="Building inference index"):
            idx_start = batch_num * batch_size
            idx_end = min(idx_start + batch_size, n)
            if idx_end < starting_point:
                continue

            effective_batch_size = idx_end - idx_start
            if effective_batch_size == 0:
                continue

            input_images = [self.reader.read_series_idx(i)[0].data for i in range(idx_start, idx_end)]
            with torch.no_grad():
                input_batch = torch.stack(input_images, dim=0).to("cuda").to(torch.float)
                output_conv = self.conv(input_batch)
                output_conv = torch.nn.functional.normalize(output_conv, dim=0)
                output_conv = output_conv.cpu().numpy()
                for i in range(effective_batch_size):
                    index[self.reader.series_list[idx_start + i]] = output_conv[i]

            if len(index) >= flush_interval:
                index_part_name = f"{current_frame_start}-{idx_end}.npy"
                current_frame_start = idx_end
                index_part_path = os.path.join(model_index_dir, index_part_name)
                np.save(index_part_path, index)
                index = {}

        index_part_name = f"{current_frame_start}-{n}.npy"
        index_part_path = os.path.join(model_index_dir, index_part_name)
        np.save(index_part_path, index)
