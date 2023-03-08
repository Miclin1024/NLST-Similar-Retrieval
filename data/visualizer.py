import os
import cmd
import shutil
import readline
import paramiko
import traceback
import pandas as pd
from definitions import *
from typing import Optional
from data.reader import NLSTDataReader

class LogVisualizer(cmd.Cmd):
    intro = """    __               _    ___                  ___                
   / /   ____  ____ | |  / (_)______  ______ _/ (_)___  ___  _____
  / /   / __ \/ __ `/ | / / / ___/ / / / __ `/ / /_  / / _ \/ ___/
 / /___/ /_/ / /_/ /| |/ / (__  ) /_/ / /_/ / / / / /_/  __/ /    
/_____/\____/\__, / |___/_/____/\__,_/\__,_/_/_/ /___/\___/_/     
            /____/                                                
"""

    eval_module: Optional[str] = "same_patient"
    experiment_name: Optional[str] = "r3d_18_moco_v66"
    checkpoint_name: Optional[str] = "epoch20"
    current_experiment_df: Optional[pd.DataFrame] = None
    reader = NLSTDataReader(manifests=list(map(lambda elem: int(elem), os.environ.get("MANIFEST_ID").split(","))))

    def __init__(self):
        super().__init__()
        self._try_load()

    @property
    def prompt(self):
        result = ""
        if self.eval_module is None:
            return result + "$ "
        else:
            result += f"({self.eval_module}"
        if self.experiment_name is None:
            return result + ") $ "
        else:
            result += f"->{self.experiment_name}"
        if self.checkpoint_name is None:
            return result + ") $ "
        else:
            result += f"/{self.checkpoint_name}"
        return result + ") $ "

    def do_eval(self, arg):
        self.eval_module = arg
        self._try_load()

    def do_experiment(self, arg):
        self.experiment_name = arg
        self._try_load()

    def do_checkpoint(self, arg):
        self.checkpoint_name = arg
        self._try_load()

    @staticmethod
    def do_clear_viewport(arg):
        viewport = os.path.join(LOG_DIR, "viewport")
        count = 0
        for filename in os.listdir(viewport):
            file_path = os.path.join(viewport, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    count += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    count += 1
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        print(f"{viewport} cleared, {count} items deleted.")

    def _try_load(self):
        def format_size(size):
            power = 2 ** 10
            n = 0
            units = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
            while size > power:
                size /= power
                n += 1
            return f'{size:.2f}{units[n]}'

        if self.eval_module and self.experiment_name and self.checkpoint_name:
            checkpoint_path = os.path.join(self.eval_module, self.experiment_name, f"{self.checkpoint_name}.csv")
            path = os.path.join(LOG_DIR, checkpoint_path)

            file_size = os.path.getsize(path)
            print(f"{path} ({format_size(file_size)}) loaded.")
            try:
                self.current_experiment_df = pd.read_csv(path)
                self.current_experiment_df.set_index("scan_id", inplace=True)
            except Exception as e:
                print(f"An error occurred while reading the log data: {e}")

    def do_view(self, arg):
        args = parse(arg)
        scan_id = args[0]
        assert self.current_experiment_df is not None, "No log data loaded yet."

        experiment_row = self.current_experiment_df.loc[scan_id].to_dict()
        observed_scans = experiment_row["observed_scans"].split(", ")
        observed_patient_ids = [round(float(val)) for val in experiment_row["observed_patient_ids"].split(", ")]
        observed_scores = [float(val) for val in experiment_row["observed_scores"].split(", ")]

        true_pid = int(experiment_row['true_patient_id'])
        true_index = observed_patient_ids.index(true_pid)

        print(f"*** Evaluation Report for {scan_id} ***")
        print(f"* [Series ID]: {scan_id}")
        print(f"* [Patient ID]: {true_pid}")
        print(f"* [Path]: {self.reader.original_folder(scan_id)}")
        print(f"* [Most Similar Scan]: {observed_scans[0]} \n"
              f"\t[Patient ID]: {observed_patient_ids[0]} \n"
              f"\t[Score]: {observed_scores[0]} \n"
              f"\t[Path]: {self.reader.original_folder(observed_scans[0])}")
        print(f"* [Second Most Similar Scan]: {observed_scans[1]} \n"
              f"\t[Patient ID]: {observed_patient_ids[1]} \n"
              f"\t[Score]: {observed_scores[1]} \n"
              f"\t[Path]: {self.reader.original_folder(observed_scans[1])}")
        print(f"* [Least Similar Scan]: {observed_scans[-2]} \n"
              f"\t[Patient ID]: {observed_patient_ids[-2]} \n"
              f"\t[Score]: {observed_scores[-2]} \n"
              f"\t[Path]: {self.reader.original_folder(observed_scans[-2])}")
        print(f"* [Closest Same Patient Scan]: {observed_scans[true_index]} \n"
              f"\t[Rank]: {true_index} \n"
              f"\t[Score]: {observed_scores[true_index]} \n"
              f"\t[Path]: {self.reader.original_folder(observed_scans[true_index])}")

        viewport = os.path.join(LOG_DIR, "viewport")
        item_path = os.path.join(viewport, scan_id)
        if not os.path.exists(item_path):
            shutil.copytree(self.reader.original_folder(scan_id),
                            os.path.join(item_path, f"QUERY[{experiment_row['true_patient_id']}]"))
            shutil.copytree(self.reader.original_folder(observed_scans[0]),
                            os.path.join(item_path, f"SMLR_1[{observed_patient_ids[0]}][{observed_scores[0]}]"))
            shutil.copytree(self.reader.original_folder(observed_scans[1]),
                            os.path.join(item_path, f"SMLR_2[{observed_patient_ids[1]}][{observed_scores[1]}]"))
            shutil.copytree(
                self.reader.original_folder(observed_scans[true_index]),
                os.path.join(item_path, f"TRUE_1[{observed_patient_ids[true_index]}][{observed_scores[true_index]}]")
            )
            shutil.copytree(self.reader.original_folder(observed_scans[-2]),
                            os.path.join(item_path, f"LEAST_1[{observed_patient_ids[-2]}][{observed_scores[-2]}]"))

        print(f"* Raw data loaded to {os.path.abspath(item_path)}")


    def onecmd(self, line):
        try:
            return super().onecmd(line)
        except Exception as e:
            print(traceback.format_exc())
            return False  # don't stop


def parse(arg):
    return tuple(arg.split())


if __name__ == '__main__':
    LogVisualizer().cmdloop()
