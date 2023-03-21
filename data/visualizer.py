import os
import cmd
import shutil
import readline

import paramiko
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from definitions import *
from typing import Optional
from data.reader import NLSTDataReader

viewport = os.path.join(LOG_DIR, "viewport")


class LogVisualizer(cmd.Cmd):
    intro = """    __               _    ___                  ___                
   / /   ____  ____ | |  / (_)______  ______ _/ (_)___  ___  _____
  / /   / __ \/ __ `/ | / / / ___/ / / / __ `/ / /_  / / _ \/ ___/
 / /___/ /_/ / /_/ /| |/ / (__  ) /_/ / /_/ / / / / /_/  __/ /    
/_____/\____/\__, / |___/_/____/\__,_/\__,_/_/_/ /___/\___/_/     
            /____/                                                
"""

    eval_module: Optional[str] = "same_patient"
    experiment_name: Optional[str] = "r3d_18_moco_v84"
    checkpoint_name: Optional[str] = "epoch5"
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
        true_indices = np.nonzero(np.array(observed_patient_ids) == true_pid)[0]
        print(f"")
        print(f"*** Evaluation report for {scan_id} ***")
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
        for index in true_indices:
            print(f"* [Same Patient Scan]: {observed_scans[index]} \n"
                  f"\t[Rank]: {index} \n"
                  f"\t[Score]: {observed_scores[index]} \n"
                  f"\t[Path]: {self.reader.original_folder(observed_scans[index])}")

        item_path = os.path.join(viewport, f"[{true_pid}]-{self.reader.scan_year(scan_id)}")
        if not os.path.exists(item_path):
            shutil.copytree(self.reader.original_folder(scan_id),
                            os.path.join(item_path, f"QUERY[{true_pid}]"))
            for i in range(4):
                shutil.copytree(self.reader.original_folder(observed_scans[0]),
                                os.path.join(item_path,
                                             f"SMLR_{i + 1}[{observed_patient_ids[i]}][{observed_scores[i]}]"))

            for i, index in enumerate(true_indices):
                shutil.copytree(
                    self.reader.original_folder(observed_scans[index]),
                    os.path.join(item_path, f"TRUE_{i}[{observed_patient_ids[index]}][{observed_scores[index]}]")
                )

            for i in range(2):
                shutil.copytree(self.reader.original_folder(observed_scans[-2]),
                                os.path.join(item_path,
                                             f"LEAST_{i + 1}[{observed_patient_ids[-i-2]}][{observed_scores[-i-2]}]"))

        print(f"* Raw data loaded to {os.path.abspath(item_path)}")
        print(f"*** Evaluation report end ***")
        print(f"")

    def do_report(self, arg):
        assert self.current_experiment_df is not None, "No log data loaded yet."

        n = 20
        samples = self.current_experiment_df.sample(n)
        print(f"Generating report for {self.experiment_name}->{self.checkpoint_name}, {n} scans sampled.")
        report_dir = os.path.join(LOG_DIR, "reports", f"{self.experiment_name}-{self.checkpoint_name}")
        if os.path.exists(report_dir):
            print("Report exists for the current checkpoint, overwriting.")
            shutil.rmtree(report_dir)
        os.makedirs(report_dir)

        for scan_id, experiment_row in tqdm(samples.iterrows(), f"Moving scans to the report folder"):
            scan_id, experiment_row = str(scan_id), experiment_row.to_dict()

            observed_scans = experiment_row["observed_scans"].split(", ")
            observed_patient_ids = [round(float(val)) for val in experiment_row["observed_patient_ids"].split(", ")]
            observed_scores = [float(val) for val in experiment_row["observed_scores"].split(", ")]
            true_pid = int(experiment_row['true_patient_id'])
            true_indices = np.nonzero(np.array(observed_patient_ids) == true_pid)[0]

            item_path = os.path.join(report_dir, f"[{true_pid}]-{self.reader.scan_year(scan_id)}")
            if not os.path.exists(item_path):
                shutil.copytree(
                    self.reader.original_folder(scan_id),
                    os.path.join(item_path, f"QUERY[{true_pid}]")
                )

                for i in range(20):
                    shutil.copytree(
                        self.reader.original_folder(observed_scans[0]),
                        os.path.join(
                            item_path,
                            f"SMLR_{i + 1}[{observed_patient_ids[i]}][{observed_scores[i]}]"
                        )
                    )

                for i, index in enumerate(true_indices):
                    shutil.copytree(
                        self.reader.original_folder(observed_scans[index]),
                        os.path.join(
                            item_path,
                            f"TRUE_{i}[{observed_patient_ids[index]}][{observed_scores[index]}]"
                        )
                    )

                for i in range(5):
                    shutil.copytree(
                        self.reader.original_folder(observed_scans[-2]),
                        os.path.join(
                            item_path,
                            f"LEAST_{i + 1}[{observed_patient_ids[-i - 2]}][{observed_scores[-i - 2]}]"
                        )
                    )

        print(f"Report generated at {os.path.abspath(report_dir)}.")

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
