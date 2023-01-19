
import os
import pandas as pd
from tqdm import tqdm
from shutil import copytree
from definitions import *
from data.reader import NLSTDataReader
from evaluations.same_patient import SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE


class OrganizeLog_3DSlicer_View():

    manifest: int 
    nlst_reader: NLSTDataReader

    def __init__(self, manifest):
        self.manifest = manifest
        self.nlst_reader = NLSTDataReader(manifest, test_mode = False)

    def organize_scans(self, log_table: pd.DataFrame, log_table_name: str):

        '''Organize scans as recorded from a log table into a folder to upload onto
        3D Slicer for viewing.'''
        
        for _, r in tqdm(log_table.iterrows(), desc=f"Copying files from log table {log_table_name}"):
            
            row = r.to_dict()
            col_names = ["SID"] + [f"SMLR_{i}" for i in range(SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE)]
            query_scan_id = row["SID"]

            for col in col_names:
                id = row[col]
            
                manifest_row = self.nlst_reader.manifest.loc[id].to_dict()
                path = manifest_row["File Location"]
                series_folder = os.path.join(self.nlst_reader.manifest_folder, path)

                # folder structure is as below
                #  >> SID of query scan
                #       >> [query_scan]_scanid
                #       >> [SMLR_0]_scanid
                #       >> [SMLR_1]_scanid
                #       >> ...

                prefix = "[query_scan]" if col == "SID" else f"[{col}]"
                new_loc = os.path.join(ROOT_DIR, "data", "3D_slicer", log_table_name, query_scan_id, f"{prefix}_{id}")
                copytree(src=series_folder, dst=new_loc)
    
    def return_pid_logs(self, log_table: pd.DataFrame) -> pd.DataFrame:
        '''Given a log table, return the pid of all of the scans in the log table.'''

        col_names = [f"SMLR_{i}" for i in range(SIMILAR_LOG_TOP_SCAN_RETRIEVAL_SIZE)]
        pid_dataframe = {f"{column}_pid":[] for column in col_names}

        for _, r in log_table.iterrows():
            
            row = r.to_dict()
            for col in col_names:
                series_id = row[col]
                pid_dataframe[f"{col}_pid"].append(self.nlst_reader.manifest.loc[series_id]["Subject ID"])

        pid_dataframe["query_pid"] = log_table["PID"].values
        return pd.DataFrame(pid_dataframe)

if __name__ == '__main__':

    manifest = os.environ.get("MANIFEST_ID")
    organizer = OrganizeLog_3DSlicer_View(manifest)

    models = ["resnet_2d_random_v0", "r3d_18_random_v0", "r3d_18_pretrained_v0"]
    
    # for i, model in enumerate(models):
    #     log = pd.read_csv(os.path.join(ROOT_DIR,f"logs/same_patient/{models[i]}/0.csv"))

    #     organizer.nlst_reader.manifest.loc[id]
        # organizer.organize_scans(log_table=log, log_table_name=model)


    log = pd.read_csv(os.path.join(ROOT_DIR,f"logs/same_patient/{models[2]}/0.csv"))
    print(organizer.return_pid_logs(log))