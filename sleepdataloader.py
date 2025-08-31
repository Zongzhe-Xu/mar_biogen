# dataset.py
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Sequence, Optional, Tuple, Union, List, Dict

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from scipy.signal import resample

import time


class SleepEpochDataset(Dataset):
    """
    A PyTorch Dataset that delivers 30-second sleep epochs together
    with optional patient-level labels.

    Parameters
    ----------
    epoch_df : pd.DataFrame
        Epoch-level table. Must contain at least:
            - "nsrrid"       : patient ID
            - "epoch_id"     : 1-based epoch index
            - "path_head"    : prefix to the signal file on disk
    patient_df : pd.DataFrame | str | Path
        Patient-level table or the CSV file path; must contain
        "nsrrid" plus any downstream label columns.
    split : {"train", "val", "test"}
        Which split to use. Partitioning is done by patient ID
        so all epochs from the same patient stay in the same split.
    target_cols : str | Sequence[str] | None
        The patient-level label column(s) to return.  None = no labels
        (e.g. for self-supervised pre-training).
    test_size, val_size : float
        Fractions for patient-level train/val/test split.
    random_state : int
        Seed for deterministic splitting.
    sample_rate : int
        Sampling rate (Hz) of the pre-processed signal.
    cache_size : int
        LRU cache size for whole-night signals to avoid reloading.
    transform : callable | None
        Optional transform / augmentation applied to the epoch tensor.
    split_ids : dict | None
        Pre-defined {"train": [...], "val": [...], "test": [...]} lists.
        If supplied, overrides the random split.
    """

    def __init__(
        self,
        epoch_df: pd.DataFrame,
        patient_df: Union[pd.DataFrame, str, Path],
        split: str = "train",
        *,
        target_cols: Optional[Union[str, Sequence[str]]] = None,
        train_edf_cols = None,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 1337,
        sample_rate: int = 128,
        cache_size: int = 8,
        transform=None,
        split_ids: Optional[Dict[str, Sequence[str]]] = None,
    ):
        assert split in {"train", "val", "test"}
        self.transform = transform
        self.sample_rate = sample_rate
        self.target_cols = (
            [target_cols] if isinstance(target_cols, str) else target_cols
        )

        # ───── patient-level DataFrame ─────
        if isinstance(patient_df, (str, Path)):
            patient_df = pd.read_csv(patient_df)
        self.patient_df = patient_df.set_index("nsrrid")

        # ───── create / use patient splits ─────
        if split_ids is None:
            ids = self.patient_df.index.unique().tolist()

            train_ids, temp_ids = train_test_split(
                ids,
                test_size=(val_size + test_size),
                random_state=random_state,
            )
            rel_val = val_size / (val_size + test_size)
            val_ids, test_ids = train_test_split(
                temp_ids,
                test_size=1.0 - rel_val,
                random_state=random_state,
            )
            split_ids = {
                "train": train_ids,
                "val": val_ids,
                "test": test_ids,
            }

        self.split_ids = split_ids
        self.epoch_df = (
            epoch_df[epoch_df["nsrrid"].isin(split_ids[split])]
            .reset_index(drop=True)
        )

        # ───── LRU cache for whole-night signals ─────
        self._load_patient_array = lru_cache(maxsize=cache_size)(
            self._load_patient_array
        )
        self.train_edf_cols = train_edf_cols

    # ───────── Dataset API ─────────

    def __len__(self) -> int:
        return len(self.epoch_df)

    def __getitem__(self, idx: int):
        row = self.epoch_df.iloc[idx]
        nsrrid = row["nsrrid"]
        epoch_id = int(row["epoch_id"])

        # 1) load full-night signal and slice to this epoch
        t10 = time.perf_counter() 
        
        dfs = []
        for col_name in self.train_edf_cols:
            temp_sig = self._load_patient_array(nsrrid, row["path_head"], col_name = col_name)
            dfs.append(temp_sig)
        full_sig = pd.concat(dfs, axis = 1)
        t11 = time.perf_counter() 
        # print(f"time of get channel data {idx} row: {t11 - t10:.3f} s")
        
        t20 = time.perf_counter() 
        start_sec = (epoch_id - 1) * 30
        end_sec = epoch_id * 30
        epoch_sig = full_sig.loc[start_sec: end_sec]
        epoch_sig = epoch_sig.iloc[:-1] # remove the last point
        
        x = torch.tensor(epoch_sig.values, dtype=torch.float32)
        
        # 1.5) can add other transformation here
        if self.transform:
            x = self.transform(x)
        t21 = time.perf_counter() 
        # print(f"time of truncate and ransform data {idx} row: {t21 - t20:.3f} s")
        
        # 2) add patient-level label(s) if requested
        if self.target_cols:
            y = torch.tensor(
                self.patient_df.loc[nsrrid, self.target_cols].values.astype(float),
                dtype=torch.float32,
            )
            return x, y
        else:
            return x.permute(1,0), ""


    def _build_signal_path(self, path_head: str, col_name = 'ECG') -> Path:
        """
        Convert path_head to the actual signal file path.
        Default: '<path_head>_data.npz' with key 'signal'.
        Adjust if your filenames / keys differ.
        """
        return Path(path_head + f"_{col_name}.npz")

    def _load_patient_array(self, nsrrid: str, path_head: str, col_name = 'ECG') -> np.ndarray:
        """
        Load the full-night signal into a NumPy array.
        If you have multiple channels, return shape (C, T).
        """
        fp = self._build_signal_path(path_head, col_name)
        if not fp.is_file():
            raise FileNotFoundError(f"Signal file missing: {fp}")
        with np.load(fp, allow_pickle=False) as npz:
            data = npz['values']
            index = npz['index']

            df_stg = pd.DataFrame(
                data,
                columns=[col_name]
            )
            df_stg.insert(0, "sec", index)
            
            sig = df_stg.set_index("sec")           
            
        return sig.astype(np.float32)


    
    
    
    
    

import os
import time
import torch
import pandas as pd
from torch.utils.data import Dataset
from typing import List
import numpy as np
from tqdm import tqdm
from einops import rearrange
import wfdb
import itertools
from torch.utils.data import DataLoader

from pathlib import Path
SPLIT_DIR = Path("/projects/besp/shared_data/mimic-ecg-preprocessed/")
class ECG_Text_Dataset(Dataset):
    """ Dataset for MIMIC-IV-ECG"""
    def __init__(self, 
                 split: str, 
                 dataset_dir: str, 
                 dataset_list: List = ["mimic-iv-ecg"], 
                 data_pct: float = 1, 
                 transforms = None,
                 n_views: int = 1,
                 use_cmsc: bool = False,
                 use_rlm: bool = False,
                 num_beats: int = 1,
                 ):
        
        super().__init__()
        
        self.split = split
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.data_pct = data_pct
        self.use_cmsc = use_cmsc
        self.use_rlm = use_rlm
        self.n_views = n_views
        if transforms is None:
            self.augs = []
        else:
            self.augs = transforms
        # if self.use_rlm:
        #     # random mask 50% leads for each samplesa
        #     self.augs.append(
        #         RandomLeadsMask(p=1, mask_leads_selection="random", mask_leads_prob=0.5)
        #         )

        all_df = []
        for dataset_name in self.dataset_list:
            df = pd.read_csv(SPLIT_DIR / f"{dataset_name}/preprocessed_reports.csv", low_memory=False)
            
            df['path'] = ''
            df["path"] = df.apply(lambda x: os.path.join('files', 'p' + str(x['subject_id'])[:4], 'p' + str(x['subject_id']), 's' + str(x['study_id']), str(x['study_id'])),axis=1)
            df["path"] = df["path"].apply(lambda x: os.path.join(self.dataset_dir, dataset_name, x))
            print(f"Loading {dataset_name} {self.split} dataset: total {len(df)} samples")
            all_df.append(df)
        self.df = pd.concat(all_df)
        # sample data
        self.df = self.df.sample(frac=self.data_pct).reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = torch.tensor([row["subject_id"]]).long()
        report = row["total_report"]

        # 1) 读取 WFDB
        ecg = wfdb.rdsamp(row["path"])[0].T              # (leads, T)

        # 2) 检查并修复 NaN/Inf
        if not np.isfinite(ecg).all():
            # 对每个导联单独处理
            for i in range(ecg.shape[0]):
                bad = ~np.isfinite(ecg[i])
                if bad.any():
                    good = ~bad
                    if good.any():
                        # 用该导联的中位数填充坏点
                        fill = np.median(ecg[i, good])
                        ecg[i, bad] = fill
                    else:
                        # 整条导联全坏：置零（或选择丢样本/跳过）
                        ecg[i, :] = 0.0
        ecg = ecg[:, ::10] # 500hz to 50 hz
        
        # ecg = resample(ecg, 1024, axis=1, window='hann')
        # 3) 稳健的逐导联归一化到 [0,1]
        #    避免 (max-min)=0 造成的除零
        lead_min = ecg.min(axis=1, keepdims=True)
        lead_ptp = np.ptp(ecg, axis=1, keepdims=True)  # max-min
        ecg = (ecg - lead_min) / np.maximum(lead_ptp, 1e-8)

        # 4) 转 torch（避免多余拷贝）
        ecg = torch.from_numpy(ecg.astype(np.float32, copy=False))

        # ====== 后续增广不变 ======
        if self.use_cmsc:
            num_samples = ecg.size(1)
            ecg1 = ecg[:, :num_samples//2]
            ecg2 = ecg[:, num_samples//2:]
            for aug in self.augs:
                ecg1 = aug(ecg1)
                ecg2 = aug(ecg2)
            ecg = torch.stack([ecg1, ecg2], dim=0)
            patient_id = torch.cat([patient_id, patient_id], dim=0)
        else:
            if self.n_views == 1:
                for aug in self.augs:
                    ecg = aug(ecg)
            else:
                ecg_list = []
                for _ in range(self.n_views):
                    ecg_ = ecg.clone()
                    for aug in self.augs:
                        ecg_ = aug(ecg_)
                    ecg_list.append(ecg_)
                ecg = torch.stack(ecg_list, dim=0)
                patient_id = torch.cat([patient_id]*self.n_views, dim=0)
                
        if torch.isnan(ecg).any() or torch.isinf(ecg).any():
            raise ValueError(f"NaN/Inf detected after augmentations at idx={idx}")
        
        return ecg, 0