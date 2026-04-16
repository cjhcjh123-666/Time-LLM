import os
import numpy as np
from torch.utils.data import Dataset

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4


FORECAST_DATASETS = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'ECL': Dataset_Custom,
    'Traffic': Dataset_Custom,
    'Weather': Dataset_Custom,
    'm4': Dataset_M4,
}


class ForecastTaskDataset(Dataset):
    """
    Wrap existing forecasting datasets and expose a unified multitask sample format.
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base_dataset[index]
        return {
            'task_name': 'forecast',
            'x': seq_x,
            'y': seq_y,
            'x_mark': seq_x_mark,
            'y_mark': seq_y_mark,
            'label': np.array([-1], dtype=np.int64),
            'obs_mask': np.ones_like(seq_x, dtype=np.float32),
            'miss_mask': np.zeros_like(seq_x, dtype=np.float32),
        }


class ClassificationTaskDataset(Dataset):
    """
    Lightweight .ts classification loader for UCR/UAE style datasets.
    It supports the common format:
      <dimension_1_values>:<dimension_2_values>:...:<label>
    """

    def __init__(self, args, flag='train'):
        assert flag in ['train', 'val', 'test']
        self.args = args
        self.flag = flag
        self.seq_len = args.seq_len
        self.samples, self.labels = self._load_split()
        self.label_to_id = self._build_label_map(self.labels)

    @staticmethod
    def _build_label_map(labels):
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}

    def _resolve_ts_path(self):
        suffix_map = {'train': 'TRAIN', 'val': 'TRAIN', 'test': 'TEST'}
        split_suffix = suffix_map[self.flag]

        # Case 1: user passes a direct .ts path via --data_path
        if isinstance(self.args.data_path, str) and self.args.data_path.endswith('.ts'):
            direct_path = os.path.join(self.args.root_path, self.args.data_path)
            if os.path.exists(direct_path):
                return direct_path

        # Case 2: <root>/<dataset>/<dataset>_{TRAIN,TEST}.ts
        candidates = [
            os.path.join(self.args.root_path, self.args.data, f'{self.args.data}_{split_suffix}.ts'),
            os.path.join(self.args.root_path, 'UCR', self.args.data, f'{self.args.data}_{split_suffix}.ts'),
            os.path.join(self.args.root_path, 'UAE', self.args.data, f'{self.args.data}_{split_suffix}.ts'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError('Cannot locate classification .ts file for dataset {}'.format(self.args.data))

    @staticmethod
    def _parse_dimension(dim_text):
        values = []
        for token in dim_text.split(','):
            token = token.strip()
            if token == '' or token == '?':
                values.append(np.nan)
            else:
                values.append(float(token))
        return values

    def _load_split(self):
        file_path = self._resolve_ts_path()
        samples, labels = [], []
        in_data = False

        with open(file_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if line == '':
                    continue
                if line.lower().startswith('@data'):
                    in_data = True
                    continue
                if not in_data or line.startswith('@'):
                    continue

                parts = line.split(':')
                if len(parts) < 2:
                    continue

                label = parts[-1].strip()
                dims = [self._parse_dimension(part) for part in parts[:-1]]
                min_len = min(len(dim) for dim in dims)
                if min_len == 0:
                    continue
                dims = [dim[:min_len] for dim in dims]
                x = np.array(dims, dtype=np.float32).T  # [T, C]
                samples.append(x)
                labels.append(label)

        if len(samples) == 0:
            raise ValueError('No valid samples parsed from {}'.format(file_path))
        return samples, labels

    def _fit_to_seq_len(self, x):
        t, c = x.shape
        if t == self.seq_len:
            return x
        if t > self.seq_len:
            return x[-self.seq_len:, :]
        out = np.zeros((self.seq_len, c), dtype=np.float32)
        out[-t:, :] = x
        return out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self._fit_to_seq_len(self.samples[index])
        label = self.label_to_id[self.labels[index]]
        # classification has no timestamp features; keep a placeholder shape
        x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        y_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
        return {
            'task_name': 'classification',
            'x': x,
            'y': x.copy(),
            'x_mark': x_mark,
            'y_mark': y_mark,
            'label': np.array([label], dtype=np.int64),
            'obs_mask': np.ones_like(x, dtype=np.float32),
            'miss_mask': np.zeros_like(x, dtype=np.float32),
        }


class AnomalyTaskDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('AnomalyTaskDataset will be added in next incremental change.')


class ImputationTaskDataset(Dataset):
    """
    Build imputation samples by applying synthetic masks on forecasting windows.
    """
    def __init__(self, base_dataset, mask_rate=0.2, mask_mode='random'):
        self.base_dataset = base_dataset
        self.mask_rate = mask_rate
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.base_dataset)

    def _build_masks(self, x):
        obs_mask = np.ones_like(x, dtype=np.float32)
        miss_mask = np.zeros_like(x, dtype=np.float32)
        t, c = x.shape
        total = t * c
        miss_count = max(1, int(total * self.mask_rate))

        if self.mask_mode == 'block':
            block_len = max(1, int(t * self.mask_rate))
            start = np.random.randint(0, max(1, t - block_len + 1))
            miss_mask[start:start + block_len, :] = 1.0
        else:
            flat_idx = np.random.choice(total, miss_count, replace=False)
            miss_mask.reshape(-1)[flat_idx] = 1.0
        obs_mask = 1.0 - miss_mask
        return obs_mask, miss_mask

    def __getitem__(self, index):
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.base_dataset[index]
        obs_mask, miss_mask = self._build_masks(seq_x)
        masked_x = seq_x * obs_mask
        return {
            'task_name': 'imputation',
            'x': masked_x.astype(np.float32),
            'y': seq_x.astype(np.float32),  # reconstruction target
            'x_mark': seq_x_mark,
            'y_mark': seq_x_mark,
            'label': np.array([-1], dtype=np.int64),
            'obs_mask': obs_mask.astype(np.float32),
            'miss_mask': miss_mask.astype(np.float32),
        }
