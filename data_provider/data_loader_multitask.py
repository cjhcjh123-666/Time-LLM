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
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('ClassificationTaskDataset will be added in next incremental change.')


class AnomalyTaskDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('AnomalyTaskDataset will be added in next incremental change.')


class ImputationTaskDataset(Dataset):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('ImputationTaskDataset will be added in next incremental change.')
