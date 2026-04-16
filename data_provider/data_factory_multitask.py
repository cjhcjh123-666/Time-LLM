from torch.utils.data import DataLoader

from data_provider.data_loader_multitask import (
    FORECAST_DATASETS,
    ForecastTaskDataset,
    ClassificationTaskDataset,
    AnomalyTaskDataset,
    ImputationTaskDataset,
)


def _build_forecast_dataset(args, flag):
    dataset_cls = FORECAST_DATASETS[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if args.data == 'm4':
        base_dataset = dataset_cls(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            seasonal_patterns=args.seasonal_patterns
        )
    else:
        base_dataset = dataset_cls(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=args.freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )

    return ForecastTaskDataset(base_dataset)


def data_provider_multitask(args, flag):
    """
    Unified data provider entrypoint for four-task training.
    Current incremental stage:
      - forecast: available
      - classification/anomaly/imputation: interface reserved
    """
    if flag == 'test':
        shuffle_flag = False
        drop_last = True
    else:
        shuffle_flag = True
        drop_last = True

    batch_size = args.batch_size
    task_name = args.task_name

    if task_name in ['long_term_forecast', 'short_term_forecast', 'forecast']:
        data_set = _build_forecast_dataset(args, flag)
        if args.data == 'm4':
            drop_last = False
    elif task_name == 'classification':
        data_set = ClassificationTaskDataset(args, flag=flag)
        drop_last = False
    elif task_name == 'anomaly_detection':
        data_set = AnomalyTaskDataset(args, flag=flag)
    elif task_name == 'imputation':
        base_dataset = _build_forecast_dataset(args, flag).base_dataset
        data_set = ImputationTaskDataset(
            base_dataset=base_dataset,
            mask_rate=args.mask_rate,
            mask_mode=args.mask_mode
        )
        drop_last = False
    else:
        raise ValueError('Unknown task_name: {}'.format(task_name))

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    return data_set, data_loader
