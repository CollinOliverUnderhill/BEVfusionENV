import argparse
import os
import random
import time
from typing import Any

import numpy as np
import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import convert_sync_batchnorm, get_root_logger, recursive_eval


def _maybe_rewrite_path(path: Any, data_root: str) -> Any:
    """Replace default dataset roots inside strings or lists.

    Many existing configs hardcode ``data/nuscenes``. This helper lets users
    pass a K-Radar root via ``--data-root`` without editing the YAML.
    """

    if isinstance(path, str):
        return path.replace("data/nuscenes", data_root)
    if isinstance(path, list):
        return [_maybe_rewrite_path(p, data_root) for p in path]
    return path


def override_data_root(cfg: Config, data_root: str) -> None:
    if not data_root:
        return

    cfg.data_root = data_root
    for split in ["train", "val", "test"]:
        if split not in cfg.data:
            continue
        split_cfg = cfg.data[split]
        if isinstance(split_cfg, dict):
            if "data_root" in split_cfg:
                split_cfg["data_root"] = data_root
            if "ann_file" in split_cfg:
                split_cfg["ann_file"] = _maybe_rewrite_path(
                    split_cfg["ann_file"], data_root
                )
        # nested datasets (e.g., RepeatDataset)
        if "dataset" in split_cfg and isinstance(split_cfg["dataset"], dict):
            nested = split_cfg["dataset"]
            if "data_root" in nested:
                nested["data_root"] = data_root
            if "ann_file" in nested:
                nested["ann_file"] = _maybe_rewrite_path(
                    nested["ann_file"], data_root
                )


def main():
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data_root/ann_file to point at a K-Radar tree.",
    )
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)
    override_data_root(cfg, args.data_root)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(f"Config:\n{cfg.pretty_text}")

    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, " f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=True,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
