import argparse

import torch
from mmcv import Config
from torchpack import distributed as dist
from torchpack.utils.config import configs

from mmdet3d.apis import multi_gpu_test, single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.utils import recursive_eval

from train_kradar import override_data_root


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data_root/ann_file to point at a K-Radar tree.",
    )
    parser.add_argument("--out", type=str, help="output result file in pickle format")
    parser.add_argument(
        "--fuse-conv-bn",
        action="store_true",
        help="Whether to fuse conv and bn, this will slightly increase the testing speed",
    )
    parser.add_argument(
        "--gpu-collect",
        action="store_true",
        help="whether to use gpu to collect results",
    )
    parser.add_argument(
        "--tmpdir", help="tmp directory used for collecting results to be merged"
    )
    parser.add_argument(
        "--eval",
        type=str,
        nargs="+",
        help='evaluation metrics, e.g., "bbox" "map"',
    )
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="pytorch")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )
    args, rest = parser.parse_known_args()
    configs.load(args.config, recursive=True)
    configs.update(rest)
    return args


def main():
    args = parse_args()
    cfg = Config(recursive_eval(configs), filename=args.config)
    override_data_root(cfg, args.data_root)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        dist.init()
        torch.cuda.set_device(dist.local_rank())

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    from mmdet3d.models import build_model

    model = build_model(cfg.model)
    if cfg.get("fp16", None):
        from mmcv.runner import wrap_fp16_model

        wrap_fp16_model(model)
    if args.fuse_conv_bn:
        from mmcv.cnn.utils import fuse_conv_bn

        model = fuse_conv_bn(model)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    model = model.cuda()

    if not distributed:
        model = torch.nn.DataParallel(model)
        outputs = single_gpu_test(model, data_loader)
    else:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)

    rank = 0 if not distributed else dist.get_rank()
    if rank == 0:
        from mmdet3d.apis import evaluate_results

        evaluate_results(dataset, outputs, args.out, args.eval)


if __name__ == "__main__":
    main()
