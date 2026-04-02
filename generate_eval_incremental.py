import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='ShapeNetCore.v2.PC15k/')
    parser.add_argument('--category', default='chair')

    parser.add_argument('--batch_size', type=int, default=50, help='input batch size')
    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')

    parser.add_argument('--generate', type=str2bool, default=True)
    parser.add_argument('--eval_gen', type=str2bool, default=True)

    parser.add_argument('--nc', default=3)
    parser.add_argument('--npoints', default=2048)
    parser.add_argument('--beta_start', default=0.0001)
    parser.add_argument('--beta_end', default=0.02)
    parser.add_argument('--schedule_type', default='linear')
    parser.add_argument('--time_num', default=1000)

    parser.add_argument('--attention', default=True)
    parser.add_argument('--dropout', default=0.1)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--loss_type', default='mse')
    parser.add_argument('--model_mean_type', default='eps')
    parser.add_argument('--model_var_type', default='fixedsmall')

    parser.add_argument('--model', default='', required=True, help='path to model (to continue training)')
    parser.add_argument('--eval_path', default='')
    parser.add_argument('--manualSeed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', type=int, default=0, metavar='S', help='gpu id (default: 0)')

    parser.add_argument('--batch_output_dirname', default='batches', help='subdir under syn/ to store per-batch artifacts')
    parser.add_argument('--metrics_log_name', default='batch_metrics.jsonl', help='JSONL file that records per-batch metrics')
    parser.add_argument('--save_running_snapshots', type=str2bool, default=True, help='save samples_so_far.pth and refs_so_far.pth after each batch')
    parser.add_argument('--export_xyz', type=str2bool, default=True, help='export every generated sample to .xyz')
    parser.add_argument('--export_ply', type=str2bool, default=True, help='export every generated sample to .ply')

    opt = parser.parse_args()

    import torch
    opt.cuda = torch.cuda.is_available()
    return opt


def make_batch_artifact_paths(
    output_dir: os.PathLike[str] | str,
    batch_idx: int,
    metrics_log_name: str = "batch_metrics.jsonl",
) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_prefix = f"batch_{batch_idx:05d}"
    return {
        "samples": output_dir / f"{batch_prefix}_samples.pth",
        "refs": output_dir / f"{batch_prefix}_refs.pth",
        "preview": output_dir / f"{batch_prefix}.png",
        "metrics_log": output_dir / metrics_log_name,
        "samples_so_far": output_dir / "samples_so_far.pth",
        "refs_so_far": output_dir / "refs_so_far.pth",
        "metrics_summary": output_dir / "metrics_summary.json",
    }


def _default_save_fn(obj: Any, path: os.PathLike[str] | str) -> None:
    import torch
    torch.save(obj, path)


def _to_jsonable_scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (int, float)):
        return float(value)
    return value


def to_jsonable_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {str(key): _to_jsonable_scalar(value) for key, value in metrics.items()}


def _batch_length(batch: Any) -> int:
    try:
        return int(len(batch))
    except TypeError:
        return 0


def _iter_point_clouds(point_cloud_batch: Any):
    if hasattr(point_cloud_batch, "detach"):
        point_cloud_batch = point_cloud_batch.detach()
    if hasattr(point_cloud_batch, "cpu"):
        point_cloud_batch = point_cloud_batch.cpu()
    if hasattr(point_cloud_batch, "tolist"):
        point_cloud_batch = point_cloud_batch.tolist()
    for point_cloud in point_cloud_batch:
        yield point_cloud


def _write_xyz(path: os.PathLike[str] | str, point_cloud: Any) -> None:
    path = Path(path)
    lines = [f"{float(x):.6f} {float(y):.6f} {float(z):.6f}" for x, y, z in point_cloud]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_point_cloud_batch(
    output_dir: os.PathLike[str] | str,
    batch_prefix: str,
    point_cloud_batch: Any,
    export_xyz: bool = True,
    export_ply: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ply_paths = []
    xyz_paths = []
    pcwrite = None
    if export_ply:
        from utils.visualize import pcwrite as _pcwrite
        pcwrite = _pcwrite

    for sample_idx, point_cloud in enumerate(_iter_point_clouds(point_cloud_batch)):
        stem = f"{batch_prefix}_sample_{sample_idx:03d}"
        if export_xyz:
            xyz_path = output_dir / f"{stem}.xyz"
            _write_xyz(xyz_path, point_cloud)
            xyz_paths.append(xyz_path)
        if export_ply and pcwrite is not None:
            import numpy as np
            ply_path = output_dir / f"{stem}.ply"
            pcwrite(str(ply_path), np.asarray(point_cloud, dtype=np.float32))
            ply_paths.append(ply_path)

    return {"xyz": xyz_paths, "ply": ply_paths}


def persist_batch_artifacts(
    output_dir: os.PathLike[str] | str,
    batch_idx: int,
    generated_batch: Any,
    reference_batch: Any,
    metrics: Optional[Mapping[str, Any]],
    generated_so_far: Any = None,
    reference_so_far: Any = None,
    save_fn: Optional[Callable[[Any, os.PathLike[str] | str], None]] = None,
    metrics_log_name: str = "batch_metrics.jsonl",
) -> Dict[str, str]:
    paths = make_batch_artifact_paths(output_dir, batch_idx, metrics_log_name=metrics_log_name)
    save = save_fn or _default_save_fn
    save(generated_batch, paths["samples"])
    save(reference_batch, paths["refs"])
    if generated_so_far is not None:
        save(generated_so_far, paths["samples_so_far"])
    if reference_so_far is not None:
        save(reference_so_far, paths["refs_so_far"])

    record = {
        "batch_idx": int(batch_idx),
        "num_generated": _batch_length(generated_batch),
        "num_reference": _batch_length(reference_batch),
        "metrics": to_jsonable_metrics(metrics or {}),
    }
    with paths["metrics_log"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {key: str(value) for key, value in paths.items()}


def evaluate_generated_batch(sample_pcs, ref_pcs, eval_batch_size: int) -> Dict[str, Any]:
    from metrics.evaluation_metrics import EMD_CD, compute_all_metrics
    from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD

    effective_batch_size = max(1, min(int(eval_batch_size), int(sample_pcs.shape[0]), int(ref_pcs.shape[0])))

    metrics = {}
    metrics.update(EMD_CD(sample_pcs, ref_pcs, effective_batch_size))
    metrics.update(compute_all_metrics(sample_pcs, ref_pcs, effective_batch_size))
    metrics["JSD"] = JSD(sample_pcs.detach().cpu().numpy(), ref_pcs.detach().cpu().numpy())
    return to_jsonable_metrics(metrics)


def _update_running_metric_summary(running_summary: Dict[str, float], batch_metrics: Mapping[str, Any]) -> Dict[str, float]:
    for key, value in to_jsonable_metrics(batch_metrics).items():
        if isinstance(value, (int, float)):
            running_summary[key] = running_summary.get(key, 0.0) + float(value)
    return running_summary


def _mean_metric_summary(running_summary: Mapping[str, float], completed_batches: int) -> Dict[str, float]:
    if completed_batches <= 0:
        return {}
    return {key: value / float(completed_batches) for key, value in running_summary.items()}


def generate_and_evaluate_incrementally(model, opt, output_dir: str, logger):
    import torch
    from tqdm import tqdm

    import test_generation as base

    _, test_dataset = base.get_dataset(opt.dataroot, opt.npoints, opt.category)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
    )

    batch_output_dir = Path(output_dir) / opt.batch_output_dirname
    batch_output_dir.mkdir(parents=True, exist_ok=True)

    generated_all = []
    reference_all = []
    running_summary = {}
    completed_batches = 0

    iterator = tqdm(test_dataloader, desc='Generating Samples')
    for batch_idx, data in enumerate(iterator):
        x = data['test_points'].transpose(1, 2)
        m, s = data['mean'].float(), data['std'].float()

        if opt.cuda:
            x = x.cuda()

        gen = model.gen_samples(x.shape, 'cuda' if opt.cuda else 'cpu', clip_denoised=False).detach().cpu()
        x = x.detach().cpu()

        gen = gen.transpose(1, 2).contiguous()
        x = x.transpose(1, 2).contiguous()

        gen = gen * s + m
        x = x * s + m

        generated_all.append(gen)
        reference_all.append(x)

        batch_metrics = {}
        if opt.eval_gen:
            batch_metrics = evaluate_generated_batch(gen, x, opt.batch_size)
            _update_running_metric_summary(running_summary, batch_metrics)
            completed_batches += 1

        generated_so_far = None
        reference_so_far = None
        if opt.save_running_snapshots:
            generated_so_far = torch.cat(generated_all, dim=0)
            reference_so_far = torch.cat(reference_all, dim=0)

        record = persist_batch_artifacts(
            output_dir=batch_output_dir,
            batch_idx=batch_idx,
            generated_batch=gen,
            reference_batch=x,
            metrics=batch_metrics,
            generated_so_far=generated_so_far,
            reference_so_far=reference_so_far,
            metrics_log_name=opt.metrics_log_name,
        )

        base.visualize_pointcloud_batch(
            str(Path(record["preview"])),
            gen,
            os.path.join(os.path.dirname(__file__), 'assets', 'cameraparams'),
        )
        export_point_cloud_batch(
            output_dir=batch_output_dir,
            batch_prefix=f"batch_{batch_idx:05d}_samples",
            point_cloud_batch=gen,
            export_xyz=opt.export_xyz,
            export_ply=opt.export_ply,
        )

        if batch_metrics:
            logger.info(
                "Batch %d saved to %s | MMD-CD=%.6f MMD-EMD=%.6f fscore=%.6f JSD=%.6f",
                batch_idx,
                record["samples"],
                batch_metrics.get("MMD-CD", float("nan")),
                batch_metrics.get("MMD-EMD", float("nan")),
                batch_metrics.get("fscore", float("nan")),
                batch_metrics.get("JSD", float("nan")),
            )
            summary = _mean_metric_summary(running_summary, completed_batches)
            Path(record["metrics_summary"]).write_text(
                json.dumps({
                    "completed_batches": completed_batches,
                    "mean_metrics": summary,
                }, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        else:
            logger.info("Batch %d saved to %s", batch_idx, record["samples"])

    samples = torch.cat(generated_all, dim=0)
    refs = torch.cat(reference_all, dim=0)
    torch.save(samples, opt.eval_path)
    torch.save(refs, str(Path(opt.eval_path).with_name("refs.pth")))

    logger.info("Saved final concatenated samples to %s", opt.eval_path)
    logger.info("Saved final concatenated refs to %s", str(Path(opt.eval_path).with_name("refs.pth")))
    return samples, refs


def main(opt):
    import torch
    import torch.nn as nn

    import test_generation as base

    if opt.category == 'chair':
        opt.beta_start = 0.0001
        opt.beta_end = 0.02
        opt.schedule_type = 'linear'
    else:
        opt.beta_start = 0.0001
        opt.beta_end = 0.008
        opt.schedule_type = 'warm0.1'

    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    dir_id = os.path.dirname(__file__)
    output_dir = base.get_output_dir(dir_id, exp_id)
    base.copy_source(__file__, output_dir)
    logger = base.setup_logging(output_dir)
    outf_syn, = base.setup_output_subdirs(output_dir, 'syn')

    betas = base.get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    model = base.Model(opt, betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

    device = 'cuda' if opt.cuda else 'cpu'
    if opt.cuda:
        model = model.cuda()

        def _transform_(m):
            return nn.parallel.DataParallel(m)

        model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():
        logger.info("Resume Path:%s" % opt.model)
        resumed_param = torch.load(opt.model, map_location=device)
        model.load_state_dict(resumed_param['model_state'])

        if opt.generate:
            opt.eval_path = os.path.join(outf_syn, 'samples.pth')
            Path(opt.eval_path).parent.mkdir(parents=True, exist_ok=True)
            generate_and_evaluate_incrementally(model, opt, outf_syn, logger)


if __name__ == '__main__':
    import test_generation as base

    opt = parse_args()
    base.set_seed(opt)
    main(opt)
