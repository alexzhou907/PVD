import importlib.util
import logging
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modules"))
from point_cloud_state import PointCloudState, dense_to_point_cloud_state, point_cloud_state_from_points_list


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_generation.py"


def load_train_generation():
    for name in [
        "utils",
        "utils.file_utils",
        "utils.visualize",
        "datasets",
        "datasets.shapenet_data_pc",
        "model",
        "model.pvcnn_generation",
        "pvd_train_generation_test",
    ]:
        sys.modules.pop(name, None)

    utils_pkg = ModuleType("utils")
    utils_file_utils = ModuleType("utils.file_utils")
    utils_file_utils.np = np
    utils_file_utils.logger = logging.getLogger("test")
    utils_visualize = ModuleType("utils.visualize")

    datasets_pkg = ModuleType("datasets")
    datasets_shapenet = ModuleType("datasets.shapenet_data_pc")

    class DummyShapeNet15kPointClouds:
        pass

    datasets_shapenet.ShapeNet15kPointClouds = DummyShapeNet15kPointClouds

    model_pkg = ModuleType("model")
    model_pvcnn_generation = ModuleType("model.pvcnn_generation")

    class FakePVCNN2Base(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, inputs, t):
            return inputs

    model_pvcnn_generation.PVCNN2Base = FakePVCNN2Base

    sys.modules["utils"] = utils_pkg
    sys.modules["utils.file_utils"] = utils_file_utils
    sys.modules["utils.visualize"] = utils_visualize
    sys.modules["datasets"] = datasets_pkg
    sys.modules["datasets.shapenet_data_pc"] = datasets_shapenet
    sys.modules["model"] = model_pkg
    sys.modules["model.pvcnn_generation"] = model_pvcnn_generation

    spec = importlib.util.spec_from_file_location("pvd_train_generation_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerationStateDiffusionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_train_generation()
        betas = self.module.get_betas("linear", 1e-4, 0.02, 4)
        self.diffusion = self.module.GaussianDiffusion(
            betas=betas,
            loss_type="mse",
            model_mean_type="eps",
            model_var_type="fixedsmall",
        )

    def make_state(self):
        coords = torch.arange(48, dtype=torch.float32).view(2, 3, 8) / 100.0
        return dense_to_point_cloud_state(coords, grid_size=0.25)

    def test_q_sample_accepts_point_cloud_state(self):
        state = self.make_state()
        t = torch.tensor([0, 1], dtype=torch.int64)
        noise = torch.zeros_like(state.coords)

        data_t = self.diffusion.q_sample(state, t, noise=noise)

        self.assertIsInstance(data_t, PointCloudState)
        self.assertEqual(data_t.coords.shape, state.coords.shape)
        self.assertTrue(torch.equal(data_t.sample_inds, state.sample_inds))
        self.assertTrue(torch.equal(data_t.sample_sizes, state.sample_sizes))

    def test_p_losses_reduce_back_to_sample_axis_for_state(self):
        state = self.make_state()
        t = torch.tensor([0, 1], dtype=torch.int64)
        noise = torch.zeros_like(state.coords)

        losses = self.diffusion.p_losses(
            denoise_fn=lambda data, step: torch.zeros_like(data.coords),
            data_start=state,
            t=t,
            noise=noise,
        )

        self.assertEqual(losses.shape, torch.Size([2]))

    def test_model_get_loss_iter_accepts_point_cloud_state(self):
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1, state_grid_size=0.25)
        betas = self.module.get_betas("linear", 1e-4, 0.02, 4)
        model = self.module.Model(args, betas, "mse", "eps", "fixedsmall")
        state = self.make_state()
        noise = torch.zeros_like(state.coords)

        def zero_state_denoise(self, data, t):
            return torch.zeros_like(data.coords)

        model._denoise = MethodType(zero_state_denoise, model)
        losses = model.get_loss_iter(state, noises=noise)

        self.assertEqual(losses.shape, torch.Size([2]))

    def test_model_requires_explicit_state_grid_size(self):
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1)
        betas = self.module.get_betas("linear", 1e-4, 0.02, 4)

        with self.assertRaises(ValueError):
            self.module.Model(args, betas, "mse", "eps", "fixedsmall")

    def test_model_get_loss_iter_converts_dense_batch_to_state_with_configured_grid_size(self):
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1, state_grid_size=0.25)
        betas = self.module.get_betas("linear", 1e-4, 0.02, 4)
        model = self.module.Model(args, betas, "mse", "eps", "fixedsmall")

        dense = torch.arange(48, dtype=torch.float32).view(2, 3, 8) / 100.0
        noise = torch.zeros_like(dense)

        def zero_state_denoise(self, data, t):
            self._seen_grid_size = data.grid_size
            self._seen_type = type(data).__name__
            return torch.zeros_like(data.coords)

        model._denoise = MethodType(zero_state_denoise, model)
        losses = model.get_loss_iter(dense, noises=noise)

        self.assertEqual(losses.shape, torch.Size([2]))
        self.assertEqual(model._seen_type, "PointCloudState")
        self.assertEqual(model._seen_grid_size, 0.25)

    def test_model_get_loss_iter_accepts_variable_length_state_batch(self):
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1, state_grid_size=0.25)
        betas = self.module.get_betas("linear", 1e-4, 0.02, 4)
        model = self.module.Model(args, betas, "mse", "eps", "fixedsmall")
        state = point_cloud_state_from_points_list(
            [
                torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=torch.float32),
                torch.tensor([[3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
            ],
            grid_size=0.25,
        )
        noise = torch.zeros_like(state.coords)

        def zero_state_denoise(self, data, t):
            return torch.zeros_like(data.coords)

        model._denoise = MethodType(zero_state_denoise, model)
        losses = model.get_loss_iter(state, noises=noise)

        self.assertEqual(losses.shape, torch.Size([2]))

    def test_model_all_kl_accepts_point_cloud_state(self):
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1, state_grid_size=0.25)
        betas = self.module.get_betas("linear", 1e-4, 0.02, 4)
        model = self.module.Model(args, betas, "mse", "eps", "fixedsmall")
        state = self.make_state()

        def zero_state_denoise(self, data, t):
            return torch.zeros_like(data.coords)

        model._denoise = MethodType(zero_state_denoise, model)

        diagnostics = model.all_kl(state)

        self.assertEqual(diagnostics["total_bpd_b"].shape, torch.Size([]))
        self.assertEqual(diagnostics["terms_bpd"].shape, torch.Size([]))
        self.assertEqual(diagnostics["prior_bpd_b"].shape, torch.Size([]))
        self.assertEqual(diagnostics["mse_bt"].shape, torch.Size([]))

    def test_q_sample_invalidates_existing_geometry_cache(self):
        state = self.make_state().replace_query_context(
            query_coords=torch.randn(6, 3),
            query_sample_inds=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([3, 3], dtype=torch.long),
        ).replace_geometry(
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([2, 3], dtype=torch.long),
            num_neighbors=torch.tensor([1, 1], dtype=torch.long),
            downsample_indices=torch.tensor([0, 2], dtype=torch.long),
            i_upsample=torch.tensor([0, 1], dtype=torch.long),
            j_upsample=torch.tensor([1, 0], dtype=torch.long),
            k_upsample=torch.tensor([3, 2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        data_t = self.diffusion.q_sample(state, torch.tensor([0, 1], dtype=torch.int64), noise=torch.zeros_like(state.coords))

        self.assertIsNone(data_t.query_coords)
        self.assertIsNone(data_t.i)
        self.assertIsNone(data_t.downsample_indices)
        self.assertIsNone(data_t.kernel_size)
        self.assertEqual(data_t.sort_by, "k")

    def test_p_sample_invalidates_existing_geometry_cache(self):
        state = self.make_state().replace_query_context(
            query_coords=torch.randn(6, 3),
            query_sample_inds=torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([3, 3], dtype=torch.long),
        ).replace_geometry(
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([2, 3], dtype=torch.long),
            num_neighbors=torch.tensor([1, 1], dtype=torch.long),
            downsample_indices=torch.tensor([0, 2], dtype=torch.long),
            i_upsample=torch.tensor([0, 1], dtype=torch.long),
            j_upsample=torch.tensor([1, 0], dtype=torch.long),
            k_upsample=torch.tensor([3, 2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        sample = self.diffusion.p_sample(
            denoise_fn=lambda data, step: torch.zeros_like(data.coords),
            data=state,
            t=torch.tensor([1, 0], dtype=torch.int64),
            noise_fn=lambda **kwargs: torch.zeros(kwargs["size"], dtype=kwargs["dtype"], device=kwargs["device"]),
        )

        self.assertIsNone(sample.query_coords)
        self.assertIsNone(sample.i)
        self.assertIsNone(sample.downsample_indices)
        self.assertIsNone(sample.kernel_size)
        self.assertEqual(sample.sort_by, "k")


if __name__ == "__main__":
    unittest.main()
