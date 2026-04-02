import importlib.util
import logging
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRAIN_GENERATION_PATH = Path(__file__).resolve().parents[1] / "train_generation.py"


def load_train_generation():
    for name in [
        "utils",
        "utils.file_utils",
        "utils.visualize",
        "datasets",
        "datasets.shapenet_data_pc",
        "model",
        "model.pvcnn_generation",
        "pvd_train_generation_sampling_test",
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

    spec = importlib.util.spec_from_file_location("pvd_train_generation_sampling_test", TRAIN_GENERATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SamplingStateIOTests(unittest.TestCase):
    def make_model(self):
        module = load_train_generation()
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1, state_grid_size=0.25)
        betas = module.get_betas("linear", 1e-4, 0.02, 4)
        model = module.Model(args, betas, "mse", "eps", "fixedsmall")
        return module, model

    @staticmethod
    def zero_noise(size, dtype, device):
        return torch.zeros(size=size, dtype=dtype, device=device)

    def test_gen_samples_returns_dense_shape_after_state_internal_loop(self):
        module, model = self.make_model()
        seen_types = []
        testcase = self

        def fake_p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
            seen_types.append(type(data).__name__)
            testcase.assertIsInstance(data, module.PointCloudState)
            testcase.assertEqual(data.grid_size, 0.25)
            return data.replace_coords_and_features(torch.zeros_like(data.coords))

        model.diffusion.p_sample = MethodType(fake_p_sample, model.diffusion)

        out = model.gen_samples((2, 3, 4), device=torch.device("cpu"), noise_fn=self.zero_noise, clip_denoised=False)

        self.assertEqual(out.shape, torch.Size([2, 3, 4]))
        self.assertTrue(seen_types)
        self.assertEqual(set(seen_types), {"PointCloudState"})

    def test_gen_sample_traj_returns_dense_snapshots_after_state_internal_loop(self):
        module, model = self.make_model()
        seen_types = []
        testcase = self

        def fake_p_sample(self, denoise_fn, data, t, noise_fn, clip_denoised=False, return_pred_xstart=False):
            seen_types.append(type(data).__name__)
            testcase.assertIsInstance(data, module.PointCloudState)
            testcase.assertEqual(data.grid_size, 0.25)
            return data.replace_coords_and_features(torch.zeros_like(data.coords))

        model.diffusion.p_sample = MethodType(fake_p_sample, model.diffusion)

        traj = model.gen_sample_traj((2, 3, 4), device=torch.device("cpu"), freq=2, noise_fn=self.zero_noise, clip_denoised=False)

        self.assertTrue(traj)
        self.assertTrue(all(x.shape == torch.Size([2, 3, 4]) for x in traj))
        self.assertTrue(seen_types)
        self.assertEqual(set(seen_types), {"PointCloudState"})


if __name__ == "__main__":
    unittest.main()
