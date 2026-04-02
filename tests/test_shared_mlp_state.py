import importlib.util
from pathlib import Path
from types import ModuleType
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SHARED_MLP_PATH = Path(__file__).resolve().parents[1] / "modules" / "shared_mlp.py"
PVCNN_GENERATION_PATH = Path(__file__).resolve().parents[1] / "model" / "pvcnn_generation.py"


def load_shared_mlp_module():
    spec = importlib.util.spec_from_file_location("pvd_shared_mlp_test", SHARED_MLP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pvcnn_generation_module(shared_mlp_module):
    sys.modules.pop("modules", None)
    sys.modules.pop("pvd_pvcnn_generation_test", None)

    modules_stub = ModuleType("modules")

    class DummyPassThrough(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, inputs):
            return inputs

    class DummySwish(nn.Module):
        def forward(self, x):
            return x

    modules_stub.SharedMLP = shared_mlp_module.SharedMLP
    modules_stub.PVConv = DummyPassThrough
    modules_stub.PointelligencePVConv = DummyPassThrough
    modules_stub.PointNetSAModule = DummyPassThrough
    modules_stub.PointNetAModule = DummyPassThrough
    modules_stub.PointNetFPModule = DummyPassThrough
    modules_stub.Attention = DummyPassThrough
    modules_stub.Swish = DummySwish
    modules_stub.clear_pointelligence_forward_cache = lambda: None
    sys.modules["modules"] = modules_stub

    spec = importlib.util.spec_from_file_location("pvd_pvcnn_generation_test", PVCNN_GENERATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SharedMLPStateTests(unittest.TestCase):
    def test_shared_mlp_dim1_accepts_flat_points(self):
        module = load_shared_mlp_module()
        mlp = module.SharedMLP(8, [16], dim=1)
        x2 = torch.randn(10, 8)
        x3 = x2.unsqueeze(-1)

        out2 = mlp(x2)
        out3 = mlp(x3).squeeze(-1)

        self.assertTrue(torch.allclose(out2, out3))
        self.assertEqual(out2.shape, torch.Size([10, 16]))

    def test_apply_classifier_accepts_2d_and_3d_features(self):
        shared_mlp_module = load_shared_mlp_module()
        module = load_pvcnn_generation_module(shared_mlp_module)

        model = module.PVCNN2Base.__new__(module.PVCNN2Base)
        nn.Module.__init__(model)
        model.classifier = nn.Sequential(nn.Conv1d(4, 3, 1))

        out2 = model._apply_classifier(torch.randn(10, 4))
        out3 = model._apply_classifier(torch.randn(2, 4, 5))

        self.assertEqual(out2.shape, torch.Size([10, 3]))
        self.assertEqual(out3.shape, torch.Size([2, 3, 5]))


if __name__ == "__main__":
    unittest.main()
