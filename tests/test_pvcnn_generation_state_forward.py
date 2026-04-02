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
from point_cloud_state import dense_to_point_cloud_state, point_cloud_state_from_points_list


PVCNN_GENERATION_PATH = Path(__file__).resolve().parents[1] / "model" / "pvcnn_generation.py"
TRAIN_GENERATION_PATH = Path(__file__).resolve().parents[1] / "train_generation.py"
SHARED_MLP_PATH = Path(__file__).resolve().parents[1] / "modules" / "shared_mlp.py"


def load_shared_mlp_module():
    spec = importlib.util.spec_from_file_location("pvd_shared_mlp_for_forward_test", SHARED_MLP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pvcnn_generation_module():
    shared_mlp_module = load_shared_mlp_module()

    for name in ["modules", "pvd_pvcnn_generation_forward_test"]:
        sys.modules.pop(name, None)

    modules_stub = ModuleType("modules")

    class DummyPassThrough(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.init_kwargs = kwargs
            out_channels = kwargs.get("out_channels")
            if isinstance(out_channels, (list, tuple)) and out_channels:
                last = out_channels[-1]
                self.out_channels = last[-1] if isinstance(last, (list, tuple)) else last

        def forward(self, inputs):
            return inputs

    class DummyVoxelPVConv(DummyPassThrough):
        pass

    class DummyPointelligencePVConv(DummyPassThrough):
        pass

    class DummySwish(nn.Module):
        def forward(self, x):
            return x

    modules_stub.SharedMLP = shared_mlp_module.SharedMLP
    modules_stub.PVConv = DummyVoxelPVConv
    modules_stub.PointelligencePVConv = DummyPointelligencePVConv
    modules_stub.PointNetSAModule = DummyPassThrough
    modules_stub.PointNetAModule = DummyPassThrough
    modules_stub.PointNetFPModule = DummyPassThrough
    modules_stub.Attention = DummyPassThrough
    modules_stub.Swish = DummySwish
    modules_stub.clear_pointelligence_forward_cache = lambda: None
    sys.modules["modules"] = modules_stub

    spec = importlib.util.spec_from_file_location("pvd_pvcnn_generation_forward_test", PVCNN_GENERATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._dummy_voxel_cls = DummyVoxelPVConv
    module._dummy_pointelligence_cls = DummyPointelligencePVConv
    return module


def load_train_generation():
    for name in [
        "utils",
        "utils.file_utils",
        "utils.visualize",
        "datasets",
        "datasets.shapenet_data_pc",
        "model",
        "model.pvcnn_generation",
        "pvd_train_generation_forward_test",
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
            self.init_kwargs = kwargs

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

    spec = importlib.util.spec_from_file_location("pvd_train_generation_forward_test", TRAIN_GENERATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PVCNNGenerationStateForwardTests(unittest.TestCase):
    def test_pointelligence_backend_selects_pointelligence_pvconv(self):
        module = load_pvcnn_generation_module()

        layers, _, _ = module.create_pointnet_components(
            blocks=[(16, 1, 8)],
            in_channels=3,
            embed_dim=8,
            pvconv_backend="pointelligence",
        )

        self.assertIsInstance(layers[0], module._dummy_pointelligence_cls)

    def test_hybrid_backend_only_replaces_configured_resolutions(self):
        module = load_pvcnn_generation_module()

        class TinyPVCNN(module.PVCNN2Base):
            sa_blocks = [
                ((16, 1, 32), (8, 0.1, 8, (8, 16))),
                ((32, 1, 16), (4, 0.2, 8, (16, 32))),
                ((64, 1, 8), (2, 0.4, 8, (32, 64))),
            ]
            fp_blocks = [
                ((64, 64), (64, 1, 8)),
                ((32, 32), (32, 1, 16)),
                ((16, 16), (16, 1, 32)),
            ]

        model = TinyPVCNN(
            num_classes=3,
            embed_dim=64,
            use_att=False,
            dropout=0.1,
            extra_feature_channels=0,
            pvconv_backend="pointelligence",
            pvconv_local_attention=False,
            pvconv_pointelligence_resolutions=frozenset({8}),
        )

        voxel_modules = [
            submodule for submodule in model.modules() if isinstance(submodule, module._dummy_voxel_cls)
        ]
        point_modules = [
            submodule for submodule in model.modules() if isinstance(submodule, module._dummy_pointelligence_cls)
        ]

        self.assertEqual(sorted(submodule.init_kwargs["resolution"] for submodule in voxel_modules), [16, 16, 32, 32])
        self.assertEqual(sorted(submodule.init_kwargs["resolution"] for submodule in point_modules), [8, 8])

    def test_local_attention_can_be_disabled_without_disabling_global_attention(self):
        module = load_pvcnn_generation_module()

        class TinyPVCNN(module.PVCNN2Base):
            sa_blocks = [
                ((16, 1, 8), (8, 0.1, 8, (8, 16))),
            ]
            fp_blocks = [
                ((16, 16), (16, 1, 8)),
            ]

        model = TinyPVCNN(
            num_classes=3,
            embed_dim=64,
            use_att=True,
            dropout=0.1,
            extra_feature_channels=0,
            pvconv_backend="voxel",
            pvconv_local_attention=False,
        )

        self.assertIsNotNone(model.global_att)
        pvconv_modules = [submodule for submodule in model.modules() if isinstance(submodule, module._dummy_voxel_cls)]
        self.assertTrue(pvconv_modules)
        self.assertTrue(all(submodule.init_kwargs["attention"] is False for submodule in pvconv_modules))

    def test_decoder_local_attention_hits_a_fp_pvconv_stage(self):
        module = load_pvcnn_generation_module()

        class TinyPVCNN(module.PVCNN2Base):
            sa_blocks = [
                ((16, 1, 32), (8, 0.1, 8, (8, 16))),
                ((32, 1, 16), (4, 0.2, 8, (16, 32))),
                ((64, 1, 8), (2, 0.4, 8, (32, 64))),
            ]
            fp_blocks = [
                ((16, 16), (16, 2, 8)),
                ((16, 16), (16, 2, 16)),
                ((16, 16), (16, 2, 32)),
            ]

        model = TinyPVCNN(
            num_classes=3,
            embed_dim=64,
            use_att=True,
            dropout=0.1,
            extra_feature_channels=0,
            pvconv_backend="voxel",
            pvconv_local_attention=True,
        )

        fp_pvconv_modules = []
        for fp_layer in model.fp_layers:
            fp_pvconv_modules.extend(
                submodule for submodule in fp_layer.modules() if isinstance(submodule, module._dummy_voxel_cls)
            )

        self.assertTrue(fp_pvconv_modules)
        self.assertTrue(any(submodule.init_kwargs["attention"] is True for submodule in fp_pvconv_modules))

    def test_pointelligence_backend_rejects_local_attention(self):
        module = load_pvcnn_generation_module()

        with self.assertRaisesRegex(ValueError, "local attention"):
            module.create_pointnet_components(
                blocks=[(16, 1, 8)],
                in_channels=3,
                embed_dim=8,
                pvconv_backend="pointelligence",
                pvconv_local_attention=True,
            )

    def test_pvcnn_forward_accepts_state(self):
        module = load_pvcnn_generation_module()
        model = module.PVCNN2Base.__new__(module.PVCNN2Base)
        nn.Module.__init__(model)

        captured = {}

        def fake_forward_state(self, inputs, t):
            captured["coords_shape"] = inputs.coords.shape
            captured["t"] = t.clone()
            return torch.zeros(inputs.coords.shape[0], 3)

        model._forward_state = MethodType(fake_forward_state, model)

        dense = torch.randn(2, 3, 16)
        state = dense_to_point_cloud_state(dense)
        t = torch.tensor([0, 1], dtype=torch.int64)

        out = module.PVCNN2Base.forward(model, state, t)

        self.assertEqual(captured["coords_shape"], torch.Size([32, 3]))
        self.assertTrue(torch.equal(captured["t"], t))
        self.assertEqual(out.shape, torch.Size([32, 3]))

    def test_global_attention_state_path_preserves_variable_length_sample_sizes(self):
        module = load_pvcnn_generation_module()
        model = module.PVCNN2Base.__new__(module.PVCNN2Base)
        nn.Module.__init__(model)
        model.embed_dim = 4
        model.embedf = nn.Identity()
        model.sa_layers = nn.ModuleList([nn.Module()])
        model.fp_layers = nn.ModuleList([nn.Module()])
        model.global_att = nn.Identity()
        model.classifier = nn.Identity()

        def sa_forward(_, inputs):
            return inputs

        captured = {}

        def fp_forward(_, inputs):
            captured["sample_sizes"] = inputs[1].sample_sizes.clone()
            return inputs[1], inputs[-1]

        model.sa_layers[0].forward = MethodType(sa_forward, model.sa_layers[0])
        model.fp_layers[0].forward = MethodType(fp_forward, model.fp_layers[0])
        model._apply_classifier = MethodType(lambda self, features: features, model)

        state = point_cloud_state_from_points_list(
            [
                torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
                torch.tensor([[2.0, 2.1, 2.2], [3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
            ],
            grid_size=0.25,
        )
        t = torch.tensor([0, 1], dtype=torch.int64)

        module.PVCNN2Base._forward_state(model, state, t)

        self.assertEqual(captured["sample_sizes"].tolist(), [2, 3])

    def test_model_denoise_passes_state_directly_to_backbone(self):
        module = load_train_generation()
        args = SimpleNamespace(nc=3, embed_dim=64, attention=False, dropout=0.1, state_grid_size=0.25)
        betas = module.get_betas("linear", 1e-4, 0.02, 4)
        model = module.Model(args, betas, "mse", "eps", "fixedsmall")

        called = {}

        class FakeBackbone(nn.Module):
            def forward(self, data, t):
                called["data_type"] = type(data)
                called["coords_shape"] = data.coords.shape
                called["t"] = t.clone()
                return torch.zeros_like(data.coords)

        model.model = FakeBackbone()

        dense = torch.randn(2, 3, 16)
        state = dense_to_point_cloud_state(dense, grid_size=0.25)
        t = torch.tensor([0, 1], dtype=torch.int64)

        out = model._denoise(state, t)

        self.assertEqual(called["data_type"].__name__, "PointCloudState")
        self.assertEqual(called["coords_shape"], torch.Size([32, 3]))
        self.assertTrue(torch.equal(called["t"], t))
        self.assertEqual(out.shape, torch.Size([32, 3]))

    def test_model_passes_pvconv_backend_to_backbone(self):
        module = load_train_generation()
        args = SimpleNamespace(
            nc=3,
            embed_dim=64,
            attention=True,
            dropout=0.1,
            state_grid_size=0.25,
            pvconv_backend="pointelligence",
            pvconv_local_attention=False,
        )
        betas = module.get_betas("linear", 1e-4, 0.02, 4)
        model = module.Model(args, betas, "mse", "eps", "fixedsmall")

        self.assertEqual(model.model.init_kwargs["pvconv_backend"], "pointelligence")
        self.assertTrue(model.model.init_kwargs["use_att"])
        self.assertFalse(model.model.init_kwargs["pvconv_local_attention"])
        self.assertEqual(model.model.init_kwargs["pvconv_pointelligence_resolutions"], frozenset({8, 16, 32}))

    def test_model_passes_hybrid_resolution_selection_to_backbone(self):
        module = load_train_generation()
        args = SimpleNamespace(
            nc=3,
            embed_dim=64,
            attention=True,
            dropout=0.1,
            state_grid_size=0.25,
            pvconv_backend="pointelligence",
            pvconv_local_attention=False,
            pvconv_pointelligence_resolutions="8,16",
        )
        betas = module.get_betas("linear", 1e-4, 0.02, 4)
        model = module.Model(args, betas, "mse", "eps", "fixedsmall")

        self.assertEqual(model.model.init_kwargs["pvconv_pointelligence_resolutions"], frozenset({8, 16}))

    def test_model_rejects_pointelligence_local_attention(self):
        module = load_train_generation()
        args = SimpleNamespace(
            nc=3,
            embed_dim=64,
            attention=True,
            dropout=0.1,
            state_grid_size=0.25,
            pvconv_backend="pointelligence",
            pvconv_local_attention=True,
        )
        betas = module.get_betas("linear", 1e-4, 0.02, 4)

        with self.assertRaisesRegex(ValueError, "local attention"):
            module.Model(args, betas, "mse", "eps", "fixedsmall")

    def test_parse_args_uses_explicit_boolean_flags_for_attention(self):
        module = load_train_generation()
        saved_argv = sys.argv
        try:
            sys.argv = [
                "train_generation.py",
                "--attention",
                "False",
                "--pvconv_local_attention",
                "False",
                "--pvconv_pointelligence_resolutions",
                "8,16",
            ]
            args = module.parse_args()
        finally:
            sys.argv = saved_argv

        self.assertIs(args.attention, False)
        self.assertIs(args.pvconv_local_attention, False)
        self.assertEqual(args.pvconv_pointelligence_resolutions, frozenset({8, 16}))

    def test_parse_args_defaults_pointelligence_resolutions_to_all_pvconv_scales(self):
        module = load_train_generation()
        saved_argv = sys.argv
        try:
            sys.argv = ["train_generation.py"]
            args = module.parse_args()
        finally:
            sys.argv = saved_argv

        self.assertEqual(args.pvconv_pointelligence_resolutions, frozenset({8, 16, 32}))


if __name__ == "__main__":
    unittest.main()
