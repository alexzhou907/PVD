import importlib.util
import logging
import pickle
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[1]
TRAIN_GENERATION_PATH = ROOT / "train_generation.py"
sys.path.insert(0, str(ROOT))


def load_train_generation():
    for name in [
        "utils",
        "utils.file_utils",
        "utils.visualize",
        "datasets",
        "datasets.shapenet_data_pc",
        "model",
        "model.pvcnn_generation",
        "pvd_train_generation_state_batch_test",
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

    class FakePVCNN2Base(torch.nn.Module):
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

    spec = importlib.util.spec_from_file_location("pvd_train_generation_state_batch_test", TRAIN_GENERATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyGenerationDataset(Dataset):
    def __init__(self):
        self.samples = [
            {
                "idx": 2,
                "train_points": torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=torch.float32),
                "test_points": torch.tensor([[9.0, 9.1, 9.2]], dtype=torch.float32),
                "mean": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                "std": np.array([[1.0]], dtype=np.float32),
                "cate_idx": 4,
                "sid": "sid-0",
                "mid": "mid-0",
            },
            {
                "idx": 0,
                "train_points": torch.tensor([[3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
                "test_points": torch.tensor([[8.0, 8.1, 8.2]], dtype=torch.float32),
                "mean": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
                "std": np.array([[2.0]], dtype=np.float32),
                "cate_idx": 7,
                "sid": "sid-1",
                "mid": "mid-1",
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class TaggedGenerationDataset(Dataset):
    def __init__(self, base_idx):
        self.samples = [
            {
                "idx": base_idx + 0,
                "train_points": torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
                "test_points": torch.tensor([[9.0, 9.1, 9.2]], dtype=torch.float32),
                "mean": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                "std": np.array([[1.0]], dtype=np.float32),
                "cate_idx": 1,
                "sid": f"sid-{base_idx}",
                "mid": f"mid-{base_idx}",
            },
            {
                "idx": base_idx + 1,
                "train_points": torch.tensor([[2.0, 2.1, 2.2], [3.0, 3.1, 3.2]], dtype=torch.float32),
                "test_points": torch.tensor([[8.0, 8.1, 8.2]], dtype=torch.float32),
                "mean": np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
                "std": np.array([[2.0]], dtype=np.float32),
                "cate_idx": 2,
                "sid": f"sid-{base_idx + 1}",
                "mid": f"mid-{base_idx + 1}",
            },
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class GenerationTrainStateBatchTests(unittest.TestCase):
    def test_collate_generation_train_samples_returns_state_batch(self):
        module = load_train_generation()
        dataset = TinyGenerationDataset()

        batch = module.collate_generation_train_samples(dataset.samples, state_grid_size=0.25)

        self.assertIsInstance(batch["state"], module.PointCloudState)
        self.assertNotIn("train_points", batch)
        self.assertEqual(batch["state"].sample_sizes.tolist(), [3, 2])
        self.assertEqual(batch["state"].grid_size, 0.25)
        self.assertTrue(torch.equal(batch["idx"], torch.tensor([2, 0], dtype=torch.long)))
        self.assertEqual(batch["sid"], ["sid-0", "sid-1"])
        self.assertEqual(batch["mid"], ["mid-0", "mid-1"])
        self.assertEqual(batch["mean"].shape, torch.Size([2, 1, 3]))
        self.assertEqual(batch["cate_idx"].shape, torch.Size([2]))

    def test_gather_generation_noise_flattens_per_sample_noise(self):
        module = load_train_generation()
        noise_bank = torch.arange(5 * 4 * 3, dtype=torch.float32).view(5, 4, 3)
        idx = torch.tensor([3, 1], dtype=torch.long)
        sample_sizes = torch.tensor([3, 2], dtype=torch.long)

        noise = module.gather_generation_noise(noise_bank, idx, sample_sizes, device=torch.device("cpu"))
        expected = torch.cat([noise_bank[3, :3], noise_bank[1, :2]], dim=0)

        self.assertEqual(noise.shape, torch.Size([5, 3]))
        self.assertTrue(torch.equal(noise, expected))

    def test_get_dataloader_uses_state_collate_for_train_batches(self):
        module = load_train_generation()
        dataset = TinyGenerationDataset()
        opt = SimpleNamespace(distribution_type=None, bs=2, workers=0, state_grid_size=0.5)

        dataloader, _, _, _ = module.get_dataloader(opt, dataset, None)
        batch = next(iter(dataloader))

        self.assertIsInstance(batch["state"], module.PointCloudState)
        self.assertNotIn("train_points", batch)
        self.assertEqual(sorted(batch["state"].sample_sizes.tolist()), [2, 3])
        self.assertEqual(sorted(batch["idx"].tolist()), [0, 2])
        self.assertEqual(batch["state"].grid_size, 0.5)

    def test_get_dataloader_uses_test_dataset_for_test_batches(self):
        module = load_train_generation()
        train_dataset = TaggedGenerationDataset(100)
        test_dataset = TaggedGenerationDataset(200)
        opt = SimpleNamespace(distribution_type=None, bs=2, workers=0, state_grid_size=0.5)

        _, test_dataloader, _, _ = module.get_dataloader(opt, train_dataset, test_dataset)
        batch = next(iter(test_dataloader))

        self.assertEqual(sorted(batch["idx"].tolist()), [200, 201])

    def test_move_generation_batch_to_device_preserves_string_lists_and_moves_state(self):
        module = load_train_generation()
        state = module.point_cloud_state_from_points_list(
            [
                torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
                torch.tensor([[2.0, 2.1, 2.2]], dtype=torch.float32),
            ],
            grid_size=0.25,
        )
        batch = {
            "state": state,
            "idx": torch.tensor([3, 1], dtype=torch.long),
            "sid": ["sid-0", "sid-1"],
            "nested": [torch.tensor([1.0], dtype=torch.float32), "keep-me"],
        }

        moved = module.move_generation_batch_to_device(batch, torch.device("cpu"))

        self.assertIsInstance(moved["state"], module.PointCloudState)
        self.assertEqual(moved["state"].coords.device.type, "cpu")
        self.assertEqual(moved["idx"].device.type, "cpu")
        self.assertEqual(moved["sid"], ["sid-0", "sid-1"])
        self.assertEqual(moved["nested"][1], "keep-me")

    def test_collated_state_is_pickleable_for_multiprocess_dataloader(self):
        module = load_train_generation()
        dataset = TinyGenerationDataset()
        batch = module.collate_generation_train_samples(dataset.samples, state_grid_size=0.25)

        restored = pickle.loads(pickle.dumps(batch["state"]))

        self.assertIsInstance(restored, module.PointCloudState)
        self.assertTrue(torch.equal(restored.coords, batch["state"].coords))
        self.assertTrue(torch.equal(restored.sample_sizes, batch["state"].sample_sizes))
        self.assertEqual(restored.grid_size, batch["state"].grid_size)

    def test_destroy_process_group_if_initialized_skips_uninitialized_single_process(self):
        module = load_train_generation()

        class DummyDist:
            def __init__(self):
                self.destroy_calls = 0

            def is_available(self):
                return True

            def is_initialized(self):
                return False

            def destroy_process_group(self):
                self.destroy_calls += 1

        dummy_dist = DummyDist()
        module.dist = dummy_dist

        module.destroy_process_group_if_initialized()

        self.assertEqual(dummy_dist.destroy_calls, 0)

    def test_destroy_process_group_if_initialized_destroys_initialized_group(self):
        module = load_train_generation()

        class DummyDist:
            def __init__(self):
                self.destroy_calls = 0

            def is_available(self):
                return True

            def is_initialized(self):
                return True

            def destroy_process_group(self):
                self.destroy_calls += 1

        dummy_dist = DummyDist()
        module.dist = dummy_dist

        module.destroy_process_group_if_initialized()

        self.assertEqual(dummy_dist.destroy_calls, 1)


if __name__ == "__main__":
    unittest.main()
