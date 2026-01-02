import pytest
import torch

from src.config import DatasetConfig
from src.dataset import DatasetFactory, DoublePendulumDataset


def test_pendulum_dataset_shape():
    ds = DoublePendulumDataset(size=10, sequence_length=20)
    assert len(ds) == 10

    # Check item shape
    traj = ds[0]  # Should be (sequence_length, 4)
    assert traj.shape == (20, 4)
    assert isinstance(traj, torch.Tensor)


def test_dataset_factory():
    config = DatasetConfig(name="pendulum", size=10, sequence_length=15, dt=0.01)
    ds = DatasetFactory.get_dataset(config)
    assert isinstance(ds, DoublePendulumDataset)
    assert len(ds) == 10
    assert ds[0].shape == (15, 4)


def test_dataset_factory_invalid():
    config = DatasetConfig(name="non_existent_dataset")
    with pytest.raises(ValueError):
        DatasetFactory.get_dataset(config)
