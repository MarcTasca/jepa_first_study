from src.config import DatasetConfig, ExperimentConfig


def test_default_config():
    config = ExperimentConfig()
    assert config.dataset.name == "pendulum"
    assert config.training.batch_size == 64
    assert config.model.embedding_dim == 32


def test_nested_config_overrides():
    ds_config = DatasetConfig(name="circle", size=50)
    config = ExperimentConfig(dataset=ds_config)
    assert config.dataset.name == "circle"
    assert config.dataset.size == 50
    # Check default remaining unchanged
    assert config.training.jepa_epochs == 100
