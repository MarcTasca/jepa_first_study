import pytest
import torch

from src.models import Decoder, Encoder, Predictor


@pytest.fixture
def input_dim():
    return 8


@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def embedding_dim():
    return 16


@pytest.fixture
def batch_size():
    return 4


def test_encoder_shape(input_dim, hidden_dim, embedding_dim, batch_size):
    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, embedding_dim=embedding_dim)
    x = torch.randn(batch_size, input_dim)
    z = encoder(x)
    assert z.shape == (batch_size, embedding_dim)


def test_predictor_shape(embedding_dim, hidden_dim, batch_size):
    predictor = Predictor(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    z = torch.randn(batch_size, embedding_dim)
    z_next = predictor(z)
    assert z_next.shape == (batch_size, embedding_dim)


def test_decoder_shape(embedding_dim, hidden_dim, input_dim, batch_size):
    # Decoder output info is usually same as input dim (reconstruction)
    decoder = Decoder(embedding_dim=embedding_dim, output_dim=input_dim, hidden_dim=hidden_dim)
    z = torch.randn(batch_size, embedding_dim)
    x_hat = decoder(z)
    assert x_hat.shape == (batch_size, input_dim)
