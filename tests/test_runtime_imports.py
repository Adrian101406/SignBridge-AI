from __future__ import annotations


def test_tensorflow_runtime_executes_an_operation() -> None:
    import tensorflow as tf

    assert int(tf.reduce_sum(tf.constant([1, 2])).numpy()) == 3


def test_torch_and_torchaudio_can_execute_together() -> None:
    import torch
    import torchaudio

    waveform = torch.zeros((1, 160), dtype=torch.float32)
    resampled = torchaudio.functional.resample(waveform, 16000, 8000)

    assert resampled.shape == (1, 80)
