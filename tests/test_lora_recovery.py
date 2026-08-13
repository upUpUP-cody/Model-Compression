import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense_baseline import MLP
from src.recovery.lora_recovery import lora_recovery


def test_lora_recovery_updates_validation_accuracy():
    model = MLP(input_dim=4, hidden_dims=[8], num_classes=2, use_batch_norm=False)
    loader = DataLoader(TensorDataset(torch.randn(16, 4), torch.randint(0, 2, (16,))), batch_size=4)
    recovered, history = lora_recovery(
        model,
        loader,
        loader,
        epochs=1,
        learning_rate=0.01,
        device="cpu",
        lora_rank=2,
        verbose=False,
    )
    assert history["recovery_level"] == 2
    assert recovered(torch.randn(2, 4)).shape == (2, 2)
