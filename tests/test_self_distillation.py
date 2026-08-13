import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense_baseline import MLP
from src.recovery.self_distillation import self_distillation_recovery


def test_self_distillation_runs_one_epoch():
    teacher = MLP(input_dim=4, hidden_dims=[8], num_classes=2, use_batch_norm=False)
    student = MLP(input_dim=4, hidden_dims=[4], num_classes=2, use_batch_norm=False)
    loader = DataLoader(TensorDataset(torch.randn(12, 4), torch.randint(0, 2, (12,))), batch_size=4)
    recovered, history = self_distillation_recovery(
        student,
        teacher,
        loader,
        loader,
        epochs=1,
        learning_rate=0.01,
        device="cpu",
        verbose=False,
    )
    assert history["recovery_level"] == 3
    assert recovered(torch.randn(2, 4)).shape == (2, 2)
