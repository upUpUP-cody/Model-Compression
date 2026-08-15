import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.dense_baseline import MLP
from src.models.resnet_cifar import ResNetCIFAR
from src.recovery.lora_recovery import LoRAConv2d, lora_recovery


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


def test_lora_conv_forward_under_cuda_autocast():
    if not torch.cuda.is_available():
        return
    base = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
    wrapped = LoRAConv2d(base, rank=2, alpha=4.0).cuda()
    x = torch.randn(2, 3, 8, 8, device="cuda")
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = wrapped(x)
    assert out.shape == (2, 8, 8, 8)
    assert torch.isfinite(out.float()).all()


def test_lora_recovery_resnet_one_step_fp16_cuda():
    if not torch.cuda.is_available():
        return
    model = ResNetCIFAR(num_classes=10, base_width=16)
    loader = DataLoader(
        TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))),
        batch_size=4,
    )
    recovered, history = lora_recovery(
        model,
        loader,
        loader,
        epochs=1,
        learning_rate=0.01,
        device="cuda:0",
        precision="fp16",
        lora_rank=2,
        verbose=False,
    )
    assert history["recovery_level"] == 2
    assert recovered(torch.randn(2, 3, 32, 32, device="cuda:0")).shape == (2, 10)
