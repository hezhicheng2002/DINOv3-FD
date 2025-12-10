import os
from typing import Iterator, Tuple

import torch
import torch.nn as nn
import math
import torch


def vit_target_linear_modules(model: nn.Module) -> Iterator[Tuple[str, nn.Linear]]:
    """
    Yield (name, module) for target Linear layers in ViT blocks that we want to adapt.
    Default includes: blocks[*].attn.qkv / attn.proj / mlp.fc1 / mlp.fc2
    """
    targets = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and any(k in name for k in targets):
            yield name, m


def _get_parent_and_attr(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    parent_path, attr = parts[:-1], parts[-1]
    parent = root
    for p in parent_path:
        parent = getattr(parent, p)
    return parent, attr


########################
# LyCORIS (LoHa/LoKr)  #
########################

def inject_lycoris_vit(model: nn.Module, rank: int = 8, algo: str = "loha"):
    """
    Attach LyCORIS adapters to selected Linear layers in ViT encoder.
    Requires 'lycoris' package available in environment (installed from local repo is fine).
    """
    try:
        from lycoris.wrapper import create_lycoris
    except Exception:
        # Fallback: try to add local LyCORIS repo to path
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent
        cand = repo_root / "LyCORIS"
        if cand.exists():
            sys.path.insert(0, str(cand))
        try:
            from lycoris.wrapper import create_lycoris  # type: ignore
        except Exception as e:
            raise ImportError(f"LyCORIS not available: {e}")

    # The wrapper can traverse the module and wrap eligible layers internally.
    # To constrain the scope, we supply a preset that targets Linear modules and names we care about.
    net = create_lycoris(
        model,
        multiplier=1.0,
        linear_dim=rank,
        linear_alpha=rank,
        algo=algo,
        preset="full",  # use default; wrapper filters by module/name internally
        enable_conv=False,
    )
    # After creation, LycorisNetwork holds references and will inject on forward.
    # Ensure LyCORIS parameters are trainable
    for p in net.parameters():
        p.requires_grad_(True)
    return net


##########
# IA3    #
##########

def inject_ia3_vit(model: nn.Module) -> nn.Module:
    try:
        from peft import IA3Config, get_peft_model
    except Exception as e:
        raise ImportError(f"peft IA3Config not available: {e}")
    cfg = IA3Config(
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        feedforward_modules=["mlp.fc1", "mlp.fc2"],
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, cfg)


##########
# VeRA   #
##########

def inject_vera_vit(model: nn.Module, rank: int = 8) -> nn.Module:
    try:
        from peft import VeraConfig, get_peft_model  # type: ignore[attr-defined]
    except Exception as e:
        raise ImportError(f"peft VeraConfig not available in this version: {e}")
    cfg = VeraConfig(
        r=rank,
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, cfg)


##########
# PaCA   #
##########

def inject_paca_vit(model: nn.Module, r: int = 8, alpha: int = 16) -> nn.Module:
    """
    Attach PaCA adapters to ViT linear layers. Falls back to local paca/peft copy when
    upstream peft lacks PacaConfig.
    """
    try:
        from peft import PacaConfig, get_peft_model  # type: ignore[attr-defined]
    except Exception:
        import sys
        from pathlib import Path
        # Try local bundled PaCA peft
        local_root = Path(__file__).resolve().parent / "paca" / "peft" / "src"
        if local_root.exists():
            sys.path.insert(0, str(local_root))
            import importlib
            for k in list(sys.modules.keys()):
                if k == "peft" or k.startswith("peft."):
                    sys.modules.pop(k)
        try:
            from peft import PacaConfig, get_peft_model  # type: ignore[attr-defined]
        except Exception as e:
            raise ImportError(f"PaCA peft is not available: {e}")

    cfg = PacaConfig(
        r=r,
        paca_alpha=alpha,
        target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
        task_type="FEATURE_EXTRACTION",
    )
    return get_peft_model(model, cfg)


#############################
# Smoke helper (assertions) #
#############################

def assert_adapter_smoke(model: nn.Module, adapter_param_filter=lambda n, p: p.requires_grad):
    """
    Minimal assertions for smoke test: count linear-like modules, and check trainable size.
    """
    n_lin = sum(1 for n, m in model.named_modules()
                if isinstance(m, (nn.Linear,)) or m.__class__.__name__.lower().startswith("linear4bit"))
    print(f"[SMOKE] linear-like modules in encoder: {n_lin}")
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for n, p in model.named_parameters() if adapter_param_filter(n, p))
    print(f"[SMOKE] total params={n_total/1e6:.2f}M, trainable={n_train/1e6:.2f}M")
    assert n_train > 0 and n_train < max(1, int(n_total * 0.2)), "Trainable params abnormal."


def one_batch_step(model: nn.Module, batch, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cuda"):
    model.train()
    x, y = batch
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    out = model(x)  # assumes model returns logits directly; caller can wrap if needed
    loss = criterion(out, y)
    assert torch.isfinite(loss).item(), "Loss is not finite."
    loss.backward()
    any_grad = False
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            gnorm = p.grad.data.norm().item()
            assert math.isfinite(gnorm), f"Non-finite grad on {n}"
            any_grad = True
            break
    assert any_grad, "No adapter grad found."
    tracked = None
    for n, p in model.named_parameters():
        if p.requires_grad and p.data.numel() > 0:
            tracked = (n, p.data.clone())
            break
    optimizer.step()
    if tracked is not None:
        n, before = tracked
        after = dict(model.named_parameters())[n].data
        assert not torch.allclose(before, after), f"No update observed on {n}"
    print(f"[SMOKE] one optimizer step finished. loss={loss.item():.4f}")
