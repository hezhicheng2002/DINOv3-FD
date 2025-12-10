import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
try:
    import timm  # type: ignore
except ImportError:
    timm = None
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, ConcatDataset
from torchvision import datasets, transforms
from tqdm import tqdm


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class Adapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1, out_dim: int = 1, style: str = "mlp") -> None:
        super().__init__()
        self.style = style
        if self.style == "linear" or hidden_dim <= 0:
            self.linear = nn.Linear(input_dim, out_dim)
            nn.init.xavier_uniform_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            self.fc1 = None  # type: ignore[assignment]
            self.fc2 = None  # type: ignore[assignment]
            self.act = None  # type: ignore[assignment]
            self.drop = None  # type: ignore[assignment]
        else:
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, out_dim)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.style == "linear":
            return
        assert self.fc1 is not None and self.fc2 is not None
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.style == "linear":
            assert hasattr(self, "linear") and self.linear is not None
            return self.linear(x)
        assert self.fc1 is not None and self.fc2 is not None and self.act is not None and self.drop is not None
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class OrthogonalDisentangler(nn.Module):
    def __init__(
        self,
        encoder_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        feature_dim: Optional[int] = None,
        frozen_blocks: int = 8,
        encoder_source: str = "huggingface",
        pretrained_path: Optional[str] = None,
        hf_local_files_only: bool = False,
        c_mode: str = "cls",
        c_mask_ratio: float = 0.15,
        c_attn_temp: float = 1.0,
        out_dim: int = 1,
        head_style: str = "mlp",
        debug_shapes: bool = False,
    ) -> None:
        super().__init__()
        self.encoder_source = encoder_source.lower()
        self.encoder_name = encoder_name
        self.hf_local_files_only = hf_local_files_only
        self.logger = logging.getLogger("puredino")
        self.c_mode = c_mode
        self.c_mask_ratio = c_mask_ratio
        self.c_attn_temp = c_attn_temp
        self.out_dim = max(1, int(out_dim))
        # default head style; may be overridden by args (e.g., linear_probe)
        self._head_style = head_style
        self.debug_shapes = bool(debug_shapes)
        encoder_dim: Optional[int] = None

        if self.encoder_source == "dinov3":
            try:
                from dinov3.hub import backbones  # type: ignore
            except ModuleNotFoundError:
                import importlib
                import types as _types
                import sys as _sys
                local_pkg_dir = Path(__file__).resolve().parent / "dinov3" / "dinov3"
                if not local_pkg_dir.exists():
                    raise
                # Create/override a 'dinov3' package pointing to local repo
                pkg_name = "dinov3"
                pkg = _types.ModuleType(pkg_name)
                pkg.__path__ = [str(local_pkg_dir)]  # type: ignore[attr-defined]
                _sys.modules[pkg_name] = pkg
                backbones = importlib.import_module("dinov3.hub.backbones")  # type: ignore[assignment]

            build_fn = getattr(backbones, encoder_name, None)
            if build_fn is None:
                raise ValueError(f"Unknown DINOv3 backbone: {encoder_name}")
            weights_arg = pretrained_path if pretrained_path else backbones.Weights.LVD1689M
            self.logger.info(
                f"Loading DINOv3 backbone '{encoder_name}' with weights '{weights_arg}'."
            )
            self.encoder = build_fn(pretrained=True, weights=weights_arg)
            encoder_dim = getattr(self.encoder, "num_features", None) or getattr(self.encoder, "embed_dim", None)
        elif self.encoder_source == "timm":
            if timm is None:
                raise ImportError("timm is not installed; cannot use timm encoder source.")
            self.logger.info(f"Loading timm backbone '{encoder_name}'.")
            self.encoder = timm.create_model(
                encoder_name,
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )
            encoder_dim = getattr(self.encoder, "num_features", None)
        elif self.encoder_source == "huggingface":
            try:
                from transformers import AutoModel  # type: ignore
            except ImportError as exc:
                raise ImportError("transformers must be installed for huggingface encoder source") from exc

            model_ref = pretrained_path or encoder_name
            self.logger.info(f"Loading Hugging Face backbone '{model_ref}'.")
            self.encoder = AutoModel.from_pretrained(model_ref, local_files_only=self.hf_local_files_only)
            encoder_dim = getattr(self.encoder.config, "hidden_size", None)
        else:
            raise ValueError(f"Unsupported encoder source: {encoder_source}")

        if encoder_dim is None:
            raise ValueError("Unable to determine encoder feature dimension; please specify --feature_dim explicitly.")

        if feature_dim is None or feature_dim <= 0:
            feature_dim = encoder_dim

        self.feature_dim = feature_dim
        if encoder_dim != feature_dim:
            self.logger.info(f"Projecting encoder features from {encoder_dim} to {feature_dim} dimensions.")
            self.proj = nn.Linear(encoder_dim, feature_dim)
        else:
            self.proj = nn.Identity()

        self._freeze_blocks(frozen_blocks)
        self.use_adapter_default = True
        self.adapter_disc = Adapter(feature_dim)
        self.adapter_indisc = Adapter(feature_dim)
        # Fallback simple projections if adapters are disabled at runtime
        self.proj_disc_simple = nn.Linear(feature_dim, 256)
        self.proj_indisc_simple = nn.Linear(feature_dim, 256)
        # head style may be overridden based on finetune_mode (set later via kwargs)
        head_style = self._head_style
        self.decoderA = Decoder(256, out_dim=self.out_dim, style=head_style)
        self.decoderA_fused = Decoder(512, out_dim=self.out_dim, style=head_style)
        self.decoderB = Decoder(256, out_dim=self.out_dim, style=head_style)
        # optional adaptive gating: per-sample soft routing of features between A/B
        self.use_gating: bool = False
        self.gate_layer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
        )
        # Optional feature prompt (used in prompt-tuning mode)
        self.feat_prompt = nn.Parameter(torch.zeros(feature_dim), requires_grad=False)
        self._use_feat_prompt = False
        # attention pooling for C2/C4
        self.pool_attn = nn.Linear(feature_dim, 1)

    def _freeze_blocks(self, frozen_blocks: int) -> None:
        if frozen_blocks <= 0:
            return

        blocks = None
        if self.encoder_source in {"dinov3", "timm"}:
            blocks = getattr(self.encoder, "blocks", None)
        elif self.encoder_source == "huggingface":
            candidate = None
            if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
                candidate = self.encoder.encoder.layer
            elif hasattr(self.encoder, "vit") and hasattr(self.encoder.vit, "encoder"):
                candidate = getattr(self.encoder.vit.encoder, "layer", None)
            blocks = candidate

        if blocks is None:
            self.logger.warning("Encoder does not expose transformer blocks for freezing; skipping freeze.")
            return

        blocks_iterable = list(blocks)
        frozen_blocks = min(frozen_blocks, len(blocks_iterable))
        for block in blocks_iterable[:frozen_blocks]:
            for param in block.parameters():
                param.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_source == "huggingface":
            outputs = self.encoder(pixel_values=x, output_hidden_states=False)
            if self.debug_shapes and hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                try:
                    b, n, d = outputs.last_hidden_state.shape
                    self.logger.info(f"[DEBUG SHAPE] encoder=huggingface tokens BxNxD={b}x{n}x{d}")
                except Exception:
                    pass
            if getattr(outputs, "pooler_output", None) is not None:
                feats = outputs.pooler_output
            else:
                feats = outputs.last_hidden_state[:, 0]
            feats = self.proj(feats)
            if self._use_feat_prompt:
                feats = feats + self.feat_prompt
            return feats
        elif self.encoder_source == "timm":
            feats = self.encoder.forward_features(x)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            if feats.dim() == 3:
                if self.debug_shapes:
                    try:
                        b, n, d = feats.shape
                        self.logger.info(f"[DEBUG SHAPE] encoder=timm tokens BxNxD={b}x{n}x{d}")
                    except Exception:
                        pass
                cls = feats[:, 0]
                patches = feats[:, 1:]
                # default to cls for timm unless mean/attn/mask requested
                if self.c_mode == "mean" and patches is not None:
                    v = patches.mean(dim=1)
                elif self.c_mode in {"attn_pool", "attn_mask"} and patches is not None:
                    # project patches then attn pool
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    scores = self.pool_attn(patches_proj).squeeze(-1) / max(1e-6, self.c_attn_temp)
                    weights = torch.softmax(scores, dim=1)
                    if self.c_mode == "attn_mask":
                        k = max(1, int(P * self.c_mask_ratio))
                        top_idx = torch.topk(weights, k=k, dim=1).indices
                        mask = torch.ones_like(weights)
                        mask.scatter_(1, top_idx, 0.0)
                        weights = weights * mask
                        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
                    v = (weights.unsqueeze(-1) * patches_proj).sum(dim=1)
                    feats = v
                    if self._use_feat_prompt:
                        feats = feats + self.feat_prompt
                    return feats
                else:
                    feats = cls
        else:  # dinov3 repo backbone
            outputs = self.encoder.forward_features(x)
            if isinstance(outputs, dict):
                cls = outputs.get("x_norm_clstoken")
                patches = outputs.get("x_norm_patchtokens", None)
                if self.debug_shapes and (cls is not None):
                    try:
                        B = cls.shape[0]
                        if patches is not None and patches.dim() == 3:
                            P = patches.shape[1]
                            Denc = patches.shape[2]
                            self.logger.info(f"[DEBUG SHAPE] encoder=dinov3 tokens BxNxD={B}x{1+P}x{Denc} (P={P})")
                        else:
                            # Only CLS visible; Denc from CLS
                            Denc = cls.shape[-1]
                            self.logger.info(f"[DEBUG SHAPE] encoder=dinov3 tokens BxNxD={B}x1x{Denc} (no patch tokens exposed)")
                    except Exception:
                        pass
                if cls is None:
                    raise ValueError("DINOv3 backbone did not return 'x_norm_clstoken'.")
                # C-modes on dinov3
                if self.c_mode == "mean" and patches is not None:
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    v = patches_proj.mean(dim=1)
                    feats = v
                elif self.c_mode in {"attn_pool", "attn_mask"} and patches is not None:
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    scores = self.pool_attn(patches_proj).squeeze(-1) / max(1e-6, self.c_attn_temp)
                    weights = torch.softmax(scores, dim=1)
                    if self.c_mode == "attn_mask":
                        k = max(1, int(P * self.c_mask_ratio))
                        top_idx = torch.topk(weights, k=k, dim=1).indices
                        mask = torch.ones_like(weights)
                        mask.scatter_(1, top_idx, 0.0)
                        weights = weights * mask
                        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
                    v = (weights.unsqueeze(-1) * patches_proj).sum(dim=1)
                    feats = v
                elif self.c_mode == "token_mask" and patches is not None:
                    B, P, Denc = patches.shape
                    patches_proj = self.proj(patches.reshape(B * P, Denc)).reshape(B, P, -1)
                    k = max(1, int(P * self.c_mask_ratio))
                    # random mask per sample
                    device = patches_proj.device
                    mask = torch.ones((B, P), device=device)
                    idx = torch.rand((B, P), device=device).argsort(dim=1)[:, :k]
                    mask.scatter_(1, idx, 0.0)
                    denom = mask.sum(dim=1, keepdim=True) + 1e-6
                    v = (patches_proj * mask.unsqueeze(-1)).sum(dim=1) / denom
                    feats = v
                else:
                    feats = cls
            elif isinstance(outputs, list):
                primary = outputs[0]
                if isinstance(primary, dict):
                    cls = primary.get("x_norm_clstoken")
                    patches = primary.get("x_norm_patchtokens", None)
                    if cls is None:
                        raise ValueError("Unexpected DINOv3 backbone output structure.")
                    feats = cls
                else:
                    feats = primary
            else:
                feats = outputs
        # if feats still in encoder dim, project
        if feats.dim() == 2 and feats.size(1) != self.feature_dim:
            feats = self.proj(feats)
        if self._use_feat_prompt:
            feats = feats + self.feat_prompt
        return feats

    def forward(self, x: torch.Tensor, grl_lambda: float = 1.0, use_adapter: bool = True) -> Dict[str, torch.Tensor]:
        feats = self._encode(x)
        if use_adapter:
            f_disc = self.adapter_disc(feats)
            f_indisc = self.adapter_indisc(feats)
        else:
            f_disc = self.proj_disc_simple(feats)
            f_indisc = self.proj_indisc_simple(feats)
        gate = None
        if self.use_gating:
            gate = torch.sigmoid(self.gate_layer(feats))
            f_disc = f_disc * gate
            f_indisc = f_indisc * (1.0 - gate)
        logitsA = self.decoderA(f_disc)
        logitsB = self.decoderB(grad_reverse(f_indisc, grl_lambda))
        return {
            "logitsA": logitsA,
            "logitsB": logitsB,
            "f_disc": f_disc,
            "f_indisc": f_indisc,
            "feat": feats,
            "gate": gate,
        }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(
    grayscale_to_rgb: bool = True,
    use_imagenet_norm: bool = True,
    aug_extra: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    if use_imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    to_rgb = [transforms.Grayscale(num_output_channels=3)] if grayscale_to_rgb else []

    aug_list = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
    if aug_extra:
        aug_list.extend([transforms.RandomRotation(10), transforms.ColorJitter(0.1,0.1,0.1,0.05)])
    train_tf = transforms.Compose(aug_list + [*to_rgb, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    eval_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            *to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def build_dataloaders(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    grayscale_to_rgb: bool = True,
    use_imagenet_norm: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    balance_sampler: bool = False,
    aug_extra: bool = False,
    train_fraction: float = 1.0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, datasets.ImageFolder]:
    train_tf, eval_tf = build_transforms(
        grayscale_to_rgb=grayscale_to_rgb, use_imagenet_norm=use_imagenet_norm, aug_extra=aug_extra
    )
    train_ds = datasets.ImageFolder(root=str(data_root / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(root=str(data_root / "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(root=str(data_root / "test"), transform=eval_tf)

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    # Some torch versions require prefetch_factor only if num_workers>0
    if num_workers > 0 and prefetch_factor is not None:
        common_kwargs.update(dict(prefetch_factor=prefetch_factor))

    # Optional train subset by fraction (stratified over classes if possible)
    subset_indices = None
    if isinstance(train_fraction, float) and 0.0 < train_fraction < 1.0 and hasattr(train_ds, 'targets'):
        import numpy as _np
        rng = _np.random.default_rng(int(seed))
        labels = _np.array(train_ds.targets)
        subset_indices = []
        for cls in _np.unique(labels):
            idx = _np.where(labels == cls)[0]
            k = max(1, int(round(len(idx) * train_fraction)))
            sel = rng.choice(idx, size=k, replace=False)
            subset_indices.extend(sel.tolist())
        subset_indices = _np.array(subset_indices, dtype=_np.int64)

    train_data_for_loader = train_ds
    sampler = None
    if subset_indices is not None:
        from torch.utils.data import Subset
        train_data_for_loader = Subset(train_ds, subset_indices.tolist())
        if balance_sampler and hasattr(train_ds, 'targets'):
            import numpy as _np
            labels = _np.array(train_ds.targets)[subset_indices]
            class_sample_count = _np.bincount(labels).astype(_np.float64)
            class_sample_count[class_sample_count == 0] = 1.0
            weights = 1.0 / class_sample_count
            sample_weights = weights[labels]
            sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
    elif balance_sampler and hasattr(train_ds, 'targets'):
        import numpy as _np
        labels = _np.array(train_ds.targets)
        class_sample_count = _np.bincount(labels).astype(_np.float64)
        class_sample_count[class_sample_count == 0] = 1.0
        weights = 1.0 / class_sample_count
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

    if sampler is not None:
        train_loader = DataLoader(train_data_for_loader, shuffle=False, sampler=sampler, **common_kwargs)
    else:
        train_loader = DataLoader(train_data_for_loader, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader, train_ds


class RSNAPneumoniaDataset(Dataset):
    """
    RSNA Pneumonia challenge (Kaggle) binary classification dataset.

    Expects root to contain:
      - stage_2_train_images/*.dcm
      - stage_2_train_labels.csv (columns include patientId, Target, ...)

    We derive image-level label per patientId: y=1 if any Target==1 rows for this patient.
    Splits are stratified 80/10/10 by patientId with a fixed seed for reproducibility.
    DICOMs are decoded via pydicom and converted to PIL images (grayscale);
    transforms then handle grayscale→RGB if requested by build_transforms.
    """

    def __init__(self, root: Path, split: str, transform: Optional[transforms.Compose] = None, seed: int = 42) -> None:
        super().__init__()
        try:
            import pydicom  # type: ignore
        except ImportError as exc:
            raise ImportError("pydicom is required for RSNA dataset decoding. Please install pydicom.") from exc

        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._seed = int(seed)
        self.samples: List[Tuple[str, int]] = []

        labels_csv = self.root / "stage_2_train_labels.csv"
        train_dir = self.root / "stage_2_train_images"
        if not labels_csv.exists() or not train_dir.exists():
            raise FileNotFoundError(
                f"RSNA dataset not found under {root}. Expected stage_2_train_labels.csv and stage_2_train_images/"
            )
        # Build patientId -> label mapping
        import csv
        pos = set()
        neg = set()
        with open(labels_csv, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                pid = row.get("patientId") or row.get("patientid")
                if not pid:
                    continue
                try:
                    tgt = int(float(row.get("Target", "0")))
                except Exception:
                    tgt = 0
                if tgt == 1:
                    pos.add(pid)
                else:
                    if pid not in pos:
                        neg.add(pid)
        # Stratified split 80/10/10 by patientId
        rng = np.random.RandomState(self._seed)
        pos = sorted(list(pos))
        neg = sorted(list(neg))
        def split_ids(ids: List[str]):
            n = len(ids)
            idx = np.arange(n)
            rng.shuffle(idx)
            n_train = max(1, int(0.8 * n))
            n_val = max(1, int(0.1 * n))
            train_ids = [ids[i] for i in idx[:n_train]]
            val_ids = [ids[i] for i in idx[n_train:n_train + n_val]]
            test_ids = [ids[i] for i in idx[n_train + n_val:]]
            return train_ids, val_ids, test_ids
        p_tr, p_va, p_te = split_ids(pos)
        n_tr, n_va, n_te = split_ids(neg)
        splits = {
            "train": set(p_tr + n_tr),
            "val": set(p_va + n_va),
            "test": set(p_te + n_te),
        }
        # Assemble samples for the requested split
        keep = splits.get(self.split)
        if keep is None:
            raise ValueError(f"Unsupported split: {self.split}")
        for pid in keep:
            dcm_path = train_dir / f"{pid}.dcm"
            if dcm_path.exists():
                label = 1 if pid in pos else 0
                self.samples.append((str(dcm_path), label))

    def __len__(self) -> int:
        return len(self.samples)

    def _dicom_to_pil(self, path: str):
        import pydicom  # type: ignore
        import numpy as _np
        from PIL import Image  # type: ignore
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(_np.float32)
        # Apply rescale if present
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept
        # Normalize to 0..255
        mn, mx = float(arr.min()), float(arr.max())
        if mx > mn:
            arr = (arr - mn) / (mx - mn)
        arr = (arr * 255.0).clip(0, 255).astype(_np.uint8)
        # Invert if MONOCHROME1
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = 255 - arr
        img = Image.fromarray(arr, mode="L")
        return img

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = self._dicom_to_pil(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_rsna_dataloaders(
    root: Path,
    batch_size: int,
    num_workers: int,
    grayscale_to_rgb: bool = True,
    use_imagenet_norm: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    balance_sampler: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, RSNAPneumoniaDataset]:
    train_tf, eval_tf = build_transforms(grayscale_to_rgb=grayscale_to_rgb, use_imagenet_norm=use_imagenet_norm)
    train_ds = RSNAPneumoniaDataset(root, split="train", transform=train_tf, seed=seed)
    val_ds = RSNAPneumoniaDataset(root, split="val", transform=eval_tf, seed=seed)
    test_ds = RSNAPneumoniaDataset(root, split="test", transform=eval_tf, seed=seed)

    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
    if num_workers > 0 and prefetch_factor is not None:
        common_kwargs.update(dict(prefetch_factor=prefetch_factor))

    if balance_sampler:
        import numpy as _np
        labels = _np.array([y for _, y in train_ds.samples])
        class_sample_count = _np.bincount(labels).astype(_np.float64)
        class_sample_count[class_sample_count == 0] = 1.0
        weights = 1.0 / class_sample_count
        sample_weights = weights[labels]
        sampler = WeightedRandomSampler(torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, shuffle=False, sampler=sampler, **common_kwargs)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **common_kwargs)
    return train_loader, val_loader, test_loader, train_ds


class ImageFolderBinaryDataset(Dataset):
    def __init__(self, root: Path, split: str, label_name: str, transform: Optional[transforms.Compose]=None) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.label = label_name
        self.transform = transform
        base = self.root / split
        imgfolder = datasets.ImageFolder(str(base))
        class_to_idx = {c: i for i, c in enumerate(imgfolder.classes)}
        if label_name not in class_to_idx:
            raise ValueError(f"Label '{label_name}' not found in classes: {imgfolder.classes}")
        pos_idx = class_to_idx[label_name]
        self.samples: List[Tuple[Path,int]] = []
        self.targets: List[int] = []
        for p, idx in imgfolder.samples:
            y = 1 if idx == pos_idx else 0
            self.samples.append((Path(p), y))
            self.targets.append(y)
    def __len__(self) -> int:
        return len(self.samples)
    def __getitem__(self, idx:int) -> Tuple[torch.Tensor,int]:
        p,y = self.samples[idx]
        from PIL import Image
        img = Image.open(p).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, y

def build_imagefolder_label_dataloaders(
    root: Path,
    label: str,
    batch_size:int,
    num_workers:int,
    grayscale_to_rgb: bool=True,
    use_imagenet_norm: bool=True,
    persistent_workers: bool=True,
    prefetch_factor:int=2,
) -> Tuple[DataLoader,DataLoader,DataLoader,Dataset]:
    tr, ev = build_transforms(grayscale_to_rgb, use_imagenet_norm)
    train_ds = ImageFolderBinaryDataset(root,'train',label,tr)
    val_ds = ImageFolderBinaryDataset(root,'val',label,ev)
    test_ds = ImageFolderBinaryDataset(root,'test',label,ev)
    common=dict(batch_size=batch_size,num_workers=num_workers,pin_memory=True,
                persistent_workers=persistent_workers if num_workers>0 else False)
    if num_workers>0 and prefetch_factor is not None: common.update(dict(prefetch_factor=prefetch_factor))
    return DataLoader(train_ds,shuffle=True,**common), DataLoader(val_ds,shuffle=False,**common), DataLoader(test_ds,shuffle=False,**common), train_ds


def covariance_loss(f_disc: torch.Tensor, f_indisc: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    b = f_disc.size(0)
    if b <= 1:
        return torch.tensor(0.0, device=f_disc.device)
    x = f_disc
    y = f_indisc
    if normalize:
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    cov = (x.transpose(0, 1) @ y) / (b - 1)
    return torch.norm(cov, p="fro") ** 2


def hsic_loss(
    f_disc: torch.Tensor,
    f_indisc: torch.Tensor,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None,
) -> torch.Tensor:
    # Biased HSIC with RBF kernels
    def rbf(x: torch.Tensor, sigma: Optional[float]) -> torch.Tensor:
        x = x.float()  # stabilize under AMP
        x2 = (x * x).sum(dim=1, keepdim=True)
        dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
        dist2 = torch.clamp(dist2, min=0.0)
        if sigma is None:
            # median heuristic
            vals = dist2.detach().flatten()
            # exclude zeros for median; if all zero, fallback to 1.0
            nz = vals[vals > 0]
            med = torch.median(nz) if nz.numel() > 0 else torch.tensor(1.0, device=x.device)
            med = torch.clamp(med, min=1e-6)
            sigma_val = torch.sqrt(0.5 * med)
        else:
            sigma_val = torch.tensor(sigma, device=x.device).clamp(min=1e-6)
        denom = 2.0 * sigma_val**2 + 1e-8
        k = torch.exp(-dist2 / denom)
        return k

    n = f_disc.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=f_disc.device)
    K = rbf(f_disc, sigma_x)
    L = rbf(f_indisc, sigma_y)
    H = torch.eye(n, device=f_disc.device) - (1.0 / n) * torch.ones((n, n), device=f_disc.device)
    KH = (K @ H)
    LH = (L @ H)
    hsic = torch.trace(KH @ LH) / ((n - 1) ** 2 + 1e-8)
    return hsic


def gram_loss(f_disc: torch.Tensor, f_indisc: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    b = f_disc.size(0)
    if b <= 1:
        return torch.tensor(0.0, device=f_disc.device)
    x = f_disc
    y = f_indisc
    if normalize:
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        y = torch.nn.functional.normalize(y, p=2, dim=1)
    gram = (x.transpose(0, 1) @ y) / b
    return torch.norm(gram, p="fro") ** 2


def _select_topk_dims_by_corr(f1: torch.Tensor, f2: torch.Tensor, topk_ratio: float) -> torch.Tensor:
    D = f1.size(1)
    k = max(1, int(D * float(topk_ratio)))
    f1c = f1 - f1.mean(dim=0, keepdim=True)
    f2c = f2 - f2.mean(dim=0, keepdim=True)
    num = (f1c * f2c).sum(dim=0)
    den = (f1c.pow(2).sum(dim=0).sqrt() * f2c.pow(2).sum(dim=0).sqrt() + 1e-6)
    corr = (num / den).abs()
    _, idx = torch.topk(corr, k=min(k, D))
    return idx


def decouple_loss_selective(
    f_disc: torch.Tensor,
    f_indisc: torch.Tensor,
    mode: str,
    normalize: bool,
    topk_ratio: float,
) -> torch.Tensor:
    if topk_ratio > 0.0 and mode in {"gram", "cov"}:
        idx = _select_topk_dims_by_corr(f_disc, f_indisc, topk_ratio)
        f_d = f_disc[:, idx]
        f_i = f_indisc[:, idx]
    else:
        f_d, f_i = f_disc, f_indisc
    if mode == "cov":
        return covariance_loss(f_d, f_i, normalize=normalize)
    if mode == "hsic":
        return hsic_loss(f_d, f_i)
    return gram_loss(f_d, f_i, normalize=normalize)


def decouple_loss_weighted(
    f_disc: torch.Tensor,
    f_indisc: torch.Tensor,
    mode: str,
    normalize: bool,
    topk_ratio: float,
    importance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Variant of decouple_loss_selective that can reweight feature dimensions
    based on a per-dim importance signal (e.g., |dL_cls/df_disc| averaged over
    the batch). Lower-importance dims receive relatively larger weights so that
    disentanglement focuses more on dimensions that are less critical for A.
    """
    if importance is not None:
        # importance expected shape [D]; emphasize lower-importance dims
        imp = importance.detach()
        if imp.dim() == 1 and imp.numel() == f_disc.size(1):
            # small trick: higher weight for smaller importance
            w = (imp.max() - imp + 1e-6)
            w = w / (w.mean() + 1e-6)
            f_disc = f_disc * w
            f_indisc = f_indisc * w
    return decouple_loss_selective(f_disc, f_indisc, mode=mode, normalize=normalize, topk_ratio=topk_ratio)


def supervised_contrastive_loss(z: torch.Tensor, y: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z = torch.nn.functional.normalize(z, dim=1)
    sim = torch.matmul(z, z.t()) / max(temperature, 1e-6)
    B = z.size(0)
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    y = y.view(-1)
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1)) & (~eye)
    sim_exp = torch.exp(sim) * (~eye).float()
    denom = sim_exp.sum(dim=1, keepdim=True) + 1e-6
    num = (sim_exp * pos_mask.float()).sum(dim=1, keepdim=True) + 1e-12
    valid = pos_mask.any(dim=1).float()
    loss = -(valid * torch.log(num / denom)).sum() / (valid.sum() + 1e-6)
    return loss


class MINE(nn.Module):
    def __init__(self, in_dim: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def estimate_mi(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Pos pairs
        pos = self.forward(torch.cat([x, y], dim=1))
        # Neg pairs (shuffle y)
        idx = torch.randperm(y.size(0), device=y.device)
        y_shuf = y[idx]
        neg = self.forward(torch.cat([x, y_shuf], dim=1))
        mi = pos.mean() - torch.log(torch.exp(neg).mean() + 1e-12)
        return mi


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        # expose linear-like attributes that some code paths may query
        self.in_features = in_f
        self.out_features = out_f
        self.bias = base.bias
        self.r = r
        self.alpha = alpha
        self.scale = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Linear(in_f, r, bias=False)
            self.lora_B = nn.Linear(r, out_f, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
        # freeze base
        for p in self.base.parameters():
            p.requires_grad = False
        # align LoRA module device with base right away
        try:
            dev = next(self.base.parameters()).device
            self.to(dev)
        except StopIteration:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            # ensure LoRA params on same device/dtype as input
            if self.lora_A.weight.device != x.device:
                self.lora_A.to(x.device)
                self.lora_B.to(x.device)
            out = out + self.scale * self.lora_B(self.dropout(self.lora_A(x)))
        return out


def replace_linear_with_lora(module: nn.Module, r: int, alpha: float, dropout: float) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            replace_linear_with_lora(child, r, alpha, dropout)


def entropy_regularizer(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    term = probs * torch.log(probs + eps) + (1.0 - probs) * torch.log(1.0 - probs + eps)
    return -term.mean()


def softmax_entropy_regularizer(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    ent = -(probs * (probs + eps).log()).sum(dim=1)
    return ent.mean()


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.collected: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.collected[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.collected:
                p.data.copy_(self.collected[name])
        self.collected = {}


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad = 0

    def step(self, metric: Optional[float]) -> bool:
        if metric is None or math.isnan(metric):
            # treat NaN as no improvement
            self.num_bad += 1
            return self.num_bad > self.patience
        if self.best is None:
            self.best = metric
            self.num_bad = 0
            return False
        improve = (metric - self.best) > self.min_delta if self.mode == "max" else (self.best - metric) > self.min_delta
        if improve:
            self.best = metric
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad > self.patience


@torch.no_grad()
def evaluate(
    model: OrthogonalDisentangler,
    loader: DataLoader,
    device: torch.device,
    lambda_grl: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    all_logits_a: List[torch.Tensor] = []
    all_logits_b: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images, grl_lambda=lambda_grl)
        all_logits_a.append(outputs["logitsA"].detach().cpu())
        all_logits_b.append(outputs["logitsB"].detach().cpu())
        all_targets.append(targets.detach().cpu())
    logits_a = torch.cat(all_logits_a, dim=0)
    logits_b = torch.cat(all_logits_b, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    if logits_a.dim() == 1 or logits_a.size(1) == 1:
        la = logits_a.numpy().ravel(); lb = logits_b.numpy().ravel(); ty = targets_t.numpy().ravel()
        pa = 1.0/(1.0+np.exp(-la)); pb = 1.0/(1.0+np.exp(-lb))
        pred_a = (pa>=0.5).astype(int); pred_b=(pb>=0.5).astype(int)
        bal_acc_a = balanced_accuracy_score(ty, pred_a)
        bal_acc_b = balanced_accuracy_score(ty, pred_b)
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(ty, pred_a, average='binary', zero_division=0)
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(ty, pred_b, average='binary', zero_division=0)
        metrics = {
            "acc_A": float(accuracy_score(ty, pred_a)),
            "acc_B": float(accuracy_score(ty, pred_b)),
            "bal_acc_A": float(bal_acc_a),
            "bal_acc_B": float(bal_acc_b),
            "prec_A": float(prec_a),
            "rec_A": float(rec_a),
            "f1_A": float(f1_a),
            "prec_B": float(prec_b),
            "rec_B": float(rec_b),
            "f1_B": float(f1_b),
            "auc_A": 0.5,
            "auc_B": 0.5,
        }
        # compute ROC AUC when possible
        try:
            metrics["auc_A"] = float(roc_auc_score(ty, pa))
        except ValueError:
            metrics["auc_A"] = 0.5
        try:
            metrics["auc_B"] = float(roc_auc_score(ty, pb))
        except ValueError:
            metrics["auc_B"] = 0.5
        return metrics
    else:
        # multiclass: top-1 accuracy, balanced accuracy, macro OVR AUROC
        ty = targets_t.numpy()
        pred_a = logits_a.argmax(dim=1).numpy(); pred_b = logits_b.argmax(dim=1).numpy()
        bal_acc_a = balanced_accuracy_score(ty, pred_a)
        bal_acc_b = balanced_accuracy_score(ty, pred_b)
        # macro precision/recall/f1
        prec_a, rec_a, f1_a, _ = precision_recall_fscore_support(ty, pred_a, average='macro', zero_division=0)
        prec_b, rec_b, f1_b, _ = precision_recall_fscore_support(ty, pred_b, average='macro', zero_division=0)
        # OVR macro AUROC
        pa = torch.softmax(logits_a, dim=1).numpy(); pb = torch.softmax(logits_b, dim=1).numpy()
        try:
            ovr_a = float(roc_auc_score(ty, pa, multi_class='ovr', average='macro'))
        except ValueError:
            ovr_a = 0.5
        try:
            ovr_b = float(roc_auc_score(ty, pb, multi_class='ovr', average='macro'))
        except ValueError:
            ovr_b = 0.5
        return {
            "acc_A": float(accuracy_score(ty, pred_a)),
            "acc_B": float(accuracy_score(ty, pred_b)),
            "bal_acc_A": float(bal_acc_a),
            "bal_acc_B": float(bal_acc_b),
            "prec_A": float(prec_a),
            "rec_A": float(rec_a),
            "f1_A": float(f1_a),
            "prec_B": float(prec_b),
            "rec_B": float(rec_b),
            "f1_B": float(f1_b),
            "auc_A": ovr_a,
            "auc_B": ovr_b,
        }


@torch.no_grad()
def evaluate_quick(
    model: OrthogonalDisentangler,
    loader: DataLoader,
    device: torch.device,
    max_samples: int = 512,
    lambda_grl: float = 0.0,
) -> Dict[str, float]:
    model.eval()
    seen = 0
    all_logits_a: List[torch.Tensor] = []
    all_logits_b: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images, grl_lambda=lambda_grl)
        all_logits_a.append(outputs["logitsA"].detach().cpu())
        all_logits_b.append(outputs["logitsB"].detach().cpu())
        all_targets.append(targets.detach().cpu())
        seen += images.size(0)
        if seen >= max_samples:
            break
    if len(all_targets) == 0:
        return {"acc_A": 0.0, "auc_A": 0.5, "acc_B": 0.0, "auc_B": 0.5}
    logits_a = torch.cat(all_logits_a, dim=0)
    logits_b = torch.cat(all_logits_b, dim=0)
    targets_t = torch.cat(all_targets, dim=0)
    if logits_a.dim() == 1 or logits_a.size(1) == 1:
        la = logits_a.numpy().ravel(); lb = logits_b.numpy().ravel(); ty = targets_t.numpy().ravel()
        pa = 1.0/(1.0+np.exp(-la)); pb = 1.0/(1.0+np.exp(-lb))
        pred_a=(pa>=0.5).astype(int); pred_b=(pb>=0.5).astype(int)
        metrics = {"acc_A": float(accuracy_score(ty, pred_a)), "acc_B": float(accuracy_score(ty, pred_b)), "auc_A": 0.5, "auc_B": 0.5}
        try: metrics["auc_A"] = float(roc_auc_score(ty, pa))
        except ValueError: metrics["auc_A"] = 0.5
        try: metrics["auc_B"] = float(roc_auc_score(ty, pb))
        except ValueError: metrics["auc_B"] = 0.5
        return metrics
    else:
        pred_a = logits_a.argmax(dim=1).numpy(); pred_b = logits_b.argmax(dim=1).numpy(); ty = targets_t.numpy()
        return {"acc_A": float(accuracy_score(ty, pred_a)), "acc_B": float(accuracy_score(ty, pred_b)), "auc_A": 0.5, "auc_B": 0.5}


def train(
    model: OrthogonalDisentangler,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:
    # Build optimizer params, include external adapter params if not registered under model
    base_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    # LyCORIS wrapper returns a separate network whose params may not be part of model.parameters()
    extra_params = []
    enc = getattr(model, 'encoder', None)
    if enc is not None and hasattr(enc, '_lycoris'):
        try:
            for p in enc._lycoris.parameters():  # type: ignore[attr-defined]
                if p.requires_grad and (p not in base_params):
                    extra_params.append(p)
        except Exception:
            pass
    optimizer = torch.optim.AdamW(
        [{'params': base_params}, {'params': extra_params}] if extra_params else [{'params': base_params}],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    is_binary = (model.out_dim == 1)
    if is_binary:
        # BCE with optional pos_weight
        if args.use_pos_weight and hasattr(train_loader.dataset, "targets"):
            targets_np = np.array(train_loader.dataset.targets)
            if targets_np.ndim == 1 and np.unique(targets_np).size == 2:
                pos = (targets_np == 1).sum()
                neg = (targets_np == 0).sum()
                pos_weight_value = float(neg / max(pos, 1))
                bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
            else:
                bce = nn.BCEWithLogitsLoss()
        else:
            bce = nn.BCEWithLogitsLoss()
        ce = None
    else:
        ce = nn.CrossEntropyLoss(label_smoothing=max(0.0, float(args.label_smoothing)))
        bce = None

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # MINE head and optimizer (if needed)
    mine_head: Optional[MINE] = None
    mine_opt: Optional[torch.optim.Optimizer] = None
    if args.decouple_mode == "mine":
        mine_head = MINE(in_dim=512, hidden=256).to(device)
        mine_opt = torch.optim.Adam(mine_head.parameters(), lr=args.mine_lr)

    # Scheduler: warmup + cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.min_lr
    )

    # Monitor metric: AUC for binary, ACC for multiclass
    best_score = -math.inf
    best_state: Dict[str, torch.Tensor] = {}
    best_ema_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    early = EarlyStopping(patience=args.early_stop_patience, min_delta=args.early_stop_min_delta, mode="max")

    global_step = 0
    # Optional KD teacher on A head
    kd_teacher: Optional[OrthogonalDisentangler] = None
    kd_weight = float(getattr(args, "kd_weight", 0.0))
    if bool(getattr(args, "kd_on_A", False)) and kd_weight > 0.0 and getattr(args, "kd_teacher_path", None):
        import copy
        teacher_path = args.kd_teacher_path
        try:
            kd_teacher = copy.deepcopy(model).to(device)
            raw = torch.load(teacher_path, map_location="cpu")
            if isinstance(raw, dict):
                if "state_dict" in raw and isinstance(raw["state_dict"], dict):
                    sd = raw["state_dict"]
                elif "model" in raw and isinstance(raw["model"], dict):
                    sd = raw["model"]
                else:
                    sd = raw
            else:
                sd = raw  # type: ignore[assignment]
            missing, unexpected = kd_teacher.load_state_dict(sd, strict=False)
            logger.info(
                f"[KD] Loaded teacher from {teacher_path} (missing={len(missing)}, unexpected={len(unexpected)})"
            )
            for p in kd_teacher.parameters():
                p.requires_grad = False
            kd_teacher.eval()
            if hasattr(kd_teacher, "use_gating"):
                kd_teacher.use_gating = False  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"[KD] Failed to load teacher from {teacher_path}: {e}")
            kd_teacher = None

    # Optional SMOKE: print adapter stats before training
    if args.smoke_one_batch:
        try:
            from adapters_vit import assert_adapter_smoke
            assert_adapter_smoke(model.encoder)
        except Exception as e:
            logger.warning(f"[SMOKE] adapter stats check failed: {e}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        # Adjust learning rate for warmup (epoch-level)
        if epoch <= args.warmup_epochs:
            warmup_ratio = epoch / max(1, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * warmup_ratio
        else:
            scheduler.step()

        # Adversarial warmup + optional delay + dynamic lambda
        steps_warm = max(1, int(args.adv_warmup_epochs) * max(1, len(train_loader)))
        adv_ratio = min(1.0, float(global_step) / float(steps_warm)) if args.adv_warmup_epochs > 0 else 1.0
        if epoch < int(getattr(args, "adv_delay_epochs", 0)):
            adv_ratio = 0.0
        # decay towards lambda_adv_final in the last adv_decay_epochs
        adv_decay_epochs = max(0, int(getattr(args, "adv_decay_epochs", 0)))
        adv_decay_prog = 0.0
        if adv_decay_epochs > 0 and epoch >= max(0, args.epochs - adv_decay_epochs):
            adv_decay_prog = min(1.0, (epoch - (args.epochs - adv_decay_epochs)) / float(adv_decay_epochs))
        adv_final = args.lambda_adv if args.lambda_adv_final is None else float(args.lambda_adv_final)
        adv_decay_scale = (1.0 - adv_decay_prog) + adv_decay_prog * (adv_final / max(args.lambda_adv, 1e-8))
        # dynamic scale kept in closure via nonlocal-like list
        if 'dyn_scale' not in locals():
            dyn_scale = 1.0
            dyn_bad = 0
            last_best_accA = -1.0
        curr_grl_lambda = args.grl_lambda * adv_ratio
        curr_lambda_adv = (
            args.lambda_adv
            * adv_ratio
            * adv_decay_scale
            * (dyn_scale if bool(getattr(args, "dynamic_lambda_adv", False)) else 1.0)
        )

        # Orthogonal dynamic weight (delay + decay)
        ortho_ratio = 1.0
        if epoch < int(getattr(args, "ortho_delay_epochs", 0)):
            ortho_ratio = 0.0
        ortho_decay_epochs = max(0, int(getattr(args, "ortho_decay_epochs", 0)))
        ortho_decay_prog = 0.0
        if ortho_decay_epochs > 0 and epoch >= max(0, args.epochs - ortho_decay_epochs):
            ortho_decay_prog = min(1.0, (epoch - (args.epochs - ortho_decay_epochs)) / float(ortho_decay_epochs))
        ortho_final = args.lambda_ortho if args.lambda_ortho_final is None else float(args.lambda_ortho_final)
        ortho_decay_scale = (1.0 - ortho_decay_prog) + ortho_decay_prog * (ortho_final / max(args.lambda_ortho, 1e-8))
        curr_lambda_ortho = args.lambda_ortho * ortho_ratio * ortho_decay_scale

        optimizer.zero_grad(set_to_none=True)
        for images, targets in iterator:
            images = images.to(device, non_blocking=True)
            if is_binary:
                targets_f = targets.float().unsqueeze(1).to(device, non_blocking=True)
            else:
                targets_l = targets.long().to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(images, grl_lambda=curr_grl_lambda, use_adapter=(not args.no_adapter))
                logits_a = outputs["logitsA"]
                logits_b = outputs["logitsB"]
                f_disc = outputs["f_disc"]
                f_indisc = outputs["f_indisc"]
                gate = outputs.get("gate", None)

                # Optionally use fusion head for A
                if args.fusion_concat:
                    logits_a = model.decoderA_fused(torch.cat([f_disc, f_indisc], dim=1))

                if is_binary:
                    loss_cls = bce(logits_a, targets_f)  # type: ignore[arg-type]
                else:
                    loss_cls = ce(logits_a, targets_l)  # type: ignore[arg-type]

                # Disentanglement regularizer
                if args.decouple_mode == "mine":
                    assert mine_head is not None
                    # Step T (MINE) to maximize MI using detached features
                    with torch.no_grad():
                        fd_det = f_disc.detach()
                        fi_det = f_indisc.detach()
                    mine_opt.zero_grad()  # type: ignore[union-attr]
                    mi_pos = mine_head.estimate_mi(fd_det, fi_det)  # type: ignore[union-attr]
                    loss_mine_T = -mi_pos
                    loss_mine_T.backward()
                    mine_opt.step()  # type: ignore[union-attr]
                    # Feature step: minimize MI with current MINE
                    mi_est = mine_head.estimate_mi(f_disc, f_indisc)  # type: ignore[union-attr]
                    loss_decouple = mi_est
                else:
                    # cov/hsic/gram via selective wrapper, optionally reweighted by per-dim importance
                    imp_vec: Optional[torch.Tensor] = None
                    if bool(getattr(args, "kd_on_A", False)):
                        try:
                            grad_fd = torch.autograd.grad(
                                outputs=loss_cls,
                                inputs=f_disc,
                                retain_graph=True,
                                create_graph=False,
                                allow_unused=True,
                            )[0]
                            if grad_fd is not None:
                                imp_vec = grad_fd.abs().mean(dim=0)
                        except Exception:
                            imp_vec = None
                    loss_decouple = decouple_loss_weighted(
                        f_disc,
                        f_indisc,
                        mode=args.decouple_mode,
                        normalize=args.normalize_ortho,
                        topk_ratio=float(getattr(args, "ortho_topk_ratio", 0.0)),
                        importance=imp_vec,
                    )
                # decouple loss defined above; no-op branch removed (decouple_mode always one of cov/hsic/mine/gram)

                # Adversarial branch selection
                if args.adv_mode == "grl":
                    logits_b_eff = logits_b
                elif args.adv_mode in {"cam", "cam_grl"}:
                    # compute gradient wrt f_indisc to build a mask; if A head does not depend on f_indisc, fallback to zeros
                    grad = torch.autograd.grad(
                        outputs=logits_a.sum(), inputs=f_indisc, retain_graph=True, create_graph=True, allow_unused=True
                    )[0]
                    if grad is None:
                        grad = torch.zeros_like(f_indisc)
                    sal = grad.abs().detach()
                    k = max(1, int(sal.size(1) * float(args.cam_topk)))
                    topk_vals, topk_idx = torch.topk(sal, k=k, dim=1)
                    mask = torch.zeros_like(sal)
                    mask.scatter_(1, topk_idx, 1.0)
                    masked = f_indisc * (1.0 - mask)
                    if args.adv_mode == "cam_grl":
                        masked = grad_reverse(masked, curr_grl_lambda)
                    logits_b_eff = model.decoderB(masked)
                elif args.adv_mode == "kl_uniform":
                    # Encourage B head to be uniform (task-insensitive): minimize KL(p||uniform)
                    logits_b_eff = model.decoderB(f_indisc)
                else:  # none
                    logits_b_eff = None

                if logits_b_eff is not None:
                    if args.adv_mode == "kl_uniform":
                        # softmax over classes (for binary: construct 2-class prob)
                        if logits_b_eff.dim() == 2 and logits_b_eff.size(1) > 1:
                            probs = torch.softmax(logits_b_eff, dim=1)
                            C = float(logits_b_eff.size(1))
                        else:
                            s = torch.sigmoid(logits_b_eff)
                            probs = torch.cat([s, 1.0 - s], dim=1)
                            C = 2.0
                        ent = -(probs * (probs + 1e-6).log()).sum(dim=1).mean()
                        loss_adv = -ent + math.log(C)  # KL(p||U) = -H(p) + log C
                    else:
                        if is_binary:
                            if args.adv_loss_type == "focal":
                                # binary focal with logits
                                prob = torch.sigmoid(logits_b_eff)
                                pt = prob * targets_f + (1 - prob) * (1 - targets_f)
                                alpha = float(getattr(args, "adv_focal_alpha", 0.25))
                                gamma = float(getattr(args, "adv_focal_gamma", 2.0))
                                loss_adv_term = (
                                    -alpha * (1 - pt).pow(gamma) * targets_f * torch.log(prob + 1e-8)
                                    - (1 - alpha) * (pt).pow(gamma) * (1 - targets_f) * torch.log(1 - prob + 1e-8)
                                ).mean()
                            else:
                                loss_adv_term = bce(logits_b_eff, targets_f)  # type: ignore[arg-type]
                            loss_entropy = entropy_regularizer(logits_b_eff)
                        else:
                            loss_adv_term = ce(logits_b_eff, targets_l)  # type: ignore[arg-type]
                            loss_entropy = softmax_entropy_regularizer(logits_b_eff)
                        loss_adv = loss_adv_term - args.alpha_entropy * loss_entropy
                else:
                    loss_adv = torch.tensor(0.0, device=device)

                # Optional SupCon on A branch
                loss_supcon = torch.tensor(0.0, device=device)
                if bool(getattr(args, "supcon_on_A", False)):
                    # Build labels for SupCon: binary vs multiclass
                    sup_y = targets_l if not is_binary else targets_f.view(-1).long()
                    loss_supcon = supervised_contrastive_loss(
                        f_disc, sup_y, temperature=float(getattr(args, "supcon_temp", 0.07))
                    )
                loss_gate = torch.tensor(0.0, device=device)
                gate_l1_w = float(getattr(args, "gate_l1", 0.0))
                if bool(getattr(args, "use_gate", False)) and gate_l1_w > 0.0 and gate is not None:
                    loss_gate = gate.abs().mean()
                # Optional KD on A head
                loss_kd = torch.tensor(0.0, device=device)
                if kd_teacher is not None and kd_weight > 0.0:
                    with torch.no_grad():
                        t_out = kd_teacher(images, grl_lambda=0.0, use_adapter=(not args.no_adapter))
                        t_logits = t_out["logitsA"]
                    T = float(getattr(args, "kd_temperature", 1.0))
                    eps = 1e-6
                    if is_binary:
                        s_logit = logits_a / max(T, 1e-6)
                        t_logit = t_logits / max(T, 1e-6)
                        s_prob = torch.sigmoid(s_logit)
                        t_prob = torch.sigmoid(t_logit)
                        s_log = torch.stack(
                            [torch.log(1.0 - s_prob + eps), torch.log(s_prob + eps)],
                            dim=1,
                        )
                        t_p = torch.stack([1.0 - t_prob, t_prob], dim=1)
                        loss_kd = torch.nn.functional.kl_div(s_log, t_p, reduction="batchmean")
                    else:
                        s_log = torch.log_softmax(logits_a / max(T, 1e-6), dim=1)
                        t_p = torch.softmax(t_logits / max(T, 1e-6), dim=1)
                        loss_kd = torch.nn.functional.kl_div(s_log, t_p, reduction="batchmean") * (T * T)

                loss_total = (
                    loss_cls
                    + curr_lambda_ortho * loss_decouple
                    + curr_lambda_adv * loss_adv
                    + float(getattr(args, "lambda_supcon", 0.0)) * loss_supcon
                    + gate_l1_w * loss_gate
                    + kd_weight * loss_kd
                )

                # gradient accumulation
                loss_total = loss_total / max(1, args.accum_steps)

            # Track one trainable param before step (for SMOKE)
            tracked_name, tracked_before = None, None
            if args.smoke_one_batch:
                for n, p in model.named_parameters():
                    if p.requires_grad and p.data.numel() > 0:
                        tracked_name, tracked_before = n, p.data.detach().clone()
                        break

            scaler.scale(loss_total).backward()

            if (global_step + 1) % args.accum_steps == 0:
                # optional gradient clipping (needs unscale before clipping)
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], max_norm=args.clip_grad_norm
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if args.smoke_one_batch:
                    # Verify an update happened
                    if tracked_name is not None:
                        cur = dict(model.named_parameters())[tracked_name].data
                        try:
                            updated = not torch.allclose(tracked_before, cur)
                        except Exception:
                            updated = True
                        logger.info(f"[SMOKE] one-step done. updated={updated} on {tracked_name}")
                    # Exit after first optimizer step
                    return {"smoke": 1.0}

            epoch_loss += loss_total.item() * images.size(0)
            iterator.set_postfix({
                "loss": float(loss_total.item()),
                "cls": float(loss_cls.item()),
                "dec": float(loss_decouple.item() if torch.is_tensor(loss_decouple) else float(loss_decouple)),
                "adv": float(loss_adv.item()),
            })
            global_step += 1

        # Flush remaining grads if accumulation steps not aligned
        if (global_step % max(1, args.accum_steps)) != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        epoch_loss /= len(train_loader.dataset)

        # Quick validation on subset for fast feedback
        quick_metrics = evaluate_quick(model, val_loader, device, max_samples=args.val_quick_samples, lambda_grl=0.0)

        # Full validation possibly at interval; use EMA weights if enabled
        run_full = (epoch % args.val_full_interval == 0)
        if ema is not None and run_full:
            ema.apply_to(model)
        val_metrics = evaluate(model, val_loader, device, lambda_grl=0.0) if run_full else quick_metrics
        if ema is not None and run_full:
            ema.restore(model)

        record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_cls": float(loss_cls.detach().item()) if 'loss_cls' in locals() else None,
            "loss_decouple": float(loss_decouple.detach().item()) if 'loss_decouple' in locals() and torch.is_tensor(loss_decouple) else None,
            "loss_adv": float(loss_adv.detach().item()) if 'loss_adv' in locals() else None,
            "q_acc_A": quick_metrics["acc_A"],
            "q_auc_A": quick_metrics["auc_A"],
            "q_acc_B": quick_metrics["acc_B"],
            "q_auc_B": quick_metrics["auc_B"],
            "acc_A": val_metrics.get("acc_A", float("nan")),
            "auc_A": val_metrics.get("auc_A", float("nan")),
            "acc_B": val_metrics.get("acc_B", float("nan")),
            "auc_B": val_metrics.get("auc_B", float("nan")),
            "bal_acc_A": val_metrics.get("bal_acc_A", float("nan")),
            "bal_acc_B": val_metrics.get("bal_acc_B", float("nan")),
            "prec_A": val_metrics.get("prec_A", float("nan")),
            "rec_A": val_metrics.get("rec_A", float("nan")),
            "f1_A": val_metrics.get("f1_A", float("nan")),
            "prec_B": val_metrics.get("prec_B", float("nan")),
            "rec_B": val_metrics.get("rec_B", float("nan")),
            "f1_B": val_metrics.get("f1_B", float("nan")),
        }
        history.append(record)
        logger.info(json.dumps(record))

        # Update best on full validation only
        if run_full:
            current_score = val_metrics["auc_A"] if is_binary else val_metrics["acc_A"]
            if isinstance(current_score, float) and (not math.isnan(current_score)):
                improved = current_score > best_score
            else:
                improved = False
            if improved:
                best_score = current_score
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                if ema is not None:
                    # store EMA weights too
                    ema.apply_to(model)
                    best_ema_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    ema.restore(model)
            # dynamic lambda_adv scheduling based on acc_A progression
            if bool(getattr(args, 'dynamic_lambda_adv', False)):
                accA_cur = val_metrics.get('acc_A', None)
                if isinstance(accA_cur, float):
                    if accA_cur <= (last_best_accA + float(getattr(args,'dyn_delta',5e-4))):
                        dyn_bad += 1
                    else:
                        last_best_accA = accA_cur
                        dyn_bad = 0
                    if dyn_bad >= int(getattr(args,'dyn_patience',2)):
                        dyn_bad = 0
                        dyn_scale = max(float(getattr(args,'dyn_lambda_min',0.01)), (dyn_scale * 0.7))
                        logger.info(f"[DYNAMIC] Reducing lambda_adv scale to {dyn_scale:.4f}")

            # Early stopping check
            if early.step(current_score):
                logger.info("Early stopping triggered")
                break

    # Load best weights (allow non-strict for adapter/quant buffers)
    if best_state:
        model.load_state_dict(best_state, strict=False)
    # Return also best_ema in case caller wants to save it
    if args.ema and best_ema_state is not None:
        # Attach for saving downstream
        model._best_ema_state = best_ema_state  # type: ignore[attr-defined]

    return history


@torch.no_grad()
def collect_embeddings(
    model: OrthogonalDisentangler,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    disc_feats: List[np.ndarray] = []
    indisc_feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for images, targets in tqdm(loader, desc="Collect embeddings", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images, grl_lambda=0.0)
        disc_feats.append(outputs["f_disc"].detach().cpu().numpy())
        indisc_feats.append(outputs["f_indisc"].detach().cpu().numpy())
        labels.append(targets.numpy())
    disc = np.concatenate(disc_feats, axis=0)
    indisc = np.concatenate(indisc_feats, axis=0)
    targets_np = np.concatenate(labels, axis=0)
    return disc, indisc, targets_np


def plot_embeddings(
    disc: np.ndarray,
    indisc: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_samples = disc.shape[0]
    if num_samples <= 1:
        logger.warning("Not enough samples for embedding plots; skipping visualization.")
        return
    base_perplexity = max(5, num_samples // 3)
    perplexity = max(2, min(30, num_samples - 1, base_perplexity))
    reducer_disc = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reducer_indisc = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    logger.info("Running t-SNE for disentangled embeddings")
    disc_2d = reducer_disc.fit_transform(disc)
    indisc_2d = reducer_indisc.fit_transform(indisc)
    cmap = plt.get_cmap("coolwarm")
    plt.figure(figsize=(8, 6))
    plt.scatter(disc_2d[:, 0], disc_2d[:, 1], c=labels, cmap=cmap, alpha=0.7, s=20)
    plt.title("t-SNE of F_disc")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    disc_path = output_dir / "tsne_f_disc.png"
    plt.savefig(disc_path, dpi=300)
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.scatter(indisc_2d[:, 0], indisc_2d[:, 1], c=labels, cmap=cmap, alpha=0.7, s=20)
    plt.title("t-SNE of F_indisc")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    indisc_path = output_dir / "tsne_f_indisc.png"
    plt.savefig(indisc_path, dpi=300)
    plt.close()
    try:
        import umap  # type: ignore

        logger.info("Running UMAP for disentangled embeddings")
        reducer = umap.UMAP(n_components=2, random_state=42)
        disc_umap = reducer.fit_transform(disc)
        indisc_umap = reducer.fit_transform(indisc)
        plt.figure(figsize=(8, 6))
        plt.scatter(disc_umap[:, 0], disc_umap[:, 1], c=labels, cmap=cmap, alpha=0.7, s=20)
        plt.title("UMAP of F_disc")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(output_dir / "umap_f_disc.png", dpi=300)
        plt.close()
        plt.figure(figsize=(8, 6))
        plt.scatter(indisc_umap[:, 0], indisc_umap[:, 1], c=labels, cmap=cmap, alpha=0.7, s=20)
        plt.title("UMAP of F_indisc")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.tight_layout()
        plt.savefig(output_dir / "umap_f_indisc.png", dpi=300)
        plt.close()
    except ImportError:
        logger.warning("UMAP is not installed; skipping UMAP plots.")


def create_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("puredino")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PureDINO Orthogonal Disentanglement Training")
    parser.add_argument("--data_root", type=str, required=True, help="Root path to dataset (imagefolder root)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to store logs and checkpoints")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for cosine scheduler")
    parser.add_argument("--weight_decay", type=float, default=5e-2, help="AdamW weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs before cosine schedule")
    parser.add_argument("--lambda_ortho", type=float, default=0.5, help="Orthogonal loss weight")
    parser.add_argument("--lambda_ortho_final", type=float, default=None, help="Final orthogonal weight after decay (defaults to lambda_ortho)")
    parser.add_argument("--ortho_delay_epochs", type=int, default=0, help="Epochs to delay orthogonal loss (set lambda_ortho=0)")
    parser.add_argument("--ortho_decay_epochs", type=int, default=0, help="Decay orthogonal weight over last N epochs")
    parser.add_argument("--normalize_ortho", action="store_true", help="L2-normalize features before orthogonal loss")
    parser.add_argument("--lambda_adv", type=float, default=0.5, help="Adversarial loss weight")
    parser.add_argument("--lambda_adv_final", type=float, default=None, help="Final adversarial weight after decay (defaults to lambda_adv)")
    parser.add_argument("--alpha_entropy", type=float, default=0.5, help="Entropy regularizer weight in adversarial loss")
    parser.add_argument("--grl_lambda", type=float, default=1.0, help="Gradient reversal multiplier")
    parser.add_argument("--adv_warmup_epochs", type=int, default=3, help="Warmup epochs for adversarial branch strength")
    parser.add_argument("--adv_delay_epochs", type=int, default=0, help="Epochs to delay adversarial branch (force adv_ratio=0)")
    parser.add_argument("--adv_decay_epochs", type=int, default=0, help="Decay adversarial weight over last N epochs")
    parser.add_argument(
        "--adv_loss_type",
        type=str,
        default="bce",
        choices=["bce", "focal"],
        help="Loss type for adversarial head",
    )
    parser.add_argument("--adv_focal_gamma", type=float, default=2.0, help="Gamma for focal loss in adversarial head")
    parser.add_argument("--adv_focal_alpha", type=float, default=0.25, help="Alpha for focal loss in adversarial head")
    parser.add_argument(
        "--decouple_mode",
        type=str,
        default="cov",
        choices=["cov", "hsic", "mine", "gram"],
        help="Disentanglement regularizer",
    )
    parser.add_argument("--adv_mode", type=str, default="grl", choices=["grl", "none", "cam", "cam_grl", "kl_uniform"], help="Adversarial mechanism")
    parser.add_argument("--cam_topk", type=float, default=0.2, help="CAM inversion: ratio of top dims to suppress in f_indisc")
    parser.add_argument("--no_adapter", action="store_true", help="Disable adapters; use direct projections to 256-d")
    parser.add_argument("--fusion_concat", action="store_true", help="Use [f_disc; f_indisc] fusion for A head (512-d)")
    parser.add_argument("--mine_lr", type=float, default=1e-4, help="Learning rate for MINE network")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--ema", action="store_true", help="Enable EMA of model weights")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--use_pos_weight", action="store_true", help="Enable pos_weight in BCE for class imbalance")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing for CE in multiclass")
    parser.add_argument("--val_quick_samples", type=int, default=512, help="Samples used for quick validation each epoch")
    parser.add_argument("--val_full_interval", type=int, default=1, help="Run full validation every N epochs")
    parser.add_argument("--early_stop_patience", type=int, default=5, help="Early stopping patience on full val AUC")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4, help="Minimum AUC improvement to reset patience")
    parser.add_argument("--grayscale_to_rgb", action="store_true", help="Convert grayscale images to 3-channel RGB")
    parser.add_argument("--use_imagenet_norm", action="store_true", help="Use ImageNet mean/std normalization")
    parser.add_argument("--persistent_workers", action="store_true", help="Enable persistent workers in DataLoader")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor when workers>0")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of training data to use (0<f<=1)")
    parser.add_argument("--aug_extra", action="store_true", help="Enable stronger data augmentation (rotation/color jitter)")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping max norm (<=0 to disable)")
    # SupCon on A branch
    parser.add_argument("--supcon_on_A", action="store_true", help="Enable supervised contrastive loss on A features")
    parser.add_argument("--lambda_supcon", type=float, default=0.05, help="Weight for SupCon loss")
    parser.add_argument("--supcon_temp", type=float, default=0.07, help="Temperature for SupCon loss")
    # Adaptive gating between A/B branches
    parser.add_argument(
        "--use_gate",
        action="store_true",
        help="Enable adaptive gating to route encoder features between A/B branches",
    )
    parser.add_argument(
        "--gate_l1",
        type=float,
        default=0.0,
        help="L1 regularization weight on gating activations (encourages sparsity toward A branch)",
    )
    # Knowledge distillation from baseline teacher on A head
    parser.add_argument(
        "--kd_on_A",
        action="store_true",
        help="Enable KD on A logits from a pre-trained teacher (e.g., baseline)",
    )
    parser.add_argument(
        "--kd_weight",
        type=float,
        default=0.0,
        help="Weight for KD loss on A head (0 to disable even when kd_on_A is set)",
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=1.0,
        help="Temperature for KD soft targets on A head",
    )
    parser.add_argument(
        "--kd_teacher_path",
        type=str,
        default=None,
        help="Path to teacher checkpoint (best_model(.pth) or dict with state_dict) for KD on A head",
    )
    # Orthogonality selective top-k
    parser.add_argument("--ortho_topk_ratio", type=float, default=0.0, help="Apply decouple loss only on top-k dims (0..1)")
    # Adversarial delay and dynamic lambda
    parser.add_argument("--dynamic_lambda_adv", action="store_true", help="Dynamically decrease lambda_adv when accA stalls")
    parser.add_argument("--dyn_lambda_min", type=float, default=0.01, help="Minimum dynamic lambda_adv scale")
    parser.add_argument("--dyn_delta", type=float, default=5e-4, help="Minimum accA improvement to reset patience")
    parser.add_argument("--dyn_patience", type=int, default=2, help="Epoch patience before decreasing lambda_adv scale")
    # Smoke helpers
    parser.add_argument("--smoke_one_batch", action="store_true", help="Run a single optimizer step for adapter smoke test and exit")
    # LyCORIS options
    parser.add_argument("--lyc_rank", type=int, default=8, help="LyCORIS rank (linear_dim)")
    parser.add_argument("--lyc_algo", type=str, default="loha", help="LyCORIS algo: loha|lokr")
    # VeRA ranks
    parser.add_argument("--vera_rank", type=int, default=8, help="VeRA rank")
    # Auto local triggers
    parser.add_argument("--auto_local", action="store_true", help="After global training, auto-launch local per-class runs")
    parser.add_argument("--auto_local_gpus", type=str, default="4,5,6,7", help="Comma-separated GPU indices for auto local")
    parser.add_argument("--auto_local_concurrency", type=int, default=4, help="Max concurrent local runs")
    parser.add_argument("--auto_local_prefix", type=str, default="auto_local", help="Prefix for local output dirs")
    # Dataset options
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagefolder",
        choices=[
            "imagefolder",
            "imagefolder_label",
            "rsna_pneumonia",
        ],
        help="Dataset type",
    )
    parser.add_argument("--imagefolder_balance_sampler", action="store_true", help="Use class-balanced sampler for imagefolder train")
    parser.add_argument("--imagefolder_label_name", type=str, default=None, help="Binary task label (class name) for imagefolder_label dataset")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint to warmstart from (strict=False)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--encoder_name",
        type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="Backbone identifier (DINOv3 function name, timm model id, or Hugging Face repo depending on --encoder_source)",
    )
    parser.add_argument(
        "--encoder_source",
        type=str,
        default="huggingface",
        choices=["dinov3", "timm", "huggingface"],
        help="Source library for the encoder backbone",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Optional path or identifier for pretrained weights (e.g., local .pth for DINOv3 or repo id for Hugging Face)",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=-1,
        help="Encoder feature dimension after projection (-1 to infer from backbone)",
    )
    parser.add_argument("--frozen_blocks", type=int, default=8, help="Number of ViT blocks to freeze")
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training")
    parser.add_argument("--hf_local_files_only", action="store_true", help="Force transformers to use local files only (offline)")
    parser.add_argument("--debug_shapes", action="store_true", help="Log transformer token shapes (B,N,D) during encoding")
    # Fine-tuning strategies (B-ablations)
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="frozen_adapter",
        choices=[
            "frozen_adapter",  # B0: encoder fully frozen + adapters/heads trainable
            "unfreeze_last_k",  # B1: unfreeze last K transformer blocks
            "lora",             # B2: insert LoRA into encoder, base weights frozen
            "vanilla",          # B3: full fine-tune
            "linear_probe",     # B4: only heads trainable
            "prompt",           # B5: feature prompt + heads, encoder frozen
            "adapter_ln",       # B6: train adapters + LayerNorms only
            # Tier-2 experimental modes (mapped to safe fallbacks when unavailable)
            "ia3", "vera", "dora", "qlora", "loha", "lokr", "lycoris", "shira", "paca",
        ],
        help="Fine-tuning / parameter update strategy",
    )
    parser.add_argument("--unfreeze_last_k", type=int, default=2, help="B1: number of last blocks to unfreeze")
    parser.add_argument("--lora_r", type=int, default=8, help="B2: LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16.0, help="B2: LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="B2: LoRA dropout")
    # C-line visual pooling/masking
    parser.add_argument("--c_mode", type=str, default="cls", choices=["cls", "mean", "attn_pool", "token_mask", "attn_mask"], help="C-line visual feature aggregator")
    parser.add_argument("--c_mask_ratio", type=float, default=0.15, help="C3/C4 masking ratio")
    parser.add_argument("--c_attn_temp", type=float, default=1.0, help="C2/C4 attention temperature")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    logger = create_logger(output_dir)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    # Defaults for chest X-ray: grayscale_to_rgb + ImageNet normalization + performant DataLoader
    grayscale_to_rgb = True if args.grayscale_to_rgb else False
    use_imagenet_norm = True if args.use_imagenet_norm else False
    if args.dataset == "imagefolder_label":
        if not args.imagefolder_label_name:
            raise ValueError("--imagefolder_label_name is required when --dataset imagefolder_label")
        train_loader, val_loader, test_loader, train_ds = build_imagefolder_label_dataloaders(
            data_root,
            args.imagefolder_label_name,
            args.batch_size,
            args.num_workers,
            grayscale_to_rgb=grayscale_to_rgb,
            use_imagenet_norm=use_imagenet_norm,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    elif args.dataset == "rsna_pneumonia":
        train_loader, val_loader, test_loader, train_ds = build_rsna_dataloaders(
            data_root,
            args.batch_size,
            args.num_workers,
            grayscale_to_rgb=grayscale_to_rgb,
            use_imagenet_norm=use_imagenet_norm,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            balance_sampler=args.imagefolder_balance_sampler,
            seed=args.seed,
        )
    else:
        train_loader, val_loader, test_loader, train_ds = build_dataloaders(
            data_root,
            args.batch_size,
            args.num_workers,
            grayscale_to_rgb=grayscale_to_rgb,
            use_imagenet_norm=use_imagenet_norm,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            balance_sampler=args.imagefolder_balance_sampler,
            aug_extra=args.aug_extra,
            train_fraction=float(getattr(args, 'train_fraction', 1.0)),
            seed=args.seed,
        )
    feature_dim = args.feature_dim if args.feature_dim > 0 else None
    # Determine output dimension (binary vs multiclass)
    out_dim = 1
    if args.dataset == "imagefolder":
        if hasattr(train_ds, "classes"):
            out_dim = max(1, len(getattr(train_ds, "classes")))
        else:
            out_dim = 1
    else:
        out_dim = 1

    head_style = "linear" if args.finetune_mode == "linear_probe" else "mlp"
    model = OrthogonalDisentangler(
        encoder_name=args.encoder_name,
        feature_dim=feature_dim,
        frozen_blocks=args.frozen_blocks,
        encoder_source=args.encoder_source,
        pretrained_path=args.pretrained_path,
        hf_local_files_only=args.hf_local_files_only,
        c_mode=args.c_mode,
        c_mask_ratio=args.c_mask_ratio,
        c_attn_temp=args.c_attn_temp,
        out_dim=out_dim,
        head_style=head_style,
        debug_shapes=args.debug_shapes,
    )
    if bool(getattr(args, "use_gate", False)):
        model.use_gating = True
    model.to(device)
    if args.resume_path:
        def _select_state_dict(obj: object) -> Dict[str, torch.Tensor]:
            if isinstance(obj, dict):
                # common conventions
                for k in ("state_dict", "model", "network"):
                    if k in obj and isinstance(obj[k], dict):
                        return obj[k]  # type: ignore[return-value]
                return obj  # already a plain state_dict
            raise TypeError("Unexpected checkpoint object type")

        def _strip_module_prefix(k: str) -> str:
            return k[7:] if k.startswith("module.") else k

        try:
            raw = torch.load(args.resume_path, map_location='cpu')
            ckpt_sd = _select_state_dict(raw)
            model_sd = model.state_dict()
            loadable: Dict[str, torch.Tensor] = {}
            skipped = 0
            for k, v in ckpt_sd.items():
                k2 = _strip_module_prefix(k)
                if k2 in model_sd and isinstance(v, torch.Tensor) and model_sd[k2].shape == v.shape:
                    loadable[k2] = v
                else:
                    skipped += 1
            missing_before = sum(1 for k in model_sd.keys() if k not in loadable)
            msg = f"Partial warmstart: loadable={len(loadable)} skipped={skipped} missing_in_model={missing_before}"
            model.load_state_dict(loadable, strict=False)
            logger.info(f"Warmstarted from {args.resume_path}. {msg}")
        except Exception as e:
            logger.warning(f"Failed to (partially) load resume_path {args.resume_path}: {e}")
    # Apply fine-tuning strategy (B-line)
    def set_all_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for p in module.parameters():
            p.requires_grad = requires_grad

    def set_layernorm_trainable(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    if args.finetune_mode == "frozen_adapter":  # B0
        set_all_requires_grad(model.encoder, False)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "unfreeze_last_k":  # B1
        set_all_requires_grad(model.encoder, False)
        blocks = getattr(model.encoder, "blocks", None)
        if blocks is not None:
            blocks_iter = list(blocks)
            k = max(1, args.unfreeze_last_k)
            for blk in blocks_iter[-k:]:
                for p in blk.parameters():
                    p.requires_grad = True
        else:
            set_all_requires_grad(model.encoder, True)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "lora":  # B2
        set_all_requires_grad(model.encoder, False)
        replace_linear_with_lora(model.encoder, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
        # ensure newly created LoRA modules are on the right device
        model.encoder.to(device)
    elif args.finetune_mode == "paca":
        from adapters_vit import inject_paca_vit
        set_all_requires_grad(model.encoder, False)
        model.encoder = inject_paca_vit(model.encoder, r=args.lora_r, alpha=int(args.lora_alpha))
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
        model.encoder.to(device)
    elif args.finetune_mode == "linear_probe":  # B3
        set_all_requires_grad(model.encoder, False)
        set_all_requires_grad(model.adapter_disc, False)
        set_all_requires_grad(model.adapter_indisc, False)
        set_all_requires_grad(model.proj_disc_simple, False)
        set_all_requires_grad(model.proj_indisc_simple, False)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "prompt":  # B4
        set_all_requires_grad(model.encoder, False)
        set_all_requires_grad(model.adapter_disc, False)
        set_all_requires_grad(model.adapter_indisc, False)
        model._use_feat_prompt = True
        model.feat_prompt.requires_grad = True
        args.no_adapter = True
    elif args.finetune_mode == "adapter_ln":  # B5
        set_all_requires_grad(model.encoder, False)
        set_layernorm_trainable(model.encoder)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "dora":  # Tier-2: DoRA (real)
        # Replace encoder Linear layers with DoRALayer from local dora module
        set_all_requires_grad(model.encoder, False)
        import importlib, sys, os
        sys.path.insert(0, str(Path(__file__).resolve().parent / "dora"))
        try:
            dora_mod = importlib.import_module("dora")
        except Exception as e:
            raise ImportError(f"Failed to import local DoRA module: {e}")
        if not hasattr(dora_mod, "replace_linear_with_dora"):
            raise RuntimeError("dora.replace_linear_with_dora not found; ensure local dora/dora.py is present")
        dora_mod.replace_linear_with_dora(model.encoder)
        # ensure newly created DoRA modules are on the right device
        model.encoder.to(device)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "lycoris":  # Tier-2: LyCORIS (LoHa/LoKr)
        from adapters_vit import inject_lycoris_vit
        set_all_requires_grad(model.encoder, False)
        net = inject_lycoris_vit(model.encoder, rank=getattr(args, 'lyc_rank', 8), algo=getattr(args, 'lyc_algo', 'loha'))
        for p in net.parameters():
            p.requires_grad_(True)
        # register LyCORIS network to ensure parameters are part of model.parameters()
        setattr(model.encoder, "_lycoris", net)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "qlora":  # Tier-2: QLoRA (4-bit + LoRA)
        from adapters_vit import inject_qlora_vit
        set_all_requires_grad(model.encoder, False)
        model.encoder = inject_qlora_vit(
            model.encoder,
            r=args.lora_r,
            alpha=int(args.lora_alpha),
            dropout=float(args.lora_dropout),
            target=getattr(args, 'qlora_targets', 'all'),
            compute=getattr(args, 'qlora_compute', 'bf16'),
        )
        # ensure quantized layers and LoRA adapters are on device
        model.encoder.to(device)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "ia3":  # Tier-2: IA3 (PEFT)
        from adapters_vit import inject_ia3_vit
        set_all_requires_grad(model.encoder, False)
        model.encoder = inject_ia3_vit(model.encoder)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "vera":  # Tier-2: VeRA (PEFT)
        from adapters_vit import inject_vera_vit
        set_all_requires_grad(model.encoder, False)
        model.encoder = inject_vera_vit(model.encoder, rank=getattr(args, 'vera_rank', 8))
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "shira":  # Tier-2: SHiRA (PEFT)
        from adapters_vit import inject_shira_vit
        set_all_requires_grad(model.encoder, False)
        model.encoder = inject_shira_vit(model.encoder, rank=getattr(args, 'shira_rank', 8))
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode == "vanilla":  # full fine-tune
        set_all_requires_grad(model.encoder, True)
        model._use_feat_prompt = False
        model.feat_prompt.requires_grad = False
    elif args.finetune_mode in {"ia3", "vera", "dora", "qlora", "loha", "lokr", "lycoris", "shira"}:  # Tier-2 experimental
        # No fallback allowed: require real integration.
        raise NotImplementedError(
            f"finetune_mode={args.finetune_mode} requires real integration (no fallback). "
            "Dependencies may be installed (peft/bitsandbytes/lycoris-lora), but wiring is not enabled yet."
        )
    # else: default frozen_adapter semantics already apply
    
    logger.info("Starting training")
    # Attach dataset to loader to allow pos_weight computation
    history = train(model, train_loader, val_loader, device, args, logger)
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("Evaluating on validation and test sets")
    val_metrics = evaluate(model, val_loader, device, lambda_grl=0.0)
    test_metrics = evaluate(model, test_loader, device, lambda_grl=0.0)
    metrics = {
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Validation metrics: {val_metrics}")
    logger.info(f"Test metrics: {test_metrics}")
    disc, indisc, labels = collect_embeddings(model, test_loader, device)
    np.savez(output_dir / "test_embeddings.npz", f_disc=disc, f_indisc=indisc, labels=labels)
    plot_embeddings(disc, indisc, labels, output_dir, logger)
    ckpt_path = output_dir / "best_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    # Save EMA weights when available
    if hasattr(model, "_best_ema_state"):
        ckpt_ema = output_dir / "best_model_ema.pth"
        torch.save(getattr(model, "_best_ema_state"), ckpt_ema)
        logger.info(f"EMA checkpoint saved to {ckpt_ema}")
    logger.info(f"Checkpoint saved to {ckpt_path}")
    logger.info("Training pipeline complete")

    # Auto-launch local runs after global, if requested
    if args.auto_local:
        import subprocess, time, os, shlex
        def parse_gpus(s: str):
            try:
                return [g.strip() for g in s.split(',') if g.strip()!='']
            except Exception:
                return ["0"]
        gpus = parse_gpus(args.auto_local_gpus)
        max_conc = max(1, int(args.auto_local_concurrency))
        # Determine local targets by dataset type
        tasks = []
        ts = time.strftime("%Y%m%d_%H%M%S")
        resume = str(ckpt_path)
        if args.dataset == "imagefolder" and hasattr(train_ds, 'classes') and len(train_ds.classes) > 1:
            for lab in train_ds.classes:
                out = output_dir.parent / f"{args.auto_local_prefix}_imagefolder_label_{lab}_{ts}"
                cmd = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--data_root", str(data_root),
                    "--dataset", "imagefolder_label",
                    "--imagefolder_label_name", lab,
                    "--output_dir", str(out),
                    "--resume_path", resume,
                    "--epochs", str(args.epochs),
                    "--batch_size", str(args.batch_size),
                    "--num_workers", str(args.num_workers),
                    "--lr", str(args.lr),
                    "--min_lr", str(args.min_lr),
                    "--warmup_epochs", str(args.warmup_epochs),
                    "--lambda_ortho", str(args.lambda_ortho),
                    "--lambda_adv", str(args.lambda_adv),
                    "--alpha_entropy", str(args.alpha_entropy),
                    "--grl_lambda", str(args.grl_lambda),
                    "--adv_warmup_epochs", str(args.adv_warmup_epochs),
                    "--frozen_blocks", str(args.frozen_blocks),
                    "--ema", "--ema_decay", str(args.ema_decay),
                    "--val_quick_samples", str(args.val_quick_samples),
                    "--val_full_interval", str(args.val_full_interval),
                    "--early_stop_patience", str(args.early_stop_patience),
                    "--early_stop_min_delta", str(args.early_stop_min_delta),
                    "--grayscale_to_rgb", "--use_imagenet_norm", "--persistent_workers",
                    "--prefetch_factor", str(args.prefetch_factor),
                    "--clip_grad_norm", str(args.clip_grad_norm),
                    "--amp",
                    "--encoder_source", args.encoder_source,
                    "--encoder_name", args.encoder_name,
                ]
                if args.pretrained_path: cmd += ["--pretrained_path", args.pretrained_path]
                tasks.append((cmd, out))

        procs = []
        gpu_idx = 0
        for cmd, out in tasks:
            out.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            gpu = gpus[gpu_idx % max(1, len(gpus))]
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            with open(out / 'command.txt', 'w', encoding='utf-8') as f:
                f.write(' '.join(shlex.quote(c) for c in cmd))
            logf = open(out / 'console.log', 'a', encoding='utf-8')
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=logf)
            procs.append((p, logf))
            gpu_idx += 1
            # throttle to concurrency
            while len(procs) >= max_conc:
                time.sleep(5)
                # remove finished
                procs = [(pp,ll) for (pp,ll) in procs if pp.poll() is None]
        # wait remaining
        for p, lf in procs:
            p.wait()
            lf.close()


if __name__ == "__main__":
    main()
