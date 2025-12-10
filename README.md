# Incentivizing DINOv3 Adaptation for Medical Vision Tasks via Feature Disentanglement

NOTE: This project is currently under submission to MIDL 2026.

# DINOv3-FD 

A training pipeline for feature disentanglement on vision backbones (ViT/DINOv3), with PEFT adapters (LoRA, IA3, VeRA, LyCORIS, PaCA) and dual heads for task-relevant vs. task-irrelevant features. The main entry point is `disentanglement.py`.

## What's inside
- Orthogonal + regularization disentanglement modules with configurable decouplers (`cov`, `hsic`, `mine`, `gram`) and GRL/entropy max/KL regularizers.
- Multiple fine-tuning modes: Linear Probe, LoRA, Adapter LayerNorm, IA3, LyCORIS (LoHa/LoKr), VeRA, or PACA.
- Dataset support: generic `imagefolder`, RSNA pneumonia; grayscale-to-RGB and ImageNet normalization switches are built in.
- Training utilities: EMA, early stopping, t-SNE/UMAP embedding dumps, and auto local per-class runs.

## Environment
1) Python 3.10+ and CUDA-ready PyTorch.
2) Install deps (core):
```bash
pip install -r dinov3/requirements.txt
pip install timm transformers scikit-learn matplotlib tqdm umap-learn
```
3) For PEFT add-ons as needed:
```bash
pip install peft bitsandbytes lycoris-lora
```
4) Optional local DINOv3 weights: place under `hf_models/facebook/dinov3-vitl16-pretrain-lvd1689m/` or use `--pretrained_path`.

## Data layout
- `imagefolder(ISIC/ODIR)`: `data_root/class_x/*.jpg`.
- `rsna_pneumonia(RSNA)`: point `--data_root` to dataset root.

## Quickstart (i.e. LoRA settings with MINE+KL on ISIC)
```bash
python disentanglement.py \
  --data_root /path/to/isic_full \
  --dataset imagefolder \
  --output_dir outputs/isic_lora \
  --finetune_mode lora --lora_r 8 --lora_alpha 16 --lora_dropout 0.0 \
  --decouple_mode mine --mine_lr 1e-4 \
  --lambda_ortho 0.005 --normalize_ortho \
  --lambda_adv 0.2 --adv_mode kl_uniform --adv_warmup_epochs 10 --alpha_entropy 0.5 \
  --epochs 40 --batch_size 16 --lr 5e-5 --min_lr 1e-6 --warmup_epochs 8 \
  --frozen_blocks 8 --ema --ema_decay 0.999 --clip_grad_norm 1.0 \
  --val_quick_samples 512 --val_full_interval 1 --early_stop_patience 5 \
  --grayscale_to_rgb --use_imagenet_norm --persistent_workers --prefetch_factor 2
```

## Citation
If this helps your work, please cite the upcoming MIDL 2026 submission on 'Incentivizing DINOv3 Adaptation for Medical Vision Tasks via Feature Disentanglement'.

