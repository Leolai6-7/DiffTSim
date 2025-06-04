# Diffusion-Augmented Contrastive Learning for Time-Series

This repository implements a framework for unsupervised representation learning on time-series data using contrastive learning and diffusion-based augmentation. It supports two modes:

- **Diffusion Augmentation**: Generates positive views using a diffusion model (UNet).
- **Traditional Augmentation**: Applies conventional methods like jitter, scaling, and masking.

## 🧠 Core Idea
Use self-supervised SimSiam architecture to learn time-series representations by creating augmented views with either:
- A trained diffusion model (learned generative augmentation)
- Or traditional augmentations (handcrafted)

The learned encoder can be used for downstream tasks such as classification or regression.

---

## 📂 Project Structure
```
.
├── main.py                      # Entry point for pretraining
├── models/
│   ├── simsiam.py              # SimSiam model and loss
│   ├── transformer.py          # Transformer encoder for time series
│   ├── predictor.py            # MLP heads for downstream tasks
│   └── unet.py                 # UNet for diffusion denoising
├── augmentations/
│   ├── diffusion.py            # Diffusion-based augmentation
│   └── traditional.py          # Jitter, scaling, masking
├── train/
│   ├── train_diffusion.py      # SimSiam pretraining with diffusion
│   └── train_traditional.py    # SimSiam pretraining with traditional aug
├── utils/
│   ├── dataset.py              # TimeSeriesDataset class
│   └── evaluate.py             # Accuracy, F1, AUC metrics
```

---

## 🚀 Usage
### Environment Setup
```bash
pip install torch einops scikit-learn diffusers numpy
```

### Data Format
- Input: `.npy` file with shape `(N, C, T)`
- Optional: `.npy` label file with shape `(N,)`

### Pretrain (Diffusion-based)
```bash
python main.py --mode diffusion --data data.npy --epochs 30
```

### Pretrain (Traditional Augmentation)
```bash
python main.py --mode traditional --data data.npy --epochs 30
```

---

## ✍️ Author
This project was built as part of my master's thesis at National Chengchi University, focusing on contrastive learning and diffusion models for time-series understanding.

Feel free to use or reference it in your own research or engineering projects!

---

## 📜 License
MIT License
