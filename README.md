# 🎨 DF-GAN: Deep-Fusion Generative Adversarial Network
### *Transform Text into Stunning Images* ✨

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-orange.svg)
![Python](https://img.shields.io/badge/Python-3.9-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-brightgreen.svg)

</div>

---

## 📖 Overview

**DF-GAN** is a powerful text-to-image synthesis model that generates high-quality **256×256** images directly from natural language descriptions. This repository provides everything you need: dataset preparation, DAMSM encoders, training scripts, sampling utilities, and comprehensive evaluation pipelines.

> 💡 **Key Features:** Simple architecture, powerful results, FID evaluation, and optional CLIP alignment scoring

---

## 🚀 Getting Started

### 📦 Dataset Preparation

Follow these steps to set up your datasets:

#### 1️⃣ **Download Preprocessed Metadata**
- 🐦 [**Birds Dataset**](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) → Extract to `data/`
- 🖼️ [**COCO Dataset**](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) → Extract to `data/`

#### 2️⃣ **Download Image Data**
- 🐦 [**Birds Images**](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) → Extract to `data/birds/`
- 🖼️ [**COCO2014 Images**](http://cocodataset.org/#download) → Extract to `data/coco/images/`

---

## 🏗️ Model Architecture

### 🔄 Input Pipeline

| Component | Description |
|-----------|-------------|
| 📝 **Text Prompts** | Tokenized via DAMSM vocabulary |
| 🎲 **Noise Vector (z)** | `z_dim = 100`, batch size = 20, truncation = 0.88 |
| 🔤 **DAMSM Encoders** | Text encoder + Image encoder |

### 🎯 Output Pipeline

| Output Type | Details |
|-------------|---------|
| 🖼️ **Generated Images** | PNG format, normalized from [-1, 1] → [0, 255] |
| 💾 **Checkpoints** | Periodic saves in `saved_models/` |
| 📊 **Evaluation Metrics** | FID scores & CLIP alignment (optional) |

---

## ⚙️ Key Components

### 🧠 1. Text Encoder
- **Architecture:** Bi-LSTM  
- **Output:** 256-dim word embeddings + 256-dim sentence embeddings  
- **Purpose:** Converts text descriptions into semantic vectors

### 🖼️ 2. Image Encoder
- **Backbone:** Inception v3  
- **Output:** 256-dim image embeddings  
- **Purpose:** Projects images into shared embedding space

### 🎨 3. Generator Network
📥 Input (z) → 🔄 FC Layer → 📈 8·nf·4×4 → 🔁 G_Blocks (upsampling) → 🎨 RGB → ✅ Tanh <br>

**Text Conditioning Features:**
- ✨ DFBLK + Affine modulation
- 🔗 Concatenates [z, sentence embedding]
- 🎛️ Affine-modulates feature maps in each block

### 🛡️ 4. Discriminator Network

#### **NetD (Image Discriminator)**
- 🌐 Multi-scale CNN architecture
- 📉 Downsampling via D_Block modules

#### **NetC (Conditional Discriminator)**
- 🔀 Combines image features + sentence embeddings
- ✔️ Outputs conditional real/fake logits

#### **Training Stabilization**
| Technique | Purpose |
|-----------|---------|
| ⚖️ **Hinge Loss** | Stable adversarial training |
| ❌ **Mismatched Negatives** | Better text-image alignment |
| 🎯 **MAGP** | Matching-Aware Gradient Penalty |
| 📊 **EMA** | Exponential Moving Average for stable sampling |

---

## 📊 Evaluation Metrics

### 📈 Fréchet Inception Distance (FID)
- Uses 2048-dim InceptionV3 features
- Compares generated images to dataset `.npz` statistics
- **Lower is better** ⬇️

### 🤝 CLIP Alignment (Optional)
- Generates text-to-image sample grids
- Computes cosine similarity via CLIP ViT-B/32
- Results saved in `alignment_samples/`

---

## 🎓 Training

### 💻 Environment Setup

| Component | Specification |
|-----------|---------------|
| 🐍 **Python** | 3.9 |
| 🔥 **PyTorch** | 2.5 with CUDA |
| 🎮 **GPU** | NVIDIA GeForce RTX 4070 (12GB VRAM) |

### 🏃 Start Training

Navigate to the code directory:
```bash
cd DF-GAN/code/
```

#### 🐦 **For Birds Dataset:**
```bash
scripts/train.bat ./cfg/bird.yml
```

#### 🖼️ **For COCO Dataset:**
```bash
scripts/train.bat ./cfg/coco.yml
```

### 🔄 Resume Training

If training is interrupted, configure these parameters in `train.bat`:
- `resume_epoch` → Epoch number to resume from
- `resume_model_path` → Path to checkpoint file

### 💡 Pro Tips

> ⚠️ **Note:** Our evaluation codes don't save synthesized images by default (~30,000 images).  
> To save them, set `save_image: True` in your YAML configuration file.

---

## 🏆 Performance Benchmarks

<div align="center">

| 🗂️ Dataset | 📊 FID Score ⬇️ | ⏱️ Epochs |
|------------|----------------|----------|
| 🐦 **CUB (Birds)** | **24.71** | 230 |
| 🖼️ **MS-COCO** | **15.4** | 290 |

</div>

---

## 🎨 Image Sampling

### 🖼️ Generate Images from Text

1️⃣ **Navigate to code directory:**
```bash
cd DF-GAN/code/
```

2️⃣ **Prepare your text descriptions:**
- Edit `./code/example_captions/dataset_name.txt`
- Add your custom captions (one per line)

3️⃣ **Run sampling:**

#### 🐦 **For Birds:**
```bash
 python src/sample.py --cfg cfg/bird.yml
```

#### 🖼️ **For COCO:**
```bash
 python src/sample.py --cfg cfg/coco.yml
```

### 📁 Output Location
Generated images are saved in: `./code/samples/`

---


## 📄 License

This project is released under the MIT License.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📧 Contact

For questions and feedback:
- 📮 Open an issue on GitHub
- 💬 Join our community discussions

---

<div align="center">

### ⭐ If you find this project useful, please consider giving it a star! ⭐

**Made with ❤️ by the GANners Team**

</div>

