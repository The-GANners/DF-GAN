
# DF-GAN: A Deep-Fusion Generative Adversarial Network for Text-to-Image Generation
DF-GAN is a simple but powerful text-to-image model that synthesizes images (256 x 256 dimension) directly from natural language descriptions.
This repository includes dataset preparation, DAMSM encoders, training scripts, sampling utilities, and evaluation pipelines (FID & optional CLIP alignment).


## Preparation
### Datasets
1. Download the preprocessed metadata for [birds](https://drive.google.com/file/d/1I6ybkR7L64K8hZOraEZDuHh0cCJw5OUj/view?usp=sharing) [coco](https://drive.google.com/file/d/15Fw-gErCEArOFykW3YTnLKpRcPgI_3AB/view?usp=sharing) and extract them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco2014](http://cocodataset.org/#download) dataset and extract the images to `data/coco/images/`

## Model Overview (Integrated DF-GAN Architecture)
DF-GAN is designed to convert text descriptions into high-quality images using a compact but powerful architecture. Below is the full breakdown of its internal components.

### Inputs
• Text prompts tokenized via DAMSM vocabulary <br>
• Noise vector (z): <br>
 &nbsp;&nbsp; -> default z_dim = 100 <br>
 &nbsp;&nbsp; -> batch size = 20 <br>
 &nbsp;&nbsp; -> truncation = 0.88 <br>
 &nbsp;&nbsp; -> manual_seed <br>
 &nbsp;&nbsp; -> encoder_epoch <br>
• Dataset-specific encoders: <br>
 &nbsp;&nbsp; -> DAMSM text encoder <br>
 &nbsp;&nbsp; -> DAMSM image encoder <br>

 ### Outputs
 • PNG images normalized from [-1,1] → [0,255] <br>
 • Periodic training grids & checkpoints in saved_models/ <br>
 • Evaluation artifacts: <br>
&nbsp;&nbsp; -> FID (2048-d InceptionV3 features vs .npz) <br>
&nbsp;&nbsp; -> Optional CLIP alignment grid + cosine similarity (ViT-B/32) <br>

## Key Components
### 1. Text Encoder <br>
 • Bi-LSTM <br>
 • Produces 256-dim word embeddings and 256-dim sentence embeddings <br>

### 2. Image Encoder <br>
 • Inception v3 backbone <br>
 • Projects images → 256-dim embeddings <br>

 ### 3. Generator <br>
 
 • Pipeline: <br>
 &nbsp;&nbsp; z → FC → 8·nf·4×4 → multiple G_Blocks (upsampling) → to_rgb → Tanh <br>
 • Text Conditioning :  <br>
&nbsp;&nbsp;   -> Uses DFBLK + Affine modulation <br>
&nbsp;&nbsp;   -> Concatenates [z, sentence embedding] <br>
 &nbsp;&nbsp;  -> Affine-modulates feature maps inside each block <br>

  ### 4. Discriminator <br>
 
 • NetD :  <br>
&nbsp;&nbsp;   -> Multi-scale CNN <br>
&nbsp;&nbsp;   -> Downsampling using D_Block <br>
• NetC :  <br>
&nbsp;&nbsp;   -> Concatenates Image features and Spatially replicated sentence embedding <br>
&nbsp;&nbsp;   -> Outputs conditional real/fake logit <br>
• Loss & Regularization:  <br>
&nbsp;&nbsp;   -> Hinge loss <br>
&nbsp;&nbsp;   -> Mismatched text negatives <br>
&nbsp;&nbsp;   -> MAGP (Matching-Aware Gradient Penalty) <br>
• Stabilization:  <br>
&nbsp;&nbsp;   -> EMA (Exponential Moving Average) of generator weights <br>
&nbsp;&nbsp;   -> EMA used for sampling and FID <br>

## Evaluation 
### Frechet Inception Distance (FID)
&nbsp;&nbsp;   -> 2048-dim InceptionV3 features <br>
&nbsp;&nbsp;   -> Compares to dataset .npz stats <br>

### Optional CLIP Alignment
&nbsp;&nbsp;   -> Generates grid of text → image samples <br>
&nbsp;&nbsp;   -> Computes cosine scores via CLIP ViT-B/32 <br>
&nbsp;&nbsp;   -> Saved in alignment_samples/ <br>

## Training
  ```
  cd DF-GAN/code/
  ```
### Train the DF-GAN model
  - For bird dataset: `scripts/train.bat ./cfg/bird.yml`
  - For coco dataset: `scripts/train.bat ./cfg/coco.yml`
    
### Resume training process
If your training process is interrupted unexpectedly, set **resume_epoch** and **resume_model_path** in train.bat to resume training.

### Some tips
- Our evaluation codes do not save the synthesized images (about 3w images). If you want to save them, set **save_image: True** in the YAML file.

### Performance


| Model | CUB-FID↓ | COCO-FID↓ |
| --- | --- | --- |
| DF-GAN | **24.71** | **15.41** |



## Sampling
  ```
  cd DF-GAN/code/
  ```
  
### Synthesize images from your text descriptions
  - Replace your text descriptions into the ./code/example_captions/dataset_name.txt
  - For bird dataset: `bash scripts/sample.sh ./cfg/bird.yml`
  - For coco dataset: `bash scripts/sample.sh ./cfg/coco.yml`

The synthesized images are saved at ./code/samples.

