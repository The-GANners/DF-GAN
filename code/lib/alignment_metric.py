import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import clip
import torchvision.transforms as transforms
from torchvision.utils import make_grid

class CLIPAlignmentMetric:
    def __init__(self, device="cpu"):
        print("Initializing CLIP Alignment Metric...")
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        
        # Create a transform to convert tensor to PIL image
        self.tensor_to_pil = transforms.ToPILImage()
        
    def compute_similarity(self, images, texts):
        """
        Compute CLIP similarity between images and texts
        images: tensor of shape (batch_size, 3, H, W) in range [-1, 1]
        texts: list of strings
        Returns: tensor of shape (batch_size,) containing similarity scores
        """
        # Convert images from [-1, 1] to [0, 1]
        images_norm = (images + 1) / 2
        
        # Process images for CLIP
        processed_images = []
        for img in images_norm:
            # Convert tensor to PIL
            pil_img = self.tensor_to_pil(img.cpu())
            # Apply CLIP preprocessing
            processed_img = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            processed_images.append(processed_img)
            
        processed_images = torch.cat(processed_images, dim=0)
        
        # Encode images and texts
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_images)
            text_features = self.clip_model.encode_text(clip.tokenize(texts).to(self.device))
            
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute cosine similarity
        similarities = torch.sum(image_features * text_features, dim=-1)
        return similarities.cpu()
    
    def create_alignment_visualization(self, netG, text_encoder, prompts, wordtoix, z_dim, truncation=True, trunc_rate=0.88, images_per_prompt=5):
        """
        Create a visualization grid with multiple prompts and images per prompt
        prompts: list of text prompts
        Returns: PIL image containing the visualization grid
        """
        from lib.utils import tokenize, sort_example_captions, prepare_sample_data, truncated_noise
        
        # Determine the device of text encoder
        text_encoder_device = next(text_encoder.parameters()).device
        print(f"Text encoder is on device: {text_encoder_device}")
        netG_device = next(netG.parameters()).device
        print(f"Generator is on device: {netG_device}")
        
        # Tokenize prompts
        tokenizer = get_tokenizer()
        all_captions = []
        all_cap_lens = []
        
        # Process each prompt
        for prompt in prompts:
            tokens = tokenizer.tokenize(prompt.lower())
            rev = []
            for t in tokens:
                t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            all_captions.append(rev)
            all_cap_lens.append(len(rev))
        
        # Prepare text embeddings - use text_encoder's device instead of self.device
        captions, cap_lens, sorted_indices = sort_example_captions(all_captions, all_cap_lens, text_encoder_device)
        with torch.no_grad():
            hidden = text_encoder.init_hidden(captions.size(0))
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            sent_emb = sent_emb.detach()
        
        # Reorder the embeddings to match the original prompt order
        ordered_sent_emb = torch.zeros_like(sent_emb)
        for idx, sort_idx in enumerate(sorted_indices):
            ordered_sent_emb[sort_idx] = sent_emb[idx]
        
        # Generate images for each prompt
        netG.eval()
        all_images = []
        all_similarities = []
        
        with torch.no_grad():
            for prompt_idx, prompt in enumerate(prompts):
                prompt_images = []
                
                # Generate multiple images per prompt
                for i in range(images_per_prompt):
                    if truncation:
                        noise = truncated_noise(1, z_dim, trunc_rate)
                        noise = torch.tensor(noise, dtype=torch.float).to(netG_device)  # Use netG's device
                    else:
                        noise = torch.randn(1, z_dim).to(netG_device)  # Use netG's device
                    
                    # Generate image
                    fake_img = netG(noise, ordered_sent_emb[prompt_idx].unsqueeze(0))
                    prompt_images.append(fake_img)
                
                # Compute similarities for this prompt's images
                prompt_images_tensor = torch.cat(prompt_images, dim=0)
                
                # Move to CPU for CLIP processing
                prompt_images_tensor_cpu = prompt_images_tensor.cpu()
                sims = self.compute_similarity(prompt_images_tensor_cpu, [prompt] * images_per_prompt)
                
                all_images.append(prompt_images_tensor_cpu)  # Store CPU tensors
                all_similarities.append(sims)
        
        # Combine all images and compute overall alignment score
        all_images_tensor = torch.cat(all_images, dim=0)  # Already on CPU
        all_sims_tensor = torch.cat(all_similarities, dim=0)
        avg_similarity = all_sims_tensor.mean().item()
        
        # Create grid visualization with text
        grid = create_text_image_grid(all_images_tensor, prompts, all_similarities, images_per_prompt)
        
        return grid, avg_similarity

def get_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    return RegexpTokenizer(r'\w+')

def create_text_image_grid(images, prompts, similarities, images_per_prompt):
    """
    Create a grid with text labels for the prompts and similarity scores
    """
    # Normalize images to [0, 1]
    images = (images + 1) / 2
    
    # Create a grid of images
    nrow = images_per_prompt
    grid = make_grid(images, nrow=nrow, padding=5, normalize=False)
    grid_pil = transforms.ToPILImage()(grid)
    
    # Get dimensions
    width, height = grid_pil.size
    
    # Create a new image with extra space for text
    margin_top = 30  # Space for prompt text
    margin_bottom = 20  # Space for similarity scores
    result = Image.new('RGB', (width, height + (margin_top + margin_bottom) * len(prompts)), color='white')
    result.paste(grid_pil, (0, 0))
    
    # Add text
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add prompts and similarity scores
    for i, prompt in enumerate(prompts):
        # Calculate y-position for this prompt's section
        y_pos = i * (images_per_prompt * (height // len(prompts)))
        
        # Add prompt text
        if len(prompt) > 50:
            prompt = prompt[:47] + "..."
        draw.text((10, y_pos + 5), f"Prompt: {prompt}", fill="black", font=font)
        
        # Add similarity scores
        for j in range(images_per_prompt):
            sim_score = similarities[i][j].item()
            x_pos = j * (width // images_per_prompt) + (width // images_per_prompt // 2) - 20
            draw.text((x_pos, y_pos + height // len(prompts) - 20), 
                      f"{sim_score:.3f}", fill="blue", font=font)
    
    return result
