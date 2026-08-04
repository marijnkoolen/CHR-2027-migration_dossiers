"""Utility classes for page image feature extraction using VGG16."""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np

class VGG16FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        # Load VGG16 with ImageNet weights
        weights = models.VGG16_Weights.IMAGENET1K_V1
        self.model = models.vgg16(weights=weights).to(self.device)
        
        # We want the 4096-D features from the second-to-last layer (fc7)
        # In torchvision's VGG16, classifier is:
        # [0]: Linear(in_features=25088, out_features=4096, bias=True)
        # [1]: ReLU(inplace=True)
        # [2]: Dropout(p=0.5, inplace=False)
        # [3]: Linear(in_features=4096, out_features=4096, bias=True)
        # [4]: ReLU(inplace=True)
        # [5]: Dropout(p=0.5, inplace=False)
        # [6]: Linear(in_features=4096, out_features=1000, bias=True)
        
        # Keep everything up to the second 4096 layer
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-3])
        self.model.eval()
        
        # Standard ImageNet transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def encode_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        """Extract features for a list of image paths."""
        n = len(paths)
        features = np.empty((n, 4096), dtype=np.float32)
        
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_tensors = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert('RGB')
                        batch_tensors.append(self.preprocess(img))
                    except Exception as e:
                        print(f"Error loading {p}: {e}")
                        # Fallback to zero vector if image fails
                        batch_tensors.append(torch.zeros((3, 224, 224)))
                
                x = torch.stack(batch_tensors).to(self.device)
                feat = self.model(x)
                features[i:i + len(batch_paths)] = feat.cpu().numpy()
                
                if (i + batch_size) % (batch_size * 5) == 0 or (i + batch_size) >= n:
                    print(f"Processed {min(i + batch_size, n)} / {n} images")
                    
        return features

class EfficientNetFeatureExtractor:
    def __init__(self, device='cpu'):
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.device = torch.device(device)
        self.weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = efficientnet_b0(weights=self.weights).to(self.device)
        self.model.classifier = nn.Identity()
        self.model.eval()
        self.preprocess = self.weights.transforms()

    def encode_paths(self, paths: list[Path], batch_size: int = 16) -> np.ndarray:
        n = len(paths)
        features = np.empty((n, 1280), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch_paths = paths[i:i + batch_size]
                batch_tensors = []
                for p in batch_paths:
                    try:
                        img = Image.open(p).convert('RGB')
                        batch_tensors.append(self.preprocess(img))
                    except Exception as e:
                        print(f"Error loading {p}: {e}")
                        batch_tensors.append(torch.zeros((3, 224, 224)))
                
                x = torch.stack(batch_tensors).to(self.device)
                feat = self.model(x)
                features[i:i + len(batch_paths)] = feat.cpu().numpy()
                if (i + batch_size) % (batch_size * 5) == 0:
                    print(f"  EfficientNet: {min(i + batch_size, n)} / {n}")
        return features

class BERTFeatureExtractor:
    def __init__(self, device='cpu', model_name='bert-base-uncased'):
        from transformers import AutoTokenizer, AutoModel
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode_texts(self, text_paths: list[Path], batch_size: int = 32) -> np.ndarray:
        n = len(text_paths)
        features = np.zeros((n, 768), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                chunk_paths = text_paths[i:i + batch_size]
                texts = []
                for p in chunk_paths:
                    if p.exists():
                        try:
                            # Read first 1000 chars to avoid huge files
                            texts.append(p.read_text(encoding='utf-8')[:1000])
                        except Exception:
                            texts.append("")
                    else:
                        texts.append("")
                
                enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                # Use CLS token
                cls = out.last_hidden_state[:, 0, :].cpu().numpy()
                features[i:i + len(texts)] = cls
                if (i + batch_size) % (batch_size * 5) == 0:
                    print(f"  BERT: {min(i + batch_size, n)} / {n}")
        return features
