# rag_pipeline/embedder.py

from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, SiglipProcessor, SiglipModel, SiglipTokenizer
from PIL import Image
import torch
import numpy as np


class SBERTEmbedder:
    def __init__(self, model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS", device="cpu"):
        self.model = SentenceTransformer(model_name)
        self.device = device

    def embed_text(self, text: str, convert_to_numpy=True):
        return self.model.encode(text, convert_to_numpy=convert_to_numpy)

    def embed_query(self, text: str):
        return self.embed_text(text, convert_to_numpy=True).astype("float32")

    def embed_documents(self, docs: list[str], convert_to_numpy=True):
        return self.model.encode(docs, convert_to_numpy=convert_to_numpy)

class CLIPImageEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_vec = self.model.get_image_features(**inputs)
        return img_vec.squeeze().cpu().numpy()

class CLIPTextEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

    def embed_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_vec = self.model.get_text_features(**inputs)
        return text_vec.squeeze().cpu().numpy().astype("float32")

class SiglipImageEmbedder:
    def __init__(self, model_name="google/siglip-base-patch16-224", device="cpu"):
        self.device = device
        self.model = SiglipModel.from_pretrained(model_name).to(device)
        self.processor = SiglipProcessor.from_pretrained(model_name)

    def embed_image(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_vec = self.model.get_image_features(**inputs)
        return img_vec.squeeze().cpu().numpy()

class SiglipTextEmbedder:
    def __init__(self, model_name="google/siglip-base-patch16-224", device="cpu"):
        self.device = device
        self.model = SiglipModel.from_pretrained(model_name).to(device)
        self.tokenizer = SiglipTokenizer.from_pretrained(model_name)

    def embed_text(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_vec = self.model.get_text_features(**inputs)
        
        np_vec = text_vec.squeeze().cpu().numpy()
        print(f"[DEBUG] SigLIP text vector shape: {np_vec.shape}, dtype: {np_vec.dtype}")
        # return text_vec.squeeze().cpu().numpy()
        return np_vec.astype("float32")

class HybridEmbedder:
    def __init__(self, text_embedder: SBERTEmbedder, image_embedder: CLIPTextEmbedder):
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder

    def embed_query(self, text: str):
        text_vec = self.text_embedder.embed_text(text)
        image_vec = self.image_embedder.embed_text(text)
        return {
            "text": text_vec,
            "image": image_vec
        }
    
    def embed_documents(self, docs: list[str]):
        return {
            "text": self.text_embedder.embed_documents(docs),
            "image": self.image_embedder.embed_documents(docs)
        }

def get_text_embedder(model_name="sbert", device="cpu"):
    model_name = model_name.lower()

    if "clip" in model_name:

        return CLIPTextEmbedder(model_name=model_name, device=device)

    elif model_name == "sbert":
        return SBERTEmbedder(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS", device=device)

    elif model_name == "jina":
        return SBERTEmbedder(model_name="jinaai/jina-embeddings-v2-base-en", device=device)

    elif "openai" in model_name:
        from langchain.embeddings import OpenAIEmbeddings
        return OpenAIEmbeddings()
    
    elif "siglip" in model_name.lower():
        return SiglipTextEmbedder(model_name=model_name, device=device)

    else:
        raise ValueError(f"Text embedder for model '{model_name}' is not implemented yet.")

def get_image_embedder(model_name="clip", device="cpu"):
    model_name = model_name.lower()

    if model_name == "clip":
        return CLIPImageEmbedder(model_name="openai/clip-vit-base-patch32", device=device)

    elif model_name == "siglip":
        return SiglipImageEmbedder(model_name="google/siglip-base-patch16-224", device=device)

    else:
        raise ValueError(f"Image embedder for model '{model_name}' is not implemented yet.")
    
    # if "clip" in model_name.lower():
    #     return CLIPImageEmbedder(model_name=model_name, device=device)
    # else:
    #     raise ValueError(f"Image embedder for model '{model_name}' is not implemented yet.")


def get_hybrid_embedder():
    text_embedder = SBERTEmbedder()
    image_embedder = CLIPTextEmbedder()
    return text_embedder, image_embedder #HybridEmbedder(sbert, clip)