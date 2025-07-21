import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_embeddings(image_paths, batch_size=16):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i+batch_size]]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            #L2-normalize CLIP image embeddings, the correct practice for cosine similarity.
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)  # Normalize embeddings
        embeddings.extend(outputs.cpu().numpy())
    return embeddings

