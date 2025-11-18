import pandas as pd
import numpy as np
import os
from PIL import Image
from tqdm import tqdm

csv_path = "sign_mnist_train.csv"  
output_dir = "asl_images"          
max_per_label = 100                
img_size = 28                      

data = pd.read_csv(csv_path)

os.makedirs(output_dir, exist_ok=True)
for label in range(26):  # 0–25 for A–Y
    os.makedirs(os.path.join(output_dir, chr(label + 65)), exist_ok=True)

count = {label: 0 for label in range(26)}

for i in tqdm(range(len(data)), desc="Converting CSV to images"):
    row = data.iloc[i]
    label = int(row['label'])
    
    if label in [9, 25]:
        continue
    
    if count[label] >= max_per_label:
        continue
    
    pixels = row.drop('label').to_numpy().reshape(img_size, img_size).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    letter = chr(label + 65)  
    img_path = os.path.join(output_dir, letter, f"{letter}_{count[label]+1}.png")
    img.save(img_path)
    
    count[label] += 1

print("✅ Done! Images saved in:", os.path.abspath(output_dir))
