import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

def quantize_img(img, num_colors=16):
    img = img.resize((256, 256))

    img = np.array(img)
    h, w, c = img.shape
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    quantized_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    quant_img = quantized_pixels.reshape(h, w, c)

    return quant_img

def plot_imgs(img, quant_img, num_colors: int):
    img = Image.fromarray(img)
    quant_img = Image.fromarray(quant_img)
    # quantized_pil.save(output_path)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(quant_img)
    ax[1].set_title(f"Quantized Image ({num_colors} colors)")
    ax[1].axis("off")

    plt.show()
