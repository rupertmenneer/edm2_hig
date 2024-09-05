import matplotlib.pyplot as plt
import numpy as np

# handles batch or single image, n sets number to display
def plot_array_images(images, n=8, save_path=None):
    
    if images.ndim == 2:
        images = images[np.newaxis]
    if images.ndim == 3:
        images = images[np.newaxis]
    if images.shape[0] < n:
        n = images.shape[0]

    if n == 1:
        img = images[0].transpose(1, 2, 0)
        plt.figure(figsize=(2, 5), dpi=300)
        plt.imshow(img,)
        plt.axis('off')

    else:
        _, axes = plt.subplots(1, n, figsize=(10, 10), dpi=300)
        for i in range(n):         
            img = images[i].transpose(1, 2, 0)
            axes[i].imshow(img)
            axes[i].axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()