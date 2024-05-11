import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import confusion_matrix


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data):
        num_images, num_pixels = data.shape
        self.mean = np.mean(data, axis=0)  # mean
        centered_data = data - self.mean  # z
        cov_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_indices]
        self.eigenvectors = eigenvectors[:, sorted_indices]
        self.components = self.eigenvectors[:, :self.n_components]

    def compress(self, data):
        centered_data = data - self.mean
        compressed_data = np.dot(centered_data, self.components)
        return compressed_data

    def decompress(self, compressed_data):
        decompressed_data = np.dot(compressed_data, self.components.T) + self.mean
        return decompressed_data

def mse(original, decompressed):
    return np.mean((original - decompressed) ** 2)

def similarity_percentage(original, decompressed):
    mse_value = mse(original, decompressed)
    percentage_error = (mse_value / 255**2) * 100
    similarity = 100 - percentage_error
    return similarity


def evaluate_confusion_matrix(original, reconstructed, threshold=10):
    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()
    binary_original = np.where(original_flat > threshold, 1, 0)
    binary_reconstructed = np.where(reconstructed_flat > threshold, 1, 0)
    return confusion_matrix(binary_original, binary_reconstructed)

def execute_compression():
    def open_file():
        root = tk.Tk()
        root.withdraw()  #
        file_path = filedialog.askopenfilename() 
        return file_path

    def choose_image():
        # Load your image here
        image_path = open_file()  
        image = Image.open(image_path)

        # For grayscale image
        grayscale_image = np.array(image.convert('L'))
        
        # print(grayscale_image)
        
        # Initialize PCA with the number of components
        n_components = 50
        pca = PCA(n_components)

        # Fit PCA and compress the image
        pca.fit(grayscale_image)
        compressed_image = pca.compress(grayscale_image)
        
        # print((compressed_image))
        
        # Decompress the image
        decompressed_image = pca.decompress(compressed_image)
        cm = evaluate_confusion_matrix(grayscale_image, decompressed_image)
        print("Confusion Matrix:\n", cm)
        # Decompress the image
        similarity = similarity_percentage(grayscale_image, decompressed_image)
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(grayscale_image, cmap='gray'), plt.title('Original Grayscale')
        plt.subplot(132), plt.imshow(compressed_image, cmap='gray'), plt.title('Compressed Grayscale')
        plt.subplot(133), plt.imshow(decompressed_image, cmap='gray'), plt.title('Decompressed Grayscale')

        plt.text(30, 30, f"Similarity: {similarity:.2f}%", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

        plt.show()
        Image.fromarray(compressed_image.astype(np.uint8)).save('compressed_grayscale_image.jpg')
        Image.fromarray(decompressed_image.astype(np.uint8)).save('decompressed_grayscale_image.jpg')

        # Convert the image to a NumPy array for color
        colored_image = np.array(image)

        # Separate the RGB channels
        red_channel = colored_image[:, :, 0]
        green_channel = colored_image[:, :, 1]
        blue_channel = colored_image[:, :, 2]

        # Apply PCA to each channel
        pca_red = PCA(n_components)
        pca_green = PCA(n_components)
        pca_blue = PCA(n_components)

        pca_red.fit(red_channel)
        pca_green.fit(green_channel)
        pca_blue.fit(blue_channel)

        # Compress each channel
        compressed_red = pca_red.compress(red_channel)
        compressed_green = pca_green.compress(green_channel)
        compressed_blue = pca_blue.compress(blue_channel)
        compressed_color_image = np.stack((compressed_red, compressed_green, compressed_blue), axis=-1)

        # Decompress each channel
        decompressed_red = pca_red.decompress(compressed_red)
        decompressed_green = pca_green.decompress(compressed_green)
        decompressed_blue = pca_blue.decompress(compressed_blue)
        decompressed_color_image = np.stack((decompressed_red, decompressed_green, decompressed_blue), axis=-1)


        # Calculate the similarity percentage for each channel
        similarity_red = similarity_percentage(red_channel, decompressed_red)
        similarity_green = similarity_percentage(green_channel, decompressed_green)
        similarity_blue = similarity_percentage(blue_channel, decompressed_blue)

        # Calculate the overall similarity for the colored image
        overall_similarity = np.mean([similarity_red, similarity_green, similarity_blue])
        
        print("the blue channel similarity =  " , similarity_blue)
        print("the green channel similarity =  " ,similarity_green)
        print("the red channel similarity =  " ,similarity_red)
        print("the overall      similarity =  " ,overall_similarity)


        Image.fromarray(compressed_color_image.astype(np.uint8)).save('compressed_color_image.jpg')
        Image.fromarray(decompressed_color_image.astype(np.uint8)).save('decompressed_color_image.jpg')

        # Plot and display images
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(colored_image), plt.title('Original Colored')
        plt.subplot(132), plt.imshow(compressed_color_image.astype(np.uint8)), plt.title('Compressed Colored')
        plt.subplot(133), plt.imshow(decompressed_color_image.astype(np.uint8)), plt.title('Decompressed Colored')
        plt.text(30, 30, f"Overall Similarity: {overall_similarity:.2f}%", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        plt.show()

    window = tk.Tk()
    window.geometry("750x750")  

    # Load your background image
    background_image = Image.open("C:\\Users\\Computec\\Desktop\\midnight-owl-robert-farkas.jpg")
    background_image = background_image.resize((750, 750))  
    background_photo = ImageTk.PhotoImage(background_image)

    # Create a canvas for the background image
    background_label = tk.Label(window, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Create a button for choosing an image
    button = tk.Button(window, text="Choose an image", command=choose_image)
    button.pack(pady=(350, 0))  

    window.mainloop()


execute_compression()