import os
import glob
import numpy as np
import imageio.v2 as imageio  # Use imageio.v2 to avoid deprecation warnings
from tqdm import tqdm

class GIFGenerator:
    def __init__(self, input_folder, pattern="components_frame_*.png",
                 output_gif="components_all_frames.gif", frame_rate=12):
        """
        Initialize the GIFGenerator.

        Parameters:
            input_folder (str): Folder containing the frame images.
            pattern (str): Glob pattern to match frame images.
            output_gif (str): Name of the output GIF file.
            frame_rate (float): Frame rate (frames per second) for the GIF.
        """
        self.input_folder = input_folder
        self.pattern = pattern
        self.output_gif = output_gif
        self.frame_rate = frame_rate

    def create_gif(self):
        """
        Create an animated GIF from images in the input folder.
        All images are padded to have the same dimensions if necessary.
        """
        # Build the full file pattern and get the sorted list of images
        file_pattern = os.path.join(self.input_folder, self.pattern)
        png_files = sorted(glob.glob(file_pattern),
                           key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))

        if not png_files:
            print("No frame images found to create GIF.")
            return

        images = []
        for filename in png_files:
            img = imageio.imread(filename)
            images.append(img)

        # Determine the maximum height and width among all images
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)

        # Pad each image so they all share the same dimensions.
        padded_images = []
        for img in tqdm(images):
            h, w = img.shape[:2]
            pad_h = max_h - h
            pad_w = max_w - w

            # For RGB or RGBA images, pad the last dimension with 0 (or 255 for a white background)
            if img.ndim == 3:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
                padded = np.pad(img, pad_width, mode='constant', constant_values=255)
            else:
                pad_width = ((0, pad_h), (0, pad_w))
                padded = np.pad(img, pad_width, mode='constant', constant_values=255)
            padded_images.append(padded)

        # Create the output GIF file path.
        output_path = os.path.join(self.input_folder, self.output_gif)
        # Duration (seconds per frame) is the inverse of the frame rate.
        duration = 1.0 / self.frame_rate
        # Set loop=0 to loop indefinitely.
        imageio.mimsave(output_path, padded_images, duration=duration, loop=0)
        print(f"Saved animated GIF to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Set the folder where your frame images are stored.
    global_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2025-02-13_yeast_mitographs\event1_2024-10-22_13-14-25_\crop1_snout\crop1_nellie_out\nellie_necessities"
    input_folder = os.path.join(global_path, "output_graphs")
    # Create the GIFGenerator instance.
    gif_gen = GIFGenerator(input_folder=input_folder, frame_rate=3)
    # Generate the GIF.
    gif_gen.create_gif()
