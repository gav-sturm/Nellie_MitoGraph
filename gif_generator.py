"""
GIFGenerator Module

This module provides the GIFGenerator class for creating GIF animations from frame images.
"""

import os
import numpy as np
from glob import glob
import imageio.v2 as imageio
from imageio.v3 import imwrite as iio_imwrite

class GIFGenerator:
    """
    Creates and manages GIF animations from frame images.
    
    This class is responsible for creating GIF animations from frame images,
    with options for frame rate, loop count, and quality.
    """
    
    def __init__(self, output_dir, frame_pattern="frame_*.png", output_file="animation.gif", frame_rate=5):
        """
        Initialize the GIFGenerator.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the output GIF
        frame_pattern : str, optional
            Pattern to match frame files (default: "frame_*.png")
        output_file : str, optional
            Filename for the output GIF (default: "animation.gif")
        frame_rate : int, optional
            Frame rate in frames per second (default: 5)
        """
        self.output_dir = output_dir
        self.frame_pattern = frame_pattern
        self.output_file = output_file
        self.frame_rate = frame_rate
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_gif_from_files(self, frame_files, duration=None):
        """
        Create a GIF animation from a list of frame files.
        
        Parameters:
        -----------
        frame_files : list
            List of paths to frame files
        duration : float, optional
            Duration of each frame in seconds (default: 1/frame_rate)
            
        Returns:
        --------
        str
            Path to the saved GIF file
        """
        print(f"Creating GIF animation from {len(frame_files)} frames...")
        
        # Set duration from frame rate if not provided
        if duration is None:
            duration = 1.0 / self.frame_rate
        
        # Ensure duration is at least 0.1 seconds for slower animations
        duration = max(duration, 0.1)
        
        print(f"Using frame duration of {duration} seconds (frame rate: {1/duration} fps)")
        
        # Load all frames
        frames = []
        for filename in frame_files:
            frames.append(imageio.imread(filename))
        
        # Find max dimensions
        max_h = max(frame.shape[0] for frame in frames)
        max_w = max(frame.shape[1] for frame in frames)
        
        # Pad frames to the same size if needed
        padded_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            if h != max_h or w != max_w:
                pad_h = max_h - h
                pad_w = max_w - w
                
                # Create padding arrays
                if frame.ndim == 3:  # RGB
                    pad_width = ((0, pad_h), (0, pad_w), (0, 0))
                else:  # Grayscale
                    pad_width = ((0, pad_h), (0, pad_w))
                
                # Pad with zeros (black)
                padded = np.pad(frame, pad_width, mode='constant', constant_values=0)
                padded_frames.append(padded)
            else:
                padded_frames.append(frame)
        
        # Create output path
        output_path = os.path.join(self.output_dir, self.output_file)
        
        # Save as GIF with higher quality and explicit duration in milliseconds
        iio_imwrite(output_path, padded_frames, extension='.gif', plugin='pillow', 
                   loop=0, duration=int(duration * 1000), optimize=False, quality=95, dpi=(200, 200))
        
        print(f"GIF animation saved to {output_path}")
        return output_path
    
    def create_gif_from_pattern(self):
        """
        Create a GIF animation from files matching the pattern.
        
        Returns:
        --------
        str
            Path to the saved GIF file
        """
        # Find files matching the pattern
        frame_files = sorted(glob(os.path.join(self.output_dir, self.frame_pattern)),
                           key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
        
        if not frame_files:
            print(f"No files found matching pattern: {self.frame_pattern}")
            return None
        
        return self.create_gif_from_files(frame_files)
    
    def save_frame(self, image, frame_idx):
        """
        Save a frame image.
        
        Parameters:
        -----------
        image : ndarray
            Frame image to save
        frame_idx : int
            Frame index
            
        Returns:
        --------
        str
            Path to the saved frame file
        """
        frame_file = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.png")
        imageio.imwrite(frame_file, image, quality=95, dpi=(200, 200))
        return frame_file
    
    # create a main fucntion to run on a folder of images
    def main(self):
        """
        Main function to run the GIFGenerator.
        """
        # Find all PNG files in the output directory
        frame_files = sorted(glob(os.path.join(self.output_dir, "*.png")))
        
        # Create the GIF
        self.create_gif_from_files(frame_files)
        
        print(f"GIF animation saved to {os.path.join(self.output_dir, self.output_file)}")

if __name__ == "__main__":
    output_dir = r"/Users/gabrielsturm/Documents/GitHub/Nellie_MG/event1_2024-10-22_13-14-25_/crop1_snout/crop1_nellie_out/nellie_necessities/all_images"
    gif_generator = GIFGenerator(output_dir=output_dir, frame_pattern="frame_*.png", output_file="animation.gif", frame_rate=5)
    gif_generator.main()
