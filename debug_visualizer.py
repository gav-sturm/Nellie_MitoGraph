"""
Debug Visualizer Module

This module provides utilities for visualizing and saving images at different stages
of the processing pipeline for debugging purposes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class DebugVisualizer:
    """
    Utility class for visualizing and saving images during processing.
    """
    
    def __init__(self, debug_dir="debug_output"):
        """
        Initialize the DebugVisualizer.
        
        Parameters:
        -----------
        debug_dir : str, optional
            Directory to save debug images (default: "debug_output")
        """
        self.debug_dir = debug_dir
        os.makedirs(debug_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.counter = 0
    
    def visualize(self, image, title, save=True, show=False):
        """
        Visualize and optionally save an image.
        
        Parameters:
        -----------
        image : ndarray
            Image to visualize
        title : str
            Title for the image
        save : bool, optional
            Whether to save the image (default: True)
        show : bool, optional
            Whether to display the image (default: False)
        """
        if image is None:
            print(f"Warning: Null image provided for visualization: {title}")
            return
            
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        ax.set_title(f"{title}\nShape: {image.shape}")
        
        # Add grid to show dimensions
        ax.grid(True, color='yellow', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add text with image dimensions
        ax.text(10, 20, f"Shape: {image.shape}", color='white', 
               backgroundcolor='black', fontsize=12)
        
        # Add text showing non-zero pixel percentage
        non_zero_percent = np.count_nonzero(image) / image.size * 100
        ax.text(10, 50, f"Non-zero: {non_zero_percent:.1f}%", color='white',
               backgroundcolor='black', fontsize=12)
        
        # Save the image if requested
        if save:
            self.counter += 1
            filename = f"{self.timestamp}_{self.counter:03d}_{title.replace(' ', '_')}.png"
            filepath = os.path.join(self.debug_dir, filename)
            fig.savefig(filepath, bbox_inches='tight')
            print(f"Debug image saved: {filepath}")
        
        # Show the image if requested
        if show:
            plt.show()
        else:
            plt.close(fig) 