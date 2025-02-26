"""
CompositeImageCreator Module

This module provides the CompositeImageCreator class for creating composite images
from multiple screenshots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont

class CompositeImageCreator:
    """
    Creates and manages composite images from multiple screenshots.
    
    This class is responsible for creating composite images from multiple screenshots,
    arranging them in a layout, and adding annotations.
    """
    
    @staticmethod
    def create_composite_image(images, layout='horizontal', output_file=None, title=None, bg_color='black',
                               fixed_size=(1920, 1080)):
        """
        Create a composite image by stacking the provided images.
        
        Parameters:
        -----------
        images : list
            List of images to combine
        layout : str, optional
            Layout type ('horizontal', 'vertical', 'grid', or '2x4') (default: 'horizontal')
        output_file : str, optional
            Path to save the composite image (default: None)
        title : str, optional
            Title to add to the composite image (default: None)
        bg_color : str, optional
            Background color for the composite image (default: 'black')
        fixed_size : tuple, optional
            Fixed size for the output composite image (width, height) (default: (1920, 1080))
            
        Returns:
        --------
        ndarray
            Composite image
        """
        if layout == 'horizontal':
            composite = np.hstack(images)
        elif layout == 'vertical':
            composite = np.vstack(images)
        elif layout == '2x4':
            # Create a 2x4 grid with specified layout
            # First ensure we have exactly 8 images
            while len(images) < 8:
                placeholder = np.zeros_like(images[0])
                images.append(placeholder)
            
            # Arrange in the specific order requested:
            # Top row: Raw image, Skeleton, Object image, Branch image
            # Bottom row: Depth encoded, Node-edge, Topo graph1, Topo graph2
            
            # Split images into top and bottom rows in the specified order
            if len(images) >= 8:
                # Order: [Original, Skeleton, Object Labels, Branch Labels, Depth, Node-Edge, Topo Graph, Empty]
                ordered_images = [
                    images[3],  # Original/Raw
                    images[0],  # Skeleton
                    images[2],  # Object Labels
                    images[1],  # Branch Labels
                    images[4],  # Depth Encoded
                    images[5],  # Node-Edge
                    images[6],  # Topo Graph 1
                    np.zeros_like(images[0])  # Empty placeholder for Topo Graph 2
                ]
            else:
                ordered_images = images
            
            # Calculate the target size for each image to achieve the fixed output size
            img_width = fixed_size[0] // 4
            img_height = fixed_size[1] // 2
            
            # Resize all images to the same dimensions
            resized_images = []
            for img in ordered_images:
                if img.shape[:2] != (img_height, img_width):
                    resized = resize(img, (img_height, img_width), 
                                   preserve_range=True, anti_aliasing=True).astype(img.dtype)
                    resized_images.append(resized)
                else:
                    resized_images.append(img)
            
            # Create rows
            top_row = np.hstack(resized_images[:4])
            bottom_row = np.hstack(resized_images[4:])
            composite = np.vstack([top_row, bottom_row])
            
        elif layout == 'grid':
            # Calculate number of rows and columns
            n = len(images)
            cols = int(np.ceil(np.sqrt(n)))
            rows = int(np.ceil(n / cols))
            
            # Create a blank canvas with the right dimensions
            # First, ensure all images have the same dimensions
            max_h = max(img.shape[0] for img in images)
            max_w = max(img.shape[1] for img in images)
            
            # Resize all images to the same size
            resized_images = []
            for img in images:
                if img.shape[:2] != (max_h, max_w):
                    resized = resize(img, (max_h, max_w), 
                                     preserve_range=True, anti_aliasing=True).astype(img.dtype)
                    resized_images.append(resized)
                else:
                    resized_images.append(img)
            
            # Create rows
            rows_list = []
            for i in range(rows):
                start_idx = i * cols
                end_idx = min(start_idx + cols, n)
                row_images = resized_images[start_idx:end_idx]
                
                # Pad the row if needed
                if len(row_images) < cols:
                    # Create blank images for padding
                    padding = [np.zeros((max_h, max_w, 3), dtype=np.uint8) for _ in range(cols - len(row_images))]
                    row_images.extend(padding)
                
                rows_list.append(np.hstack(row_images))
            
            composite = np.vstack(rows_list)
        else:
            raise ValueError(f"Unknown layout type: {layout}")
        
        # If fixed_size is specified and layout is not 2x4, resize the composite image
        if fixed_size and layout != '2x4':
            composite = resize(composite, (fixed_size[1], fixed_size[0]), 
                             preserve_range=True, anti_aliasing=True).astype(np.uint8)
        
        # If output_file is provided, save the composite image
        if output_file:
            # Create figure with specified background color
            fig, ax = plt.subplots(figsize=(20, 12), facecolor=bg_color)
            ax.imshow(composite)
            if title:
                ax.set_title(title, color="white" if bg_color == 'black' else 'black', fontsize=18)
            ax.axis('off')
            
            # Save figure
            fig.savefig(output_file, bbox_inches='tight', facecolor=bg_color)
            plt.close(fig)
            print(f"Composite image saved to {output_file}")
        
        return composite
    
    @staticmethod
    def create_labeled_composite(images, labels, layout='2x4', **kwargs):
        """
        Create a composite image with labels.
        
        Parameters:
        -----------
        images : list
            List of images to combine
        labels : list
            List of labels for each image
        layout : str, optional
            Layout type ('horizontal', 'vertical', 'grid', or '2x4') (default: '2x4')
        **kwargs : dict
            Additional arguments to pass to create_composite_image
            
        Returns:
        --------
        ndarray
            Composite image with labels
        """
        # Create labeled images
        labeled_images = []
        for img, label in zip(images, labels):
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(np.uint8(img))
            
            # Create a new image with space for label
            label_height = 30
            new_img = Image.new('RGB', (pil_img.width, pil_img.height + label_height), (0, 0, 0))
            new_img.paste(pil_img, (0, 0))
            
            # Add label
            draw = ImageDraw.Draw(new_img)
            try:
                # For Windows
                if os.name == 'nt':
                    font = ImageFont.truetype("arial.ttf", 20)
                else:
                    # For Unix/Linux
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except IOError:
                font = ImageFont.load_default()
            
            # Calculate text position to center it
            text_width = draw.textlength(label, font=font)
            position = ((new_img.width - text_width) // 2, pil_img.height + 5)
            draw.text(position, label, font=font, fill=(255, 255, 255))
            
            labeled_images.append(np.array(new_img))
        
        # Create composite image with the specified layout
        return CompositeImageCreator.create_composite_image(labeled_images, layout=layout, **kwargs)