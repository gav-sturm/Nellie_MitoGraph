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
from debug_visualizer import DebugVisualizer

# Create a class-level debug visualizer
debug_visualizer = DebugVisualizer()
class CompositeImageCreator:
    """
    Creates and manages composite images from multiple screenshots.
    
    This class is responsible for creating composite images from multiple screenshots,
    arranging them in a layout, and adding annotations.
    """
    
    @staticmethod
    def create_composite_image(images, labels=None, layout='2x4', output_file=None, title=None, bg_color='black',
                               fixed_size=(3840, 2160)):
        """
        Create a composite image by stacking the provided images.
        
        Parameters:
        -----------
        images : list
            List of images to combine
        labels : list, optional
            List of labels for each image
        layout : str, optional
            Layout type ('horizontal', 'vertical', 'grid', or '2x4') (default: '2x4')
        output_file : str, optional
            Path to save the composite image (default: None)
        title : str, optional
            Title to add to the composite image (default: None)
        bg_color : str, optional
            Background color for the composite image (default: 'black')
        fixed_size : tuple, optional
            Fixed size for the output composite image (width, height) (default: (3840, 2160))
            
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
                    images[0],  # Original/Raw
                    images[1],  # Object Labels
                    images[2],  # Branch Labels
                    images[3],  # Depth Encoded
                    images[4],  # Node-Edge
                    images[5],  # Node-Edge 2
                    images[6],  # Topo Graph 1
                    images[7],  # Topo Graph 2
                ]
            else:
                ordered_images = images
            
            # Instead of resizing, use padding to make all images the same size
            # Find the maximum dimensions across all images
            max_h = max(img.shape[0] for img in ordered_images)
            max_w = max(img.shape[1] for img in ordered_images)
            
            # print(f"Maximum image dimensions: {max_h}x{max_w}")
            
            # Pad all images to the same dimensions
            padded_images = []
            for i, img in enumerate(ordered_images):
                # if labels and i < len(labels):
                #     print(f"Processing image: {labels[i]}")
                #     print(f"Original shape: {img.shape}")
                
                # Calculate padding
                pad_h = max_h - img.shape[0]
                pad_w = max_w - img.shape[1]
                
                # Ensure padding is non-negative
                pad_h = max(0, pad_h)
                pad_w = max(0, pad_w)
                
                # Calculate padding for each side (center the image)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                # Create padded image
                if img.ndim == 2:
                    # For grayscale images
                    padded = np.zeros((max_h, max_w), dtype=img.dtype)
                    padded[pad_top:max_h-pad_bottom, pad_left:max_w-pad_right] = img
                else:
                    # For RGB/RGBA images
                    padded = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                    padded[pad_top:max_h-pad_bottom, pad_left:max_w-pad_right] = img
                
                padded_images.append(padded)
                
                # if labels and i < len(labels):
                #     print(f"Padded shape: {padded.shape}")
                #     print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
            # # debug visualize the padded original image
            # debug_visualizer.visualize(padded_images[0], "Padded Original Image for composite image",save=False, show=True)
            # Create rows
            top_row = np.hstack(padded_images[:4])
            bottom_row = np.hstack(padded_images[4:])
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
            
            # Pad all images to the same size
            padded_images = []
            for img in images:
                # Calculate padding
                pad_h = max_h - img.shape[0]
                pad_w = max_w - img.shape[1]
                
                # Ensure padding is non-negative
                pad_h = max(0, pad_h)
                pad_w = max(0, pad_w)
                
                # Calculate padding for each side (center the image)
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                # Create padded image
                if img.ndim == 2:
                    # For grayscale images
                    padded = np.zeros((max_h, max_w), dtype=img.dtype)
                    padded[pad_top:max_h-pad_bottom, pad_left:max_w-pad_right] = img
                else:
                    # For RGB/RGBA images
                    padded = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                    padded[pad_top:max_h-pad_bottom, pad_left:max_w-pad_right] = img
                
                padded_images.append(padded)
            
            # Create rows
            rows_list = []
            for i in range(rows):
                start_idx = i * cols
                end_idx = min(start_idx + cols, n)
                row_images = padded_images[start_idx:end_idx]
                
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
    def add_image_labels(images, labels):
        """
        Create a composite image with labels.
        
        Parameters:
        -----------
        images : list
            List of images to combine
        labels : list
            List of labels for each image
        
        Returns:
        --------
        list
            Images with labels added
        """
        # Create labeled images
        labeled_images = []
        for img, label in zip(images, labels):
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(np.uint8(img))
            
            # Create a new image with label on top of the original image
            new_img = pil_img
            
            # Add label
            draw = ImageDraw.Draw(new_img)
            text_size = pil_img.height * 0.075
            font = CompositeImageCreator.get_universal_font(text_size)  # Use a larger font size
            
            # Calculate text position to center it
            text_width = draw.textlength(label, font=font)
            x = (pil_img.width - text_width) // 2
            y = 20  # add label to the top of the image
            
            # Draw text with a black outline for better visibility
            for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
                draw.text((x+offset[0], y+offset[1]), label, font=font, fill="black")
            
            # Draw the main text
            draw.text((x, y), label, font=font, fill="white")
            new_image = np.array(new_img)
            
            labeled_images.append(new_image)
            # print(f'Label: {label}')
            # print(f'Original image shape: {img.shape}')
            # print(f"Labeled image shape: {new_image.shape}")

        # print(f"Original image shape: {images[0].shape}")
        # print(f"Labeled image shape: {labeled_images[0].shape}")

        # debug_visualizer.visualize(images[0], "Original Image",save=False, show=True)
        # debug_visualizer.visualize(labeled_images[0], "Labeled Image",save=False, show=True)
        
        return labeled_images
    

    @staticmethod
    def add_timestamp(image, timepoint, text_color='white', position='top'):
        """
        Add a timestamp to the image
        
        Parameters:
        -----------
        image : ndarray
            Image to add timestamp to
        timepoint : int
            Timepoint to display
        font_size : int, optional
            Font size for the timestamp (default: 50)
        text_color : str, optional
            Color of the timestamp text (default: 'white')
        bg_color : str, optional
            Background color for the timestamp area (default: 'black')
        position : str, optional
            Position of the timestamp ('top', 'bottom', 'top-left', etc.) (default: 'top')
            
        Returns:
        --------
        ndarray
            Image with timestamp
        """
        # Convert to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Create a PIL Image from the numpy array
        pil_image = Image.fromarray(np.uint8(image))
        
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Try to get a font
        font_size = int(pil_image.height * 0.05)
        
        # Try to use the universal font getter with fallback to default
        font = CompositeImageCreator.get_universal_font(font_size)
        text = f"Timepoint: {timepoint}"

        x = pil_image.width / 2 - font.getbbox(text)[2] / 2
        y = pil_image.height * 0.025
        
        # If we got the default font and need a larger size, create a larger bitmap font
        if font == ImageFont.load_default() and font_size > 20:
            # Create a larger bitmap font by scaling up text
            # We'll handle this by drawing text at normal size and then scaling the result
            scale_factor = font_size / 10  # Assuming default font is roughly 10px
            
            # We'll adjust the text position later to account for this scaling
            x = int(x / scale_factor)
            y = int(y / scale_factor)
            
            # Create a temporary image at a smaller size
            temp_width = int(pil_image.width / scale_factor)
            temp_height = int(pil_image.height / scale_factor)
            temp_img = Image.new('RGBA', (temp_width, temp_height), (0, 0, 0, 0))
            temp_draw = ImageDraw.Draw(temp_img)
            
            # Draw text with default font
            temp_draw.text((x, y), text, font=font, fill=text_color)
            
            # Scale up the temporary image
            temp_img = temp_img.resize((pil_image.width, pil_image.height), Image.LANCZOS)
            
            # Paste the scaled text onto the original image
            pil_image.paste(temp_img, (0, 0), temp_img)
            
            # Return early since we've already drawn the text
            return np.array(pil_image)
        
        # Normal text drawing if we have a proper font
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Convert back to numpy array
        return np.array(pil_image)

    @staticmethod
    def create_composite_image_with_consistent_padding(images, labels=None, layout='2x4', output_file=None, 
                                                      title=None, bg_color='black', fixed_size=(1920, 1080),
                                                      padding_params=None, timepoint=None):
        """
        Create a composite image with consistent padding across timepoints.
        
        Parameters:
        -----------
        images : list
            List of images to combine
        labels : list, optional
            List of labels for each image
        layout : str, optional
            Layout type ('horizontal', 'vertical', 'grid', or '2x4') (default: '2x4')
        output_file : str, optional
            Path to save the composite image (default: None)
        title : str, optional
            Title to add to the composite image (default: None)
        bg_color : str, optional
            Background color for the composite image (default: 'black')
        fixed_size : tuple, optional
            Fixed size for the output composite image (width, height) (default: (3840, 2160))
        padding_params : dict, optional
            Dictionary of padding parameters from a previous call (default: None)
            
        Returns:
        --------
        tuple
            (composite_image, padding_params) - The composite image and padding parameters for reuse
        """

        # If layout is 2x4, use our special layout
        if layout == '2x4':
            # First ensure we have exactly 8 images
            while len(images) < 8:
                placeholder = np.zeros_like(images[0])
                images.append(placeholder)
            
            # Arrange in the specific order requested
            if len(images) >= 8:
                ordered_images = [
                    images[2],  # Original/Raw
                    images[1],  # Object Labels
                    images[0],  # Branch Labels
                    images[3],  # Depth Encoded
                    images[4],  # Node-Edge
                    images[5],  # Node-Edge 2
                    images[6],  # Topo Graph 1
                    images[7],  # Topo Graph 2
                ]
            else:
                ordered_images = images
            
            # Find the maximum dimensions across all images
            max_h = max(img.shape[0] for img in ordered_images)
            max_w = max(img.shape[1] for img in ordered_images)
            
            # print(f"Maximum image dimensions: {max_h}x{max_w}")
            
            # If padding_params is provided, use those instead of calculating new ones
            if padding_params is not None:
                # print("Using provided padding parameters for consistency")
                padded_images = []
                
                for i, img in enumerate(ordered_images):
                    if i < len(padding_params):
                        params = padding_params[i]
                        pad_top, pad_bottom, pad_left, pad_right = params
                        
                        # Create padded image with the same parameters as before
                        if img.ndim == 2:
                            padded = np.zeros((max_h, max_w), dtype=img.dtype)
                        else:
                            # Explicitly create black background (all zeros)
                            padded = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                        
                        # Calculate where to place the image
                        h, w = img.shape[:2]
                        target_h = max_h - pad_top - pad_bottom
                        target_w = max_w - pad_left - pad_right
                        
                        # Resize the image to fit the target dimensions
                        from skimage.transform import resize
                        resized = resize(img, (target_h, target_w), preserve_range=True, anti_aliasing=True)
                        
                        # Place the resized image in the padded canvas
                        if img.ndim == 2:
                            padded[pad_top:pad_top+target_h, pad_left:pad_left+target_w] = resized
                        else:
                            padded[pad_top:pad_top+target_h, pad_left:pad_left+target_w] = resized
                        
                        padded_images.append(padded)
                    else:
                        # If we don't have parameters for this image, use zeros (black)
                        if img.ndim == 2:
                            padded = np.zeros((max_h, max_w), dtype=img.dtype)
                        else:
                            padded = np.zeros((max_h, max_w, img.shape[2]), dtype=img.dtype)
                        padded_images.append(padded)
            else:
                # Calculate padding for each image and store the parameters
                padded_images = []
                padding_params = []
                
                for i, img in enumerate(ordered_images):
                    # if labels and i < len(labels):
                        # print(f"Processing image: {labels[i]}")
                        # print(f"Original shape: {img.shape}")
                    
                    # Calculate padding
                    pad_h = max_h - img.shape[0]
                    pad_w = max_w - img.shape[1]
                    
                    # Ensure padding is non-negative
                    pad_h = max(0, pad_h)
                    pad_w = max(0, pad_w)
                    
                    # Calculate padding for each side (center the image)
                    pad_top = pad_h // 2
                    pad_bottom = pad_h - pad_top
                    pad_left = pad_w // 2
                    pad_right = pad_w - pad_left
                    
                    # Store padding parameters
                    padding_params.append((pad_top, pad_bottom, pad_left, pad_right))
                    
                    # Create padded image with black background
                    if img.ndim == 3 and img.shape[2] == 4:  # RGBA image
                        # Create a black background with full opacity
                        padded = np.zeros((max_h, max_w, 4), dtype=np.uint8)
                        padded[:, :, 3] = 255  # Set alpha channel to fully opaque
                        
                        # Calculate the region to place the image
                        y_start = pad_top
                        y_end = max_h - pad_bottom
                        x_start = pad_left
                        x_end = max_w - pad_right
                        
                        # Place the image
                        padded[y_start:y_end, x_start:x_end, :] = img
                        
                        # For any transparent pixels in the original image, make them black
                        # Create a mask of transparent pixels
                        transparent_mask = padded[:, :, 3] == 0
                        
                        # Set RGB channels to black (0) where transparent
                        padded[transparent_mask, 0:3] = 0
                        
                        # Set alpha to fully opaque
                        padded[:, :, 3] = 255
                    else:
                        # For RGB or grayscale images
                        padded = np.zeros((max_h, max_w), dtype=img.dtype)
                        padded[pad_top:max_h-pad_bottom, pad_left:max_w-pad_right] = img
                    
                    padded_images.append(padded)

                    # debug with before and after image shape
                    # print(f"Label: {labels[i]}")
                    # print(f"Original image shape: {img.shape}")
                    # print(f"Padded image shape: {padded.shape}")
                    
                    # if labels and i < len(labels):
                        # print(f"Padded shape: {padded.shape}")
                        # print(f"Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
            
            # Create rows with black background
            if padded_images[0].ndim == 3:
                channels = padded_images[0].shape[2]
                # For RGB/RGBA images
                top_row = np.zeros((padded_images[0].shape[0], padded_images[0].shape[1] * 4, channels), 
                                  dtype=np.uint8)  # Explicitly use uint8
                
                # If RGBA, set alpha channel to fully opaque
                if channels == 4:
                    top_row[:, :, 3] = 255
                
                bottom_row = np.zeros((padded_images[4].shape[0], padded_images[4].shape[1] * 4, channels), 
                                     dtype=np.uint8)  # Explicitly use uint8
                
                # If RGBA, set alpha channel to fully opaque
                if channels == 4:
                    bottom_row[:, :, 3] = 255
            else:
                # For grayscale images
                top_row = np.zeros((padded_images[0].shape[0], padded_images[0].shape[1] * 4), 
                                  dtype=np.uint8)  # Explicitly use uint8
                bottom_row = np.zeros((padded_images[4].shape[0], padded_images[4].shape[1] * 4), 
                                     dtype=np.uint8)  # Explicitly use uint8

            # add label to the images (we waited for the images to all be the same size first)
            padded_images = CompositeImageCreator.add_image_labels(padded_images, labels)
            
            # Place each image in the rows
            for i in range(4):
                if padded_images[0].ndim == 3:
                    top_row[:, i*padded_images[0].shape[1]:(i+1)*padded_images[0].shape[1], :] = padded_images[i]
                    bottom_row[:, i*padded_images[4].shape[1]:(i+1)*padded_images[4].shape[1], :] = padded_images[i+4]
                else:
                    top_row[:, i*padded_images[0].shape[1]:(i+1)*padded_images[0].shape[1]] = padded_images[i]
                    bottom_row[:, i*padded_images[4].shape[1]:(i+1)*padded_images[4].shape[1]] = padded_images[i+4]

            # Ensure black background for the final composite
            if top_row.ndim == 3:
                # For RGB/RGBA images
                composite = np.zeros((top_row.shape[0] + bottom_row.shape[0], top_row.shape[1], top_row.shape[2]), 
                                    dtype=np.uint8)  # Explicitly use uint8
            else:
                # For grayscale images
                composite = np.zeros((top_row.shape[0] + bottom_row.shape[0], top_row.shape[1]), 
                                    dtype=np.uint8)  # Explicitly use uint8

            # Place the rows in the composite
            composite[:top_row.shape[0], :] = top_row
            composite[top_row.shape[0]:, :] = bottom_row

        else:
            # For other layouts, use the original method
            composite = CompositeImageCreator.create_composite_image(
                images, labels=labels, layout=layout, output_file=None, 
                title=title, bg_color=bg_color, fixed_size=fixed_size
            )
            padding_params = None  # No padding params for other layouts

        # Add timestamp to the composite image
        if timepoint is not None:
            composite = CompositeImageCreator.add_timestamp(composite, timepoint)
        
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
        
        # Before returning the composite image, ensure alpha channel is handled properly
        if composite.ndim == 3 and composite.shape[2] == 4:
            # Convert RGBA to RGB with black background
            rgb_composite = np.zeros((composite.shape[0], composite.shape[1], 3), dtype=np.uint8)
            
            # Create alpha mask (normalized to 0-1)
            alpha = composite[:, :, 3] / 255.0
            
            # Apply alpha compositing over black background
            for c in range(3):
                rgb_composite[:, :, c] = composite[:, :, c] * alpha
            
            composite = rgb_composite

        return composite, padding_params

    @staticmethod
    def get_universal_font(size):
        """Get a font that works across platforms."""
        # Try common font paths across different platforms
        font_paths = [
            "arial.ttf",                                      # Windows
            "/System/Library/Fonts/Helvetica.ttc",            # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", # Linux
            "/usr/share/fonts/TTF/Arial.ttf",                 # Some Linux distros
            "/Library/Fonts/Arial.ttf"                        # Alternative macOS path
        ]
        
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        
        # If all else fails, use the default font
        return ImageFont.load_default()