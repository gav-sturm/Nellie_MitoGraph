"""
ScreenshotManager Module

This module provides the ScreenshotManager class for capturing and processing screenshots
of 3D volumetric data with a given viewpoint.
"""

import os
import time
import numpy as np
import napari
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

class ScreenshotManager:
    """
    Manages the capture and processing of screenshots.
    
    This class is responsible for capturing screenshots of 3D volumes with a given viewpoint,
    and for handling preprocessing and transformation of these screenshots.
    """
    
    def __init__(self, viewpoint_selector):
        """
        Initialize the ScreenshotManager.
        
        Parameters:
        -----------
        viewpoint_selector : ViewpointSelector
            ViewpointSelector instance with captured view parameters
        """
        self.viewpoint_selector = viewpoint_selector
    
    def capture_screenshot(self, volume, layer_type='image', wait_time=0.2, scale=1.5, intensity_percentile=50):
        """
        Open a 3D viewer for the given volume, set the camera to the captured view,
        and capture a screenshot.
        
        Parameters:
        -----------
        volume : ndarray
            Volume data to visualize
        layer_type : str, optional
            Type of layer to add ('image' or 'labels') (default: 'image')
        wait_time : float, optional
            Time to wait before capturing screenshot (default: 0.2)
        scale : float, optional
            Scale factor for the screenshot (default: 1.5)
        intensity_percentile : int, optional
            Percentile cutoff for intensity thresholding (default: 50)
            
        Returns:
        --------
        ndarray
            Screenshot image
        """
        try:
            if volume is None:
                print("Warning: Null volume provided to capture_screenshot")
                return self.create_placeholder_image("Volume data\nnot available")
                
            viewer = napari.Viewer(ndisplay=3)
            
            # Use higher quality rendering for original images
            if layer_type == 'labels':
                viewer.add_labels(volume, name="Modality")
            else:
                # For original images, use better rendering settings
                processed_volume = volume.copy()
                
                # Calculate threshold value based on user-specified percentile of non-zero values
                non_zero_values = processed_volume[processed_volume > 0]
                if len(non_zero_values) > 0:
                    threshold = np.percentile(non_zero_values, intensity_percentile)
                    # Set values below threshold to zero
                    processed_volume[processed_volume < threshold] = 0
                
                # Higher quality rendering
                viewer.add_image(processed_volume, name="Modality", rendering='mip', 
                               contrast_limits=[0, processed_volume.max()])
    
            # Set camera to captured view if available
            if self.viewpoint_selector.has_view:
                angles, center = self.viewpoint_selector.captured_view
                viewer.camera.angles = angles
                viewer.camera.center = center
                
                # Apply the exact same zoom level that was captured
                if self.viewpoint_selector.zoom_level is not None:
                    viewer.camera.zoom = self.viewpoint_selector.zoom_level
            else:
                print("No captured viewpoint; using default camera.")
            
            # Use longer delay for original images to ensure proper rendering
            if layer_type == 'image':
                time.sleep(wait_time)  # Longer delay for original images
            else:
                time.sleep(wait_time/2)  # Standard delay for other images
            
            # Use higher scale factor for better resolution
            img = viewer.screenshot(canvas_only=True, scale=scale)
            
            viewer.close()
            return img
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return self.create_placeholder_image(f"Error capturing\nscreenshot:\n{str(e)}")
    
    def generate_depth_encoded_image(self, volume):
        """
        Generate a depth-encoded image of the volume using points colored by Z-depth.
        
        Parameters:
        -----------
        volume : ndarray
            Volume data
            
        Returns:
        --------
        ndarray
            Depth-encoded image
        """
        try:
            if volume is None or volume.size == 0:
                print("Warning: Empty volume provided to generate_depth_encoded_image")
                return self.create_placeholder_image("Empty volume\nfor depth encoding")
                
            # First try using the 3D points approach
            try:
                return self._generate_depth_encoded_3d(volume)
            except Exception as e:
                print(f"Error generating 3D depth image: {e}. Trying 2D approach...")
                
            # Fall back to 2D projection if 3D fails
            return self._create_simple_depth_image(volume)
                
        except Exception as e:
            print(f"Error generating depth encoded image: {e}")
            return self.create_placeholder_image(f"Error generating\ndepth image:\n{str(e)}")
    
    def _generate_depth_encoded_3d(self, volume):
        """
        Generate a depth-encoded image using 3D points with colors.
        """
        viewer = napari.Viewer(ndisplay=3)
        
        # Get coordinates of volume voxels
        volume_mask = (volume > 0)
        z_coords, y_coords, x_coords = np.where(volume_mask)
        
        if len(z_coords) == 0:
            raise ValueError("No non-zero voxels in volume")
        
        # Use colormap approach
        norm = plt.Normalize(vmin=z_coords.min(), vmax=z_coords.max())
        cmap = plt.cm.plasma
        colors = cmap(norm(z_coords))
        
        # Add points layer
        points = np.column_stack((z_coords, y_coords, x_coords))
        points_layer = viewer.add_points(
            points,
            name="Depth-Encoded",
            size=4,  # Slightly larger for better visibility
            face_color=colors,
            edge_color="transparent",
            shading="none"  # Disable shading which might affect visibility
        )
        
        # Set camera to captured viewpoint
        if self.viewpoint_selector.has_view:
            angles, center = self.viewpoint_selector.captured_view
            viewer.camera.angles = angles
            viewer.camera.center = center
            
            # Don't reset view as it changes the perspective
            # Just ensure we're using same scale/zoom as other modalities
            try:
                viewer.camera.zoom = self.viewpoint_selector.zoom_level
            except:
                pass  # If zoom setting fails, use default
        
        time.sleep(0.5)  # Increased wait time for better rendering
        img_depth = viewer.screenshot(canvas_only=True, scale=1.5)
        
        viewer.close()
        
        # Check if the image is empty (all black)
        if np.mean(img_depth) < 0.01:  # Very dark image
            print("Warning: Depth image appears to be empty. Using alternative approach.")
            raise ValueError("Empty depth image detected")
            
        return img_depth
    
    def _create_simple_depth_image(self, volume):
        """
        Create a simple depth-encoded image as a fallback.
        
        Parameters:
        -----------
        volume : ndarray
            Volume data
            
        Returns:
        --------
        ndarray
            Simple depth-encoded image
        """
        Z, Y, X = volume.shape
        
        # Create a figure with the depth-encoded projection
        fig = plt.figure(figsize=(10, 8), facecolor='black')
        ax = fig.add_subplot(111)
        
        # Create a depth map using maximum intensity projection with z-value as color
        depth_map = np.zeros((Y, X), dtype=float)
        colored_map = np.zeros((Y, X, 4), dtype=float)
        
        # For each z-slice, update the depth map
        for z in range(Z):
            # Get the normalized z-value for coloring
            z_norm = z / max(1, Z-1)
            # Get color from plasma colormap
            color = plt.cm.plasma(z_norm)
            
            # Get mask for this slice
            mask = volume[z] > 0
            
            # Update depth map - record the deepest visible point
            depth_map = np.where(mask & (depth_map == 0), z, depth_map)
            
            # Apply color to visible voxels
            for c in range(4):
                colored_map[:, :, c] = np.where(mask & (colored_map[:, :, c] == 0), 
                                               color[c], colored_map[:, :, c])
        
        # Display the colored depth map
        ax.imshow(colored_map)
        ax.set_title("Depth Encoded", color='white', fontsize=14)
        ax.axis('off')
        
        # Convert figure to image
        fig.tight_layout()
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img_data
    
    def create_placeholder_image(self, text, width=800, height=800, bg_color=(30, 30, 30), text_color=(200, 200, 200)):
        """
        Create a placeholder image with text.
        
        Parameters:
        -----------
        text : str
            Text to display on the placeholder
        width : int, optional
            Width of the placeholder image (default: 800)
        height : int, optional
            Height of the placeholder image (default: 800)
        bg_color : tuple, optional
            Background color (R, G, B) (default: (30, 30, 30))
        text_color : tuple, optional
            Text color (R, G, B) (default: (200, 200, 200))
            
        Returns:
        --------
        ndarray
            Placeholder image
        """
        # Create a blank image
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to get a font
        try:
            # For Windows
            if os.name == 'nt':
                font = ImageFont.truetype("arial.ttf", 36)
            else:
                # For Unix/Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        except IOError:
            font = ImageFont.load_default()
        
        # Calculate text position to center it
        lines = text.split('\n')
        line_heights = [draw.textbbox((0, 0), line, font=font)[3] for line in lines]
        total_height = sum(line_heights)
        
        # Draw each line centered
        y = (height - total_height) // 2
        for i, line in enumerate(lines):
            text_width = draw.textbbox((0, 0), line, font=font)[2]
            position = ((width - text_width) // 2, y)
            draw.text(position, line, font=font, fill=text_color)
            y += line_heights[i]
        
        return np.array(img)
    
    @staticmethod
    def add_timestamp(image, timepoint):
        """
        Add a timestamp to an image.
        
        Parameters:
        -----------
        image : ndarray
            Image to add timestamp to
        timepoint : int
            Timepoint to display
            
        Returns:
        --------
        ndarray
            Image with timestamp
        """
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(np.uint8(image))
        else:
            pil_image = image
            
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Try to get a font
        try:
            # For Windows
            if os.name == 'nt':
                font = ImageFont.truetype("arial.ttf", 48)
            else:
                # For Unix/Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        except IOError:
            font = ImageFont.load_default()
            
        # Add timestamp text at the top center
        text = f"Timepoint: {timepoint}"
        text_width = draw.textlength(text, font=font)  # For PIL >= 9.2.0
        
        # Position text at top center
        position = ((pil_image.width - text_width) // 2, 10)
        
        # Draw text with white outline for visibility
        for offset in [(1,1), (-1,-1), (1,-1), (-1,1)]:
            draw.text((position[0]+offset[0], position[1]+offset[1]), text, font=font, fill="white")
        draw.text(position, text, font=font, fill="white")
        
        # Convert back to numpy array if needed
        if isinstance(image, np.ndarray):
            return np.array(pil_image)
        return pil_image