"""
ViewpointSelector Module

This module provides classes for selecting and managing 3D viewpoints for volumetric data visualization.
The module is designed to be modular and reusable, with separate classes for different responsibilities.
"""

import sys
import os
import time
import napari
import numpy as np
from tifffile import imread
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from imageio.v3 import imwrite as iio_imwrite  # For GIF creation
from glob import glob
from pathlib import Path
import argparse
import json
from napari.utils.colormaps import DirectLabelColormap
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import networkx as nx
from skimage import measure
from collections import defaultdict as dd

# Import from local modules
from TopologicalGraph import TopologicalGraph

class VolumeLoader:
    """
    Handles loading and preprocessing of different volume types.
    
    This class is responsible for loading skeleton, branch, original, and other volume types,
    and provides methods for accessing and preprocessing these volumes.
    """
    
    def __init__(self, volume_file, intensity_percentile=50):
        """
        Initialize the VolumeLoader with a path to a volume file.
        
        Parameters:
        -----------
        volume_file : str
            Path to the volume file
        intensity_percentile : int, optional
            Percentile cutoff for intensity thresholding (0-100, default: 50)
        """
        self.volume_file = volume_file
        self.intensity_percentile = intensity_percentile
        
        # Load the full volume data
        self.full_volume = imread(volume_file)
        
        # Check if we're working with 4D data
        self.is_timeseries = self.full_volume.ndim == 4
        self.num_timepoints = self.full_volume.shape[0] if self.is_timeseries else 1
        
        # For viewpoint selection, use only the first timepoint
        if self.is_timeseries:
            self.volume = self.full_volume[0].copy()  # First timepoint for UI interactions
        else:
            self.volume = self.full_volume  # Use the whole volume if it's 3D
    
    def load_volume_for_timepoint(self, timepoint=0):
        """
        Load the volume data for a specific timepoint.
        
        Parameters:
        -----------
        timepoint : int, optional
            Timepoint to load (default: 0)
            
        Returns:
        --------
        ndarray
            Volume data for the specified timepoint
        """
        if self.is_timeseries:
            if timepoint < 0 or timepoint >= self.num_timepoints:
                raise ValueError(f"Timepoint {timepoint} is out of range (0-{self.num_timepoints-1})")
            return self.full_volume[timepoint].copy()
        return self.volume.copy()
    
    def preprocess_volume(self, volume, apply_threshold=True):
        """
        Preprocess a volume by applying thresholding.
        
        Parameters:
        -----------
        volume : ndarray
            Volume data to preprocess
        apply_threshold : bool, optional
            Whether to apply intensity thresholding (default: True)
            
        Returns:
        --------
        ndarray
            Preprocessed volume
        """
        processed_volume = volume.copy()
        
        if apply_threshold:
            # Calculate threshold value based on user-specified percentile of non-zero values
            non_zero_values = processed_volume[processed_volume > 0]
            if len(non_zero_values) > 0:
                threshold = np.percentile(non_zero_values, self.intensity_percentile)
                # Set values below threshold to zero
                processed_volume[processed_volume < threshold] = 0
        
        return processed_volume
    
    @staticmethod
    def load_multiple_volumes(files_dict):
        """
        Load multiple volumes from a dictionary of file paths.
        
        Parameters:
        -----------
        files_dict : dict
            Dictionary mapping volume names to file paths
            
        Returns:
        --------
        dict
            Dictionary mapping volume names to VolumeLoader instances
        """
        volumes = {}
        for name, path in files_dict.items():
            if path and os.path.exists(path):
                volumes[name] = VolumeLoader(path)
        return volumes


class ViewpointSelector:
    """
    Handles selection of viewpoints for 3D visualization.
    
    This class is responsible for launching a 3D viewer and capturing the desired viewpoint
    parameters for later visualization.
    """
    
    def __init__(self):
        """Initialize the ViewpointSelector."""
        self.captured_view = None
        self.zoom_level = None
        self.viewport_size = None
    
    def select_viewpoint(self, volume):
        """
        Launch a 3D viewer of the volume.
        Adjust the view and press 'v' to capture the view.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume data to visualize
            
        Returns:
        --------
        tuple or None
            Captured view parameters (angles, center) or None if no view was captured
        """
        captured_view = {}
    
        viewer = napari.Viewer(ndisplay=3)
        layer = viewer.add_image(volume, name="Volume")
    
        def capture_view(event):
            print("Capturing view (screenshot)...")
            time.sleep(0.5)
            # Store zoom level and viewport size for consistent scaling
            self.zoom_level = viewer.camera.zoom
            self.viewport_size = viewer.window.qt_viewer.canvas.size
            
            # Capture screenshot but don't save to disk
            img = viewer.screenshot(canvas_only=True)
            # Store the screenshot for ROI selection
            self.screenshot_img = img
            
            # Store camera parameters
            captured_view['view'] = (viewer.camera.angles, viewer.camera.center)
            viewer.close()
    
        viewer.bind_key('v', capture_view)
        print("Adjust the 3D view. Press 'v' to capture the view and close the viewer automatically.")
        napari.run()
    
        self.captured_view = captured_view.get('view', None)
        if self.captured_view is None:
            print("No view was captured.")
        else:
            print("Captured view:", self.captured_view)
        return self.captured_view
    
    @property
    def has_view(self):
        """Check if a view has been captured."""
        return self.captured_view is not None
    
    def save_viewpoint_config(self, config_file, roi_polygon=None):
        """
        Save viewpoint and ROI parameters to a JSON file.
        
        Parameters:
        -----------
        config_file : str
            Path to save the configuration file
        roi_polygon : ndarray, optional
            ROI polygon coordinates (default: None)
        """
        config = {
            'viewpoint_angles': self.captured_view[0].tolist() if self.captured_view else None,
            'viewpoint_center': self.captured_view[1].tolist() if self.captured_view else None,
            'zoom_level': self.zoom_level,
            'viewport_size': list(self.viewport_size) if self.viewport_size else None,
            'roi_polygon': roi_polygon.tolist() if roi_polygon is not None else None
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Viewpoint configuration saved to {config_file}")
    
    def load_viewpoint_config(self, config_file):
        """
        Load viewpoint and ROI parameters from a JSON file.
        
        Parameters:
        -----------
        config_file : str
            Path to the configuration file
            
        Returns:
        --------
        dict
            Dictionary containing the loaded configuration
        """
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if config['viewpoint_angles'] and config['viewpoint_center']:
            self.captured_view = (
                np.array(config['viewpoint_angles']),
                np.array(config['viewpoint_center'])
            )
        self.zoom_level = config['zoom_level']
        self.viewport_size = tuple(config['viewport_size']) if config['viewport_size'] else None
        
        roi_polygon = None
        if config['roi_polygon']:
            roi_polygon = np.array(config['roi_polygon'])
            
        print("Loaded viewpoint configuration:")
        print(f"- Angles: {self.captured_view[0] if self.captured_view else None}")
        print(f"- Center: {self.captured_view[1] if self.captured_view else None}")
        print(f"- Zoom: {self.zoom_level}")
        print(f"- Viewport size: {self.viewport_size}")
        print(f"- ROI polygon: {roi_polygon.shape if roi_polygon is not None else None} points")
        
        return {
            'captured_view': self.captured_view,
            'zoom_level': self.zoom_level,
            'viewport_size': self.viewport_size,
            'roi_polygon': roi_polygon
        }


class ROISelector:
    """
    Handles selection and processing of Regions of Interest (ROIs).
    
    This class is responsible for allowing the user to draw an ROI on a 2D image,
    and for applying this ROI to images for cropping and filtering.
    """
    
    def __init__(self):
        """Initialize the ROISelector."""
        self.roi_polygon = None
    
    def select_roi(self, image):
        """
        Display an image in a 2D viewer and let the user draw an ROI polygon.
        
        Parameters:
        -----------
        image : ndarray
            Image to display for ROI selection
            
        Returns:
        --------
        ndarray or None
            ROI polygon coordinates, or None if no ROI was drawn
        """
        if image is None:
            print("No image available for ROI selection.")
            return None
        
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(image, name="Image")
        shapes_layer = viewer.add_shapes(name="ROI", shape_type="polygon")
        
        print("Draw an ROI on the image.")
        print("Double-click to complete the polygon - viewer will close automatically.")
        
        # Create a callback function to detect when a polygon is completed
        def on_data_change(event):
            # Check if we have at least one complete polygon
            if len(shapes_layer.data) > 0 and len(shapes_layer.data[0]) >= 3:
                # Check if the last point is close to the first point (polygon completed)
                # This happens when user double-clicks to finish the polygon
                if shapes_layer.mode == 'pan_zoom':  # Mode changes to pan_zoom after completion
                    print("ROI polygon completed. Closing viewer...")
                    # Store the polygon before closing
                    self.roi_polygon = shapes_layer.data[0]
                    # Reduced delay from 0.5 to 0.2 seconds
                    import threading
                    threading.Timer(0.2, viewer.close).start()
        
        # Connect the callback to the data change event
        shapes_layer.events.data.connect(on_data_change)
        shapes_layer.events.mode.connect(on_data_change)
        
        # Run the viewer
        napari.run()
        
        # In case the user manually closed the viewer without completing a polygon
        if not self.roi_polygon and len(shapes_layer.data) > 0:
            self.roi_polygon = shapes_layer.data[0]
            print("ROI captured.")
        elif not self.roi_polygon:
            print("No ROI drawn; using full image as ROI.")
        
        return self.roi_polygon
    
    def apply_roi_to_image(self, img, roi_polygon=None, scale_factor=(1.0, 1.0)):
        """
        Apply an ROI polygon to an image and return the filtered image.
        
        Parameters:
        -----------
        img : ndarray
            Image to filter
        roi_polygon : ndarray, optional
            ROI polygon coordinates (default: self.roi_polygon)
        scale_factor : tuple, optional
            Scale factors (y_scale, x_scale) for scaling the ROI (default: (1.0, 1.0))
            
        Returns:
        --------
        tuple
            (filtered_img, roi_bbox) where roi_bbox is (min_row, max_row, min_col, max_col)
        """
        if roi_polygon is None:
            roi_polygon = self.roi_polygon
        
        if roi_polygon is None:
            # No ROI defined, return the original image with a full-image bounding box
            h, w = img.shape[:2]
            bbox = (0, h-1, 0, w-1)
            return img, bbox
        
        # Scale the ROI polygon if needed
        y_scale, x_scale = scale_factor
        if y_scale != 1.0 or x_scale != 1.0:
            scaled_polygon = roi_polygon.copy()
            scaled_polygon[:, 0] = scaled_polygon[:, 0] * y_scale
            scaled_polygon[:, 1] = scaled_polygon[:, 1] * x_scale
        else:
            scaled_polygon = roi_polygon
        
        # Convert the polygon to a mask
        rr, cc = sk_polygon(scaled_polygon[:, 0], scaled_polygon[:, 1], shape=img.shape[:2])
        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[rr, cc] = True
        
        # Get the bounding box of the ROI
        rows, cols = np.nonzero(mask)
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            bbox = (min_row, max_row, min_col, max_col)
        else:
            h, w = img.shape[:2]
            bbox = (0, h-1, 0, w-1)
        
        # Apply mask to image
        filtered = img.copy()
        
        # Handle RGB images differently from grayscale
        if img.ndim == 3 and img.shape[2] in [3, 4]:  # RGB or RGBA
            # For RGB/RGBA images, we need to apply the mask to each channel
            for c in range(img.shape[2]):
                channel = filtered[:, :, c]
                # Set pixels outside the mask to black (0)
                channel[~mask] = 0
        else:
            # For grayscale images, simply apply the mask
            filtered[~mask] = 0
        
        return filtered, bbox
    
    def crop_to_bbox(self, img, bbox):
        """
        Crop an image to a bounding box.
        
        Parameters:
        -----------
        img : ndarray
            Image to crop
        bbox : tuple
            Bounding box (min_row, max_row, min_col, max_col)
            
        Returns:
        --------
        ndarray
            Cropped image
        """
        min_row, max_row, min_col, max_col = bbox
        return img[min_row:max_row+1, min_col:max_col+1]


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
        viewer = napari.Viewer(ndisplay=3)
        
        # Get coordinates of volume voxels
        volume_mask = (volume > 0)
        z_coords, y_coords, x_coords = np.where(volume_mask)
        
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
        
        time.sleep(0.2)
        img_depth = viewer.screenshot(canvas_only=True)
        
        viewer.close()
        
        # Check if the image is empty (all black)
        if np.mean(img_depth) < 0.01:  # Very dark image
            print("Warning: Depth image appears to be empty. Using alternative approach.")
            return self._create_simple_depth_image(volume)
            
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
        
        # Project the volume to 2D, preserving Z information
        depth_img = np.zeros((Y, X, 4), dtype=np.float32)
        
        # For each Z-slice, color the voxels
        for z in range(Z):
            # Get the normalized z-value for coloring
            z_norm = z / max(1, Z-1)
            # Get color from plasma colormap
            color = plt.cm.plasma(z_norm)
            
            # Get mask for this slice
            mask = volume[z] > 0
            
            # Apply color to this slice's voxels
            for c in range(4):
                depth_img[:, :, c] = np.where(mask, color[c], depth_img[:, :, c])
        
        return depth_img
    
    @staticmethod
    def create_placeholder_image(text, width=800, height=400, bg_color=(30, 30, 30), text_color=(200, 200, 200)):
        """
        Create a placeholder image with text.
        
        Parameters:
        -----------
        text : str
            Text to display on the placeholder
        width : int, optional
            Width of the placeholder image (default: 800)
        height : int, optional
            Height of the placeholder image (default: 400)
        bg_color : tuple, optional
            Background color (R, G, B) (default: (30, 30, 30))
        text_color : tuple, optional
            Text color (R, G, B) (default: (200, 200, 200))
            
        Returns:
        --------
        ndarray
            Placeholder image
        """
        from PIL import Image, ImageDraw, ImageFont
        
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
        from PIL import Image, ImageDraw, ImageFont
        
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


class TimepointManager:
    """
    Manages selection and parsing of timepoints for timeseries data.
    
    This class is responsible for allowing the user to select timepoints from a timeseries,
    and for parsing timepoint strings into lists of timepoint indices.
    """
    
    def __init__(self, num_timepoints):
        """
        Initialize the TimepointManager.
        
        Parameters:
        -----------
        num_timepoints : int
            Total number of timepoints in the timeseries
        """
        self.num_timepoints = num_timepoints
        self.selected_timepoints = None
    
    def select_timepoints(self):
        """
        Let the user select which timepoints to process from the timeseries.
        
        Returns:
        --------
        list
            List of selected timepoint indices
        """
        if self.num_timepoints <= 1:
            print("Not a timeseries or only one timepoint available.")
            return [0]
        
        print(f"\nTimeseries contains {self.num_timepoints} timepoints (0-{self.num_timepoints-1}).")
        print("Enter timepoints to process in one of these formats:")
        print("  - Individual timepoints separated by commas: 0,5,10,15")
        print("  - Range of timepoints: 0-20")
        print("  - Combination: 0,5,10-20,25,30-40")
        print("  - 'all' to process all timepoints")
        
        while True:
            selection = input("Enter timepoints to process: ").strip().lower()
            
            if selection == 'all':
                return list(range(self.num_timepoints))
            
            try:
                # Parse the input string to extract timepoints
                selected_timepoints = []
                for part in selection.split(','):
                    if '-' in part:
                        # Handle range
                        start, end = map(int, part.split('-'))
                        if start < 0 or end >= self.num_timepoints:
                            raise ValueError(f"Range {start}-{end} outside valid timepoints (0-{self.num_timepoints-1})")
                        selected_timepoints.extend(range(start, end + 1))
                    else:
                        # Handle individual timepoint
                        t = int(part)
                        if t < 0 or t >= self.num_timepoints:
                            raise ValueError(f"Timepoint {t} outside valid range (0-{self.num_timepoints-1})")
                        selected_timepoints.append(t)
                
                # Remove duplicates and sort
                selected_timepoints = sorted(set(selected_timepoints))
                
                if not selected_timepoints:
                    print("No valid timepoints selected. Please try again.")
                    continue
                
                print(f"Selected {len(selected_timepoints)} timepoints: {selected_timepoints}")
                self.selected_timepoints = selected_timepoints
                return selected_timepoints
                
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
    
    def parse_timepoints_string(self, timepoints_str):
        """
        Parse a timepoints string from command line args.
        
        Parameters:
        -----------
        timepoints_str : str
            Timepoints string (e.g. "0,5,10-20,30" or "all")
            
        Returns:
        --------
        list
            List of timepoint indices
        """
        if timepoints_str.lower() == 'all':
            return list(range(self.num_timepoints))
        
        selected_timepoints = []
        try:
            for part in timepoints_str.split(','):
                if '-' in part:
                    #