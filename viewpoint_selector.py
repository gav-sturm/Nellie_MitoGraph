"""
ViewpointSelector Module

This module provides the ViewpointSelector class for selecting and managing 3D viewpoints.
"""

import os
import time
import numpy as np
import napari
import json

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
        self.screenshot_img = None
    
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

    def determine_2d_transformations(self):
        """
        Determine the sequence of 2D transformations needed to match the current 3D viewpoint.
        Uses a lookup table approach to directly determine transformations.
        """
        if not self.has_view:
            print("No view has been captured. Cannot determine transformations.")
            return []
        
        # Extract the camera angles from the captured view
        angles = self.captured_view[0]
        azimuth, elevation, roll = angles
        
        # print(f"Raw angles - azimuth: {azimuth}, elevation: {elevation}, roll: {roll}")
        
        # Normalize angles to the range [0, 360)
        azimuth = azimuth % 360
        elevation = elevation % 360
        roll = roll % 360
        
        # print(f"Normalized angles - azimuth: {azimuth}, elevation: {elevation}, roll: {roll}")
        
        # Create a comprehensive lookup table for viewpoints
        # Format: (azimuth_range, elevation_range) -> transformations
        # Each range is (min, max) inclusive
        
        lookup_table = {
            # Front views (azimuth near 0)
            ((0, 45), (0, 45)): [('flip_v', None)],
            ((0, 45), (315, 360)): [('flip_v', None)],  # Near-horizontal view
            
            # Right side views (azimuth near 90)
            ((45, 135), (0, 45)): [('flip_v', None), ('rotate', 90)],
            ((45, 135), (315, 360)): [('flip_v', None), ('rotate', 90)],  # Near-horizontal view
            
            # Back views (azimuth near 180)
            ((135, 225), (0, 45)): [('flip_v', None), ('flip_h', None)],
            ((135, 225), (315, 360)): [('flip_v', None), ('flip_h', None)],  # Near-horizontal view
            
            # Left side views (azimuth near 270)
            ((225, 315), (0, 45)): [('flip_v', None), ('rotate', -90)],
            ((225, 315), (315, 360)): [('flip_v', None), ('rotate', -90)],  # Near-horizontal view
            
            # Top views (elevation near 90)
            ((0, 45), (45, 90)): [('flip_v', None), ('transpose', None)],
            ((45, 135), (45, 90)): [('flip_v', None), ('transpose', None), ('rotate', 90)],
            ((135, 225), (45, 90)): [('flip_v', None), ('transpose', None), ('flip_h', None)],
            ((225, 315), (45, 90)): [('flip_v', None), ('transpose', None), ('rotate', -90)],
            
            # Bottom views (elevation near 270)
            ((0, 45), (270, 315)): [('transpose', None)],
            ((45, 135), (270, 315)): [('transpose', None), ('rotate', 90)],
            ((135, 225), (270, 315)): [('transpose', None), ('flip_h', None)],
            ((225, 315), (270, 315)): [('transpose', None), ('rotate', -90)],
            
            # Special case for the specific viewpoint you mentioned
            ((175, 195), (340, 350)): [('flip_v', None), ('flip_h', None)],
            
            # Special cases for vertical flip
            ((0, 45), (170, 190)): [],  # Front view, vertically flipped
            ((45, 135), (170, 190)): [('rotate', 90)],  # Right view, vertically flipped
            ((135, 225), (170, 190)): [('flip_h', None)],  # Back view, vertically flipped
            ((225, 315), (170, 190)): [('rotate', -90)],  # Left view, vertically flipped
        }
        
        # Find the matching entry in the lookup table
        transformations = None
        for (azim_range, elev_range), trans in lookup_table.items():
            if azim_range[0] <= azimuth < azim_range[1] and elev_range[0] <= elevation < elev_range[1]:
                transformations = trans.copy()
                # print(f"Matched range: azimuth {azim_range}, elevation {elev_range}")
                break
        
        # If no match found, try to determine if the view is vertically flipped
        if transformations is None:
            # print("No exact match found in lookup table")
            
            # Check if the elevation is around 180 degrees (vertically flipped)
            if 160 <= elevation <= 200:
                # print("Detected vertically flipped view")
                # For vertically flipped views, don't apply the vertical flip
                if 135 <= azimuth < 225:
                    transformations = [('flip_h', None)]  # Back view, vertically flipped
                elif 45 <= azimuth < 135:
                    transformations = [('rotate', 90)]  # Right view, vertically flipped
                elif 225 <= azimuth < 315:
                    transformations = [('rotate', -90)]  # Left view, vertically flipped
                else:
                    transformations = []  # Front view, vertically flipped
            else:
                # For azimuth around 180, we generally need a horizontal flip
                if 135 <= azimuth < 225:
                    transformations = [('flip_v', None), ('flip_h', None)]
                else:
                    transformations = [('flip_v', None)]
        
        print(f"Final transformations for viewpoint (azimuth={azimuth:.1f}°, elevation={elevation:.1f}°, roll={roll:.1f}°):")
        for op, param in transformations:
            if param is not None:
                print(f"- {op} {param}°")
            else:
                print(f"- {op}")
        
        return transformations

    def apply_transformations_to_image(self, image, transformations):
        """
        Apply a sequence of 2D transformations to an image.
        
        Parameters:
        -----------
        image : ndarray
            Input image to transform
        transformations : list
            List of transformation operations from determine_2d_transformations()
            
        Returns:
        --------
        ndarray
            Transformed image
        """
        from skimage.transform import rotate
        
        result = image.copy()
        
        for op, param in transformations:
            if op == 'rotate':
                # Rotate the image by the specified angle
                result = rotate(result, param, resize=True, preserve_range=True)
                result = result.astype(image.dtype)
            elif op == 'flip_h':
                # Flip horizontally
                result = np.fliplr(result)
            elif op == 'flip_v':
                # Flip vertically
                result = np.flipud(result)
            elif op == 'transpose':
                # Transpose (swap x and y axes)
                result = np.transpose(result, axes=(1, 0, 2) if result.ndim == 3 else (1, 0))
        
        return result