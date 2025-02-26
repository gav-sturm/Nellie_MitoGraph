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