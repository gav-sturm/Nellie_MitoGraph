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
from TopologicalGraph import TopologicalGraph

class NodeTracker:
    def __init__(self, initial_pos, node_id, frame_idx):
        self.id = node_id
        self.kalman_filter = KalmanFilter(dim_x=4, dim_z=3)
        # Initialize state transition matrix
        self.kalman_filter.F = np.array([[1, 0, 0, 1],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 1],
                                        [0, 0, 0, 1]])
        # Initialize measurement matrix
        self.kalman_filter.H = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0]])
        # Initial state
        self.kalman_filter.x = np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0])
        self.last_seen = frame_idx
        self.history = [initial_pos]

class ViewpointSelector:
    def __init__(self, volume_file, output_file='final_view.png', projection_axis=0, intensity_percentile=50):
        """
        Initialize with the path to a volume file and an output filename.
        The volume is assumed to be a 4D OME-TIF with shape (T, Z, Y, X).
        This script uses only the first time point, resulting in a 3D volume (Z, Y, X).
        You can specify the projection_axis along which the max projection is computed.
        For a volume stored as (Z, Y, X), use projection_axis=0 to obtain an (Y, X) image.
        
        Parameters:
        -----------
        volume_file : str
            Path to the volume file
        output_file : str, optional
            Path to save the output file (default: 'final_view.png')
        projection_axis : int, optional
            Axis along which to project (default: 0)
        intensity_percentile : int, optional
            Percentile cutoff for intensity thresholding (0-100, default: 50)
            Higher values show less background, lower values show more detail
        """
        self.volume_file = volume_file
        self.output_file = output_file
        self.projection_axis = projection_axis
        self.intensity_percentile = intensity_percentile
        
        # Load the full volume data
        self.full_volume = imread(volume_file)
        
        # Check if we're working with 4D data
        self.is_timeseries = self.full_volume.ndim == 4
        self.num_timepoints = self.full_volume.shape[0] if self.is_timeseries else 1
        print(f"Detected {'4D timeseries' if self.is_timeseries else '3D volume'} with {self.num_timepoints} timepoint(s)")
        print(f"Using intensity percentile cutoff: {self.intensity_percentile}%")
        
        # For viewpoint selection, use only the first timepoint
        if self.is_timeseries:
            self.volume = self.full_volume[0].copy()  # First timepoint for UI interactions
        else:
            self.volume = self.full_volume  # Use the whole volume if it's 3D
            
        self.captured_view = None
        self.zoom_level = None
        self.viewport_size = None  # Store the viewport size for scaling
        self.roi_polygon = None
        self.node_registry = {}  # {node_id: {frame: position}}
        self.current_frame = 0
        self.max_node_id = 0
        self.prev_frame_nodes = None
        self.spacing = (1, 1, 1)  # Replace with your (z,y,x) scaling factors
        self.trackers = []
        self.next_id = 1
        self.max_unseen_frames = 3  # Keep track of disappeared nodes for 3 frames

    def select_viewpoint(self, volume_to_view):
        """
        Launch a 3D viewer of the skeleton volume.
        Adjust the view and press 'v' to capture the view.
        The screenshot is stored in self.skeleton_img and camera parameters in self.captured_view.
        """
        captured_view = {}
    
        viewer = napari.Viewer(ndisplay=3)
        layer = viewer.add_image(self.volume, name="Skeleton")
    
        def capture_view(event):
            print("Capturing skeleton view (screenshot)...")
            time.sleep(0.5)
            # Store zoom level and viewport size for consistent scaling
            self.zoom_level = viewer.camera.zoom
            self.viewport_size = viewer.window.qt_viewer.canvas.size
            
            # Capture screenshot but don't save to disk
            img = viewer.screenshot(canvas_only=True)
            # Store the screenshot in self.skeleton_img for ROI selection
            self.skeleton_img = img
            
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

    def crop_final_image(self):
        """
        Load the captured final image (self.output_file) and allow the user to draw an ROI
        in a 2D napari viewer. The drawn ROI is then used to crop the final image.
        The cropped image is saved as "<output_file>_cropped.png".
        """
        final_img = imageio.imread(self.output_file)
    
        # Create a 2D viewer for ROI selection.
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(final_img, name="Final Captured Image")
        shapes_layer = viewer.add_shapes(name="ROI", shape_type="polygon")
        print("Draw an ROI on the final captured image. When finished, close the viewer window to proceed.")
        napari.run()
    
        if len(shapes_layer.data) > 0:
            roi_polygon = shapes_layer.data[0]
            print("ROI captured on final image.")
        else:
            print("No ROI drawn on final image.")
            roi_polygon = None
    
        if roi_polygon is None:
            print("No ROI selected on final image. Using full final image.")
            cropped_img = final_img  # no cropping
        else:
            # Convert the drawn polygon (assumed in (y, x) order) to an ROI mask.
            poly = np.array(roi_polygon)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=final_img.shape[:2])
            mask = np.zeros(final_img.shape[:2], dtype=bool)
            mask[rr, cc] = True
            rows, cols = np.nonzero(mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            cropped_img = final_img[min_row:max_row + 1, min_col:max_col + 1]
            # (Optional) Display the cropped final image for confirmation.
            plt.figure()
            plt.imshow(cropped_img)
            plt.title("Cropped Final View")
            plt.axis('off')
            plt.show()
    
        # Append a legend (colorbar) to the cropped final image.
        cropped_output_file = os.path.splitext(self.output_file)[0] + "_cropped.png"
        # Use the depth range from the original 3D volume.
        Z = self.volume.shape[0]
        cmap = plt.get_cmap("plasma")
        norm_obj = plt.Normalize(vmin=0, vmax=Z-1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([])
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(cropped_img)
        ax.axis('off')
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Depth (Z)")
        fig.savefig(cropped_output_file, bbox_inches='tight')
        plt.close(fig)
        print(f"Cropped final view with legend saved to {cropped_output_file}")
        return cropped_output_file

    def composite_view(self, branch_file, original_file):
        # (This method is removed in favor of the new capture_modalities method.)
        pass

    def run_workflow(self, branch_file, original_file, timepoints=None, node_edge_file=None, obj_label_file=None):
        """
        Run the workflow:
          1. Select viewpoint on the skeleton volume.
          2. Let the user draw an ROI on the captured skeleton screenshot.
          3. If timeseries, let the user select timepoints to process.
          4. Capture composite screenshots of all modalities.
        Returns the path to the saved GIF file or composite image.
        
        Parameters:
        -----------
        branch_file : str
            Path to the branch volume file
        original_file : str
            Path to the original volume file
        timepoints : str, optional
            Timepoints to process (e.g., "0,5,10-20,30"). If not provided, will prompt interactively.
        node_edge_file : str, optional
            Path to the node-edge labeled skeleton file
        obj_label_file : str, optional
            Path to the object label file for coloring the node-edge skeleton
        """
        # First, select the viewpoint from the skeleton:
        self.select_viewpoint(self.volume)
        # Then, let the user draw an ROI on the captured skeleton screenshot:
        self.select_roi()
        
        # If timepoints were provided via command line, parse them
        if timepoints:
            self.selected_timepoints = self.parse_timepoints_string(timepoints)
        # Otherwise use interactive selection if it's a timeseries
        elif self.is_timeseries and self.num_timepoints > 1:
            self.selected_timepoints = self.select_timepoints()
        
        # Capture composite modalities:
        output_file = self.capture_modalities_with_preset_view(branch_file, original_file, 
                                                              node_edge_file=node_edge_file, 
                                                              obj_label_file=obj_label_file)
        
        return output_file

    def select_roi(self):
        """
        Display the captured skeleton screenshot in a 2D viewer.
        Let the user draw an ROI polygon. When the polygon is completed, the viewer automatically closes.
        The ROI is saved in self.roi_polygon.
        """
        if not hasattr(self, 'skeleton_img'):
            print("No skeleton screenshot available for ROI selection.")
            return
        
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(self.skeleton_img, name="Skeleton Screenshot")
        shapes_layer = viewer.add_shapes(name="ROI", shape_type="polygon")
        
        print("Draw an ROI on the skeleton screenshot.")
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
        if not hasattr(self, 'roi_polygon') or self.roi_polygon is None:
            if len(shapes_layer.data) > 0:
                self.roi_polygon = shapes_layer.data[0]
                print("ROI captured.")
            else:
                print("No ROI drawn; using full image as ROI.")
                self.roi_polygon = None

    def select_timepoints(self):
        """
        Let the user select which timepoints to process from the timeseries.
        Returns a list of selected timepoint indices.
        """
        if not self.is_timeseries or self.num_timepoints <= 1:
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
                return selected_timepoints
                
            except ValueError as e:
                print(f"Error: {e}. Please try again.")

    def capture_screenshot(self, volume, layer_type='image'):
        """
        Open a 3D viewer for the given volume, set the camera to the captured view,
        and capture a screenshot with improved resolution.
        """
        viewer = napari.Viewer(ndisplay=3)
        
        # Use higher quality rendering for original images
        if layer_type == 'labels':
            viewer.add_labels(volume, name="Modality")
        else:
            # For original/raw images, use better rendering settings
            if volume is not self.volume:  # If it's the original image, not skeleton
                # Remove bottom pixels from raw image to improve rendering efficiency
                processed_volume = volume.copy()
                
                # Calculate threshold value based on user-specified percentile of non-zero values
                non_zero_values = processed_volume[processed_volume > 0]
                if len(non_zero_values) > 0:
                    threshold = np.percentile(non_zero_values, self.intensity_percentile)
                    # Set values below threshold to zero
                    processed_volume[processed_volume < threshold] = 0
                    # print(f"Applied {self.intensity_percentile}% intensity threshold (values below {threshold:.2f} removed)")
                
                # Higher quality rendering for original data
                viewer.add_image(processed_volume, name="Modality", rendering='mip', 
                                contrast_limits=[0, processed_volume.max()])
            else:
                viewer.add_image(volume, name="Modality")

        if self.captured_view is not None:
            angles, center = self.captured_view
            viewer.camera.angles = angles
            viewer.camera.center = center
            
            # Apply the exact same zoom level that was captured
            if self.zoom_level is not None:
                viewer.camera.zoom = self.zoom_level
        else:
            print("No captured viewpoint; using default camera.")
        
        # Use longer delay for original images to ensure proper rendering
        if volume is not self.volume and layer_type == 'image':
            time.sleep(0.5)  # Longer delay for original images
        else:
            time.sleep(0.2)  # Standard delay for other images
        
        # Use higher scale factor for better resolution
        img = viewer.screenshot(canvas_only=True, scale=1.5)
        
        viewer.close()
        return img

    def capture_modalities(self, branch_file, original_file, obj_label_file):
        """
        Capture screenshots for skeleton, branch, and original volumes using the captured camera view.
        Apply the same ROI (if drawn) to all screenshots and combine them side-by-side.
        Save and return the composite image filename.
        """
        # Load volumes and capture screenshots for all four modalities
        img_skel, img_branch, img_obj_label, img_original, img_depth = self._capture_all_screenshots(branch_file, original_file, obj_label_file)
        
        # Apply ROI filtering and cropping
        cropped_images, _ = self._apply_roi_to_images(img_skel, img_branch, img_obj_label, img_original, img_depth)
        cropped_skel, cropped_branch, cropped_obj_label, cropped_original, cropped_depth = cropped_images
        
        # Create and save composite image
        composite_file = self._create_composite_image([cropped_original, cropped_branch, cropped_obj_label, cropped_skel, cropped_depth])
        
        return composite_file

    def _capture_all_screenshots(self, branch_file, original_file, objects_file):
        """
        Load volumes and capture screenshots for all four modalities.
        
        Parameters:
        -----------
        branch_file : str
            Path to the branch volume file
        original_file : str
            Path to the original volume file
            
        Returns: tuple of (skeleton_img, branch_img, original_img, depth_img)
        """
        # Capture screenshot for skeleton (using raw image display)
        img_skel = self.capture_screenshot(self.volume, layer_type='image')

        # Load branch and original volumes (for first timepoint)
        branch = imread(branch_file)
        if branch.ndim == 4:
            branch = branch[0]

        # Load and process object labels same as branch labels
        objects = imread(objects_file)
        if objects.ndim == 4:
            objects = objects[0]  # Use first timepoint if 4D
        if objects.dtype != np.uint32:
            objects = objects.astype(np.uint32)

        original = imread(original_file)
        if original.ndim == 4:
            original = original[0]

        # Capture branch using labels so that its pre-assigned colors are preserved
        img_branch = self.capture_screenshot(branch, layer_type='labels')

        # Capture object label image if available
        img_obj_label = self.capture_screenshot(objects, layer_type='labels')

        # Capture raw (original) image
        img_original = self.capture_screenshot(original, layer_type='image')
        
        # Generate and capture depth-encoded image
        img_depth = self._generate_depth_encoded_image()
        
        return img_skel, img_branch, img_obj_label,img_original, img_depth

    def _apply_roi_to_images(self, img_skel, img_branch, img_obj_label, img_original, img_depth, img_node_edge=None):
        """
        Apply the ROI polygon to all images and crop them to the ROI bounding box.
        """
        # Create a mask from the ROI polygon
        mask = None
        roi_bbox = None
        
        if self.roi_polygon is not None:
            # Get the dimensions of the skeleton image
            original_height, original_width = self.skeleton_img.shape[:2]
            current_height, current_width = img_skel.shape[:2]
            
            # Scale the ROI polygon if the current image dimensions differ from the original
            scaled_polygon = self.roi_polygon.copy()
            
            if (original_height != current_height or original_width != current_width):
                # Calculate scaling factors
                scale_y = current_height / original_height
                scale_x = current_width / original_width
                
                # Apply scaling to the polygon
                scaled_polygon[:, 0] = scaled_polygon[:, 0] * scale_y
                scaled_polygon[:, 1] = scaled_polygon[:, 1] * scale_x
            
            # Convert the polygon to a mask
            rr, cc = sk_polygon(scaled_polygon[:, 0], scaled_polygon[:, 1], shape=img_skel.shape[:2])
            mask = np.zeros(img_skel.shape[:2], dtype=bool)
            mask[rr, cc] = True
            
            # Get the bounding box of the ROI
            rows, cols = np.nonzero(mask)
            if len(rows) > 0 and len(cols) > 0:
                min_row, max_row = rows.min(), rows.max()
                min_col, max_col = cols.min(), cols.max()
                roi_bbox = (min_row, max_row, min_col, max_col)
        
        # Ensure all images have the same dimensions as img_skel before applying mask
        if img_branch.shape[:2] != img_skel.shape[:2]:
            img_branch = resize(img_branch, img_skel.shape[:2], 
                              preserve_range=True, anti_aliasing=True).astype(img_branch.dtype)
        
        if img_original.shape[:2] != img_skel.shape[:2]:
            img_original = resize(img_original, img_skel.shape[:2], 
                                preserve_range=True, anti_aliasing=True).astype(img_original.dtype)
        
        if img_obj_label.shape[:2] != img_skel.shape[:2]:
            img_obj_label = resize(img_obj_label, img_skel.shape[:2], 
                                preserve_range=True, anti_aliasing=True).astype(img_obj_label.dtype)
        
        if img_depth.shape[:2] != img_skel.shape[:2]:
            img_depth = resize(img_depth, img_skel.shape[:2], 
                             preserve_range=True, anti_aliasing=True).astype(img_depth.dtype)
        
        # Apply mask to each image
        if mask is not None:
            img_skel = self._apply_mask_to_image(img_skel, mask)
            img_branch = self._apply_mask_to_image(img_branch, mask)
            img_obj_label = self._apply_mask_to_image(img_obj_label, mask)
            img_original = self._apply_mask_to_image(img_original, mask)
            img_depth = self._apply_mask_to_image(img_depth, mask)
        
        # If node_edge image is provided, handle it separately
        if img_node_edge is not None:
            # Resize node_edge to match mask dimensions before applying mask
            if img_node_edge.shape[:2] != img_skel.shape[:2]:
                img_node_edge = resize(img_node_edge, img_skel.shape[:2], 
                                     preserve_range=True, anti_aliasing=True).astype(img_node_edge.dtype)
            
            if mask is not None:
                img_node_edge = self._apply_mask_to_image(img_node_edge, mask)
        
        # Crop to ROI bounding box
        if roi_bbox is not None:
            min_row, max_row, min_col, max_col = roi_bbox
            img_skel = img_skel[min_row:max_row+1, min_col:max_col+1]
            img_branch = img_branch[min_row:max_row+1, min_col:max_col+1]
            img_obj_label = img_obj_label[min_row:max_row+1, min_col:max_col+1]
            img_original = img_original[min_row:max_row+1, min_col:max_col+1]
            img_depth = img_depth[min_row:max_row+1, min_col:max_col+1]
            
            if img_node_edge is not None:
                img_node_edge = img_node_edge[min_row:max_row+1, min_col:max_col+1]
        
        return (img_skel, img_branch, img_obj_label, img_original, img_depth, img_node_edge), mask

    def _ensure_shape(self, img, target_shape):
        """Resize image to match target shape if needed."""
        if img.shape != target_shape:
            return resize(img, target_shape, preserve_range=True, 
                         anti_aliasing=True).astype(img.dtype)
        return img

    def _apply_mask_to_images(self, images, mask):
        """
        Apply a binary mask to a list of images.
        
        Parameters:
        -----------
        images : list or ndarray
            List of images to mask, or a single image
        mask : ndarray
            Binary mask
        
        Returns:
        --------
        list or ndarray
            List of masked images, or a single masked image
        """
        # Handle both lists of images and single images
        if isinstance(images, list):
            return [self._apply_mask_to_image(img, mask) for img in images]
        else:
            return self._apply_mask_to_image(images, mask)

    def _apply_mask_to_image(self, img, mask):
        """
        Apply a binary mask to an image.
        
        Parameters:
        -----------
        img : ndarray
            Image to mask
        mask : ndarray
            Binary mask
        
        Returns:
        --------
        ndarray
            Masked image
        """
        # Make a copy to avoid modifying the original
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
        
        return filtered

    def _generate_depth_encoded_image(self):
        """
        Generate a depth-encoded image of the skeleton using points colored by Z-depth.
        Returns: depth-encoded image as a screenshot
        """
        # Create a standard napari viewer - this worked before
        viewer = napari.Viewer(ndisplay=3)
        
        # Get coordinates of skeleton voxels
        volume_mask = (self.volume > 0)
        z_coords, y_coords, x_coords = np.where(volume_mask)
        
        # Use the same colormap approach that worked before
        norm = plt.Normalize(vmin=z_coords.min(), vmax=z_coords.max())
        cmap = plt.cm.plasma
        colors = cmap(norm(z_coords))
        
        # Add points layer with the same parameters as before
        points = np.column_stack((z_coords, y_coords, x_coords))
        points_layer = viewer.add_points(
            points,
            name="Depth-Encoded Skeleton",
            size=4,  # Slightly larger for better visibility
            face_color=colors,
            edge_color="transparent",
            shading="none"  # Disable shading which might affect visibility
        )
        
        # Set camera to captured viewpoint
        if self.captured_view is not None:
            angles, center = self.captured_view
            viewer.camera.angles = angles
            viewer.camera.center = center
            
            # Important: Don't reset view as it changes the perspective
            # Just ensure we're using same scale/zoom as other modalities
            try:
                # Use the same zoom level as in capture_screenshot
                # This is a reasonable default that should match other modalities
                viewer.camera.zoom = self.zoom_level
            except:
                pass  # If zoom setting fails, use default
        
        # Reduced wait time from 0.5 to 0.2 seconds
        time.sleep(0.2)
        img_depth = viewer.screenshot(canvas_only=True)
        
        # Print depth image dimensions to help debug
        # print(f"Depth image dimensions before closing viewer: {img_depth.shape}")
        
        viewer.close()
        
        # Check if the image is empty (all black)
        if np.mean(img_depth) < 0.01:  # Very dark image
            print("Warning: Depth image appears to be empty. Using alternative approach.")
            return self._create_simple_depth_image()
            
        return img_depth

    def _create_simple_depth_image(self):
        """Create a simple depth-encoded image as a fallback."""
        # Create a simple colored image based on the skeleton
        Z, Y, X = self.volume.shape
        
        # Project the skeleton to 2D, preserving Z information
        depth_img = np.zeros((Y, X, 4), dtype=np.float32)
        
        # For each Z-slice, color the skeleton voxels
        for z in range(Z):
            # Get the normalized z-value for coloring
            z_norm = z / max(1, Z-1)
            # Get color from plasma colormap
            color = plt.cm.plasma(z_norm)
            
            # Get mask for this slice
            mask = self.volume[z] > 0
            
            # Apply color to this slice's voxels
            for c in range(4):
                depth_img[:, :, c] = np.where(mask, color[c], depth_img[:, :, c])
        
        return depth_img

    def _create_composite_image(self, images):
        """
        Create a composite image by horizontally stacking the provided images.
        Save it with a black background and white title.
        Returns: path to saved composite image
        """
        # Combine images horizontally
        composite = np.hstack(images)
        
        # Create figure with black background
        composite_file = os.path.splitext(self.output_file)[0] + "_composite.png"
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
        ax.imshow(composite)
        ax.set_title("Composite 3D View", color="white", fontsize=14)
        ax.axis('off')
        
        # Save figure
        fig.savefig(composite_file, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"Composite image saved to {composite_file}")
        
        return composite_file

    def depth_encoded_skeleton(self):
        """
        Compute a depth-encoded 2D image from the skeleton volume.
        For each (y, x), compute the mean z location of skeleton voxels (where self.volume > 0)
        and then take the logarithm (with +1 offset) to amplify depth differences.
        Returns a 2D array of log-scaled depth values.
        """
        Z, Y, X = self.volume.shape
        mask = (self.volume > 0)
        count = np.sum(mask, axis=0)  # shape (Y, X)
        z_indices = np.arange(Z).reshape(Z, 1, 1)
        sum_depth = np.sum(mask * z_indices, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_depth = np.divide(sum_depth, count, out=np.zeros_like(sum_depth, dtype=np.float32), where=(count != 0))
        depth_img = np.log(mean_depth + 1)
        return depth_img

    def capture_depth_encoded(self):
        """
        Generate the depth-encoded image of the skeleton.
        This method computes a 2D depth map from the 3D skeleton volume using a logarithmic scale,
        then maps the result through the 'plasma' colormap.
        If an ROI was selected, the image is cropped accordingly.
        The final image is saved as "<output_file>_depth_encoded.png".
        """
        depth_img = self.depth_encoded_skeleton()  # 2D array, shape (Y, X)

        if self.roi_polygon is not None:
            poly = np.array(self.roi_polygon)  # Assumed (y, x) coordinates
            mask = np.zeros(depth_img.shape, dtype=np.uint8)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=depth_img.shape)
            mask[rr, cc] = 1
            rows, cols = np.nonzero(mask)
            if rows.size > 0 and cols.size > 0:
                min_row = int(rows.min())
                max_row = int(rows.max())
                min_col = int(cols.min())
                max_col = int(cols.max())
                depth_img = depth_img[min_row:max_row + 1, min_col:max_col + 1]

        norm_obj = plt.Normalize(vmin=depth_img.min(), vmax=depth_img.max())
        plasma = plt.get_cmap("plasma")
        colored_depth = plasma(norm_obj(depth_img))

        depth_file = os.path.splitext(self.output_file)[0] + "_depth_encoded.png"
        fig, ax = plt.subplots(figsize=(8,6), facecolor='black')
        ax.imshow(colored_depth)
        ax.set_title("Depth Encoded Skeleton (Log Scale)", color="white", fontsize=14)
        ax.axis('off')
        fig.savefig(depth_file, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        # int(f"Depth encoded image saved to {depth_file}")
        return depth_file

    def capture_modalities_with_preset_view(self, branch_file, original_file, node_edge_file=None, obj_label_file=None):
        """
        Capture screenshots for all volumes using the pre-set camera view and ROI.
        Apply the same ROI to all screenshots and combine them in a panel layout.
        
        If working with 4D timeseries data, processes all timepoints and creates a GIF.
        
        Parameters:
        -----------
        branch_file : str
            Path to the branch volume file
        original_file : str
            Path to the original volume file
        node_edge_file : str, optional
            Path to the node-edge labeled skeleton file
        obj_label_file : str, optional
            Path to the object label file for coloring the node-edge skeleton
            
        Returns:
        --------
        str
            Path to the saved composite image or GIF
        """
        # Store the node-edge and object label files
        self.node_edge_file = node_edge_file
        self.obj_label_file = obj_label_file
        
        if self.is_timeseries:
            return self._process_timeseries(branch_file, original_file)
        else:
            # For single timepoint, use frame_idx=0
            img_node_edge = self._capture_node_edge_screenshot(node_edge_file, 0) if node_edge_file else None
            
            # Original behavior for single timepoint
            img_skel, img_branch, img_obj_label, img_original, img_depth = self._capture_all_screenshots(branch_file, original_file, obj_label_file)
            
            # Create placeholder for topological graph
            img_topo_graph = self._create_placeholder_image("Topological Graph\n(To be implemented)")
            
            # Apply ROI filtering and cropping
            cropped_images, _ = self._apply_roi_to_images(img_skel, img_branch, img_obj_label, img_original, img_depth, img_node_edge)
            cropped_skel, cropped_branch, cropped_obj_label, cropped_original, cropped_depth, cropped_node_edge = cropped_images
            
            # Create output directory for the single frame
            output_dir = os.path.join(os.path.dirname(self.output_file), "timeseries_frames")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create blank images for layout
            blank_image = self._create_placeholder_image("", width=img_original.shape[1], height=img_original.shape[0])
            blank_image1 = blank_image.copy()
            blank_image2 = blank_image.copy()
            
            # Create composite with specified layout
            composite = np.vstack([
                # Top row: Original, Skeleton, Object Labels, Branch Labels
                np.hstack([img_original, img_skel, img_obj_label, img_branch]),
                
                # Bottom row: Depth, Node-Edge, Blank1, Blank2
                np.hstack([img_depth, img_node_edge, blank_image1, blank_image2])
            ])
            
            # Add timestamp to the image
            composite_with_timestamp = self._add_timestamp(composite, 0)

            # make the background black
            # Convert any pure white background pixels to black
            black_background = composite_with_timestamp.copy()
            white_pixels = np.all(black_background == [255, 255, 255], axis=-1)
            black_background[white_pixels] = [0, 0, 0]
            composite_with_timestamp = black_background
            
            # Save frame with higher quality
            frame_file = os.path.join(output_dir, "frame_0000.png")
            imageio.imwrite(frame_file, composite_with_timestamp, quality=100)
            
            return frame_file

    def _process_timeseries(self, branch_file, original_file):
        """
        Process selected timepoints in a 4D dataset and create a GIF animation.
        """
        # Determine which timepoints to process
        if hasattr(self, 'selected_timepoints') and self.selected_timepoints:
            timepoints_to_process = self.selected_timepoints
            # print(f"Processing {len(timepoints_to_process)} selected timepoints...")
        else:
            timepoints_to_process = range(self.num_timepoints)
            print(f"Processing all {self.num_timepoints} timepoints...")
        
        # Load 4D volumes
        branch_data = imread(branch_file)
        original_data = imread(original_file)
        
        # Load node-edge data if available
        node_edge_data = None
        if self.node_edge_file and os.path.exists(self.node_edge_file):
            try:
                node_edge_data = imread(self.node_edge_file)
                # print(f"Loaded node-edge data from {self.node_edge_file}")
            except Exception as e:
                print(f"Error loading node-edge data: {e}")
        
        # Load object label data if available
        obj_label_data = None
        if self.obj_label_file and os.path.exists(self.obj_label_file):
            try:
                obj_label_data = imread(self.obj_label_file)
                # print(f"Loaded object labels from {self.obj_label_file}")
            except Exception as e:
                print(f"Error loading object labels: {e}")
        
        # Verify dimensions
        if branch_data.ndim != 4 or original_data.ndim != 4:
            print("Warning: Expected 4D data. Using standard processing instead.")
            return self.capture_modalities_with_preset_view(branch_file, original_file)
        
        # Create output directory for frames
        output_dir = os.path.join(os.path.dirname(self.output_file), "timeseries_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each selected timepoint
        frame_files = []
        for t_idx, t in enumerate(timepoints_to_process):
            # print(f"Processing timepoint {t} ({t_idx+1}/{len(timepoints_to_process)})...")
            
            # Validate timepoint is in range
            if t < 0 or t >= self.num_timepoints:
                print(f"Warning: Timepoint {t} is out of range (0-{self.num_timepoints-1}). Skipping.")
                continue
            
            # Get data for this timepoint
            skeleton_data_t = self.full_volume[t] if self.is_timeseries else self.volume
            branch_data_t = branch_data[t]
            obj_label_data_t = obj_label_data[t] if obj_label_data is not None else None
            original_data_t = original_data[t]
            
            # Get node-edge data for this timepoint if available
            node_edge_data_t = None
            if node_edge_data is not None:
                if node_edge_data.ndim == 4:
                    node_edge_data_t = node_edge_data[t]
                else:
                    node_edge_data_t = node_edge_data  # Use the same data for all timepoints
            
            # Replace self.volume temporarily for depth encoding
            original_volume = self.volume
            self.volume = skeleton_data_t
            
            # Capture screenshots directly without saving temporary files
            img_skel = self.capture_screenshot(skeleton_data_t, layer_type='image')
            img_branch = self.capture_screenshot(branch_data_t, layer_type='labels')
            img_obj_label = self.capture_screenshot(obj_label_data_t, layer_type='labels')
            img_original = self.capture_screenshot(original_data_t, layer_type='image')
            img_depth = self._generate_depth_encoded_image()

            # Build graph
            TopoGraph = TopologicalGraph(self.captured_view, self.viewport_size, self.zoom_level)
            G,component_edge_pixels = TopoGraph.build_graph_for_frame(node_edge_data_t)
            
            # Assign colors
            component_colors = TopoGraph.assign_component_colors(G)
            
            # Capture node-edge screenshot if available
            if node_edge_data_t is not None:
                img_node_edge = self._capture_node_edge_screenshot(node_edge_data_t, t, component_colors)
            else:
                # Create a blank image with text
                img_node_edge = self._create_placeholder_image("Node-Edge\nData Missing", 
                                                              height=img_skel.shape[0], 
                                                              width=img_skel.shape[1])
            
            # Capture graph visualization
            img_topo_graph = TopoGraph._capture_topological_graph(G, component_colors)
            
            # Restore original volume
            self.volume = original_volume
            
            
            # Apply ROI to all images first
            cropped_images, _ = self._apply_roi_to_images(img_skel, img_branch, img_obj_label, img_original, img_depth, img_node_edge)
            cropped_skel, cropped_branch, cropped_obj_label, cropped_original, cropped_depth, cropped_node_edge = cropped_images

            # Get reference dimensions from the cropped original image
            reference_height = cropped_original.shape[0]
            reference_width = cropped_original.shape[1]

            # Resize function that maintains aspect ratio for all images
            def resize_to_reference(img):
                return resize(img, (reference_height, reference_width), 
                             preserve_range=True, anti_aliasing=True).astype(img.dtype)

            # Modify this list to control the order of images
            resized_images = [
                resize_to_reference(cropped_original),  # Raw image first
                resize_to_reference(cropped_skel),      # Skeleton
                resize_to_reference(cropped_obj_label),  # Object labels
                resize_to_reference(cropped_branch),    # Labels
                resize_to_reference(cropped_depth),     # Depth-encoded
                resize_to_reference(cropped_node_edge) if cropped_node_edge is not None 
                    else self._create_placeholder_image("Node-Edge\nMissing", height=reference_height, width=reference_width),
                resize_to_reference(img_topo_graph),
                # self._create_placeholder_image("Topological\nGraph1", height=reference_height, width=reference_width),
                self._create_placeholder_image("Topological\nGraph2", height=reference_height, width=reference_width)
            ]

            # Convert all to RGBA
            rgba_images = []
            for img in resized_images:
                if img.shape[2] == 3:
                    rgba = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
                    rgba[:,:,:3] = img
                    rgba[:,:,3] = 255
                    rgba_images.append(rgba)
                else:
                    rgba_images.append(img)

            # Split images into two rows
            top_images = rgba_images[:4]    # First 4 images
            bottom_images = rgba_images[4:] # Last 4 images

            try:
                # Create top and bottom rows
                top_row = np.hstack(top_images)
                bottom_row = np.hstack(bottom_images)
                
                # Ensure rows have same width
                if top_row.shape[1] != bottom_row.shape[1]:
                    # Resize bottom row to match top row width
                    bottom_row = resize(bottom_row, 
                                      (bottom_row.shape[0], top_row.shape[1], bottom_row.shape[2]),
                                      preserve_range=True, anti_aliasing=True).astype(np.uint8)
                
                # Combine vertically
                composite = np.vstack([top_row, bottom_row])
                
            except ValueError as e:
                print(f"Error stacking images: {e}")
                print("Image dimensions - Top row:")
                for i, img in enumerate(top_images):
                    print(f"  Image {i+1}: {img.shape}")
                print("Image dimensions - Bottom row:")
                for i, img in enumerate(bottom_images):
                    print(f"  Image {i+4}: {img.shape}")
                composite = self._create_placeholder_image("Layout Error", 
                                                           width=reference_width*3, 
                                                           height=reference_height*2)

            # Add timestamp to the image
            composite_with_timestamp = self._add_timestamp(composite, t)
            
            # Save frame with higher quality
            frame_file = os.path.join(output_dir, f"frame_{t:04d}.png")
            imageio.imwrite(frame_file, composite_with_timestamp, quality=95)
            frame_files.append(frame_file)
        
        if not frame_files:
            print("No frames were generated. Check your timepoint selection.")
            return None
        
        # Interpolate missing frames
        self._interpolate_missing_frames()
        
        # Create GIF from frames
        gif_file = os.path.splitext(self.output_file)[0] + "_timeseries.gif"
        self._create_gif(frame_files, gif_file)
        
        return gif_file

    def _track_nodes(self, current_points, frame_idx):
        """Match nodes between frames with Hungarian algorithm"""
        # Remove stale trackers first
        self.trackers = [t for t in self.trackers 
                        if (frame_idx - t.last_seen) <= self.max_unseen_frames]
        
        if not self.trackers:  # First frame case
            node_ids = list(range(self.next_id, self.next_id + len(current_points)))
            for point, nid in zip(current_points, node_ids):
                tracker = NodeTracker(tuple(point), nid, frame_idx)
                self.trackers.append(tracker)
            self.next_id += len(current_points)
            return node_ids

        # Create cost matrix (trackers vs current points)
        cost_matrix = np.zeros((len(self.trackers), len(current_points)))
        for i, tracker in enumerate(self.trackers):
            predicted_pos = tracker.kalman_filter.x[:3]
            for j, point in enumerate(current_points):
                cost_matrix[i,j] = np.linalg.norm(predicted_pos - point)
        
        # Apply Hungarian algorithm with threshold
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r,c in zip(row_ind, col_ind):
            if cost_matrix[r,c] < 20:  # Max allowed movement between frames
                matches.append((r,c))
        
        # Update matched trackers
        node_ids = [-1]*len(current_points)
        for r,c in matches:
            tracker = self.trackers[r]
            tracker.kalman_filter.predict()
            tracker.kalman_filter.update(current_points[c])
            tracker.last_seen = frame_idx
            tracker.history.append(current_points[c])
            node_ids[c] = tracker.id
        
        # Handle unmatched current points (new nodes)
        unmatched_current = set(range(len(current_points))) - set(c for _,c in matches)
        for c in unmatched_current:
            new_id = self.next_id
            tracker = NodeTracker(current_points[c], new_id, frame_idx)
            self.trackers.append(tracker)
            node_ids[c] = new_id
            self.next_id += 1
        
        return node_ids

    def _capture_node_edge_screenshot(self, node_edge_file, frame_idx, component_colors):
        """Capture node-edge skeleton with component-matched colors"""
        # Load the node-edge data
        try:
            # print(f"Node-edge file type: {type(node_edge_file)}")
            
            # Check if node_edge_file is already a numpy array
            if isinstance(node_edge_file, np.ndarray):
                # print("Using pre-loaded node-edge array")
                node_edge_data = node_edge_file
            elif isinstance(node_edge_file, str):
                # print(f"Loading node-edge data from file: {node_edge_file}")
                node_edge_data = imread(node_edge_file)
            else:
                raise ValueError("Invalid node_edge_file type - must be path string or numpy array")

            # For timeseries, use the first timepoint for viewpoint selection
            node_edge_data_view = None
            
            if isinstance(node_edge_data, np.ndarray):
                if node_edge_data.ndim == 4:
                    t_idx = 0
                    if hasattr(self, 'selected_timepoints') and self.selected_timepoints:
                        t_idx = min(self.selected_timepoints[0], node_edge_data.shape[0]-1)
                    print(f"Using timepoint {t_idx} for node-edge view")
                    node_edge_data_view = node_edge_data[t_idx]
                else:
                    node_edge_data_view = node_edge_data
            else:
                print(f"Unexpected node-edge data type: {type(node_edge_data)}")
                return self._create_placeholder_image("Error: Unexpected\ndata type")
            
            if node_edge_data_view is None:
                print("Failed to extract node-edge data view")
                return self._create_placeholder_image("Error: Failed to\nextract data view")
            
            # print(f"Node-edge data view shape: {node_edge_data_view.shape}")
            
        except Exception as e:
            import traceback
            print(f"Error loading node-edge data: {e}")
            print(traceback.format_exc())
            return self._create_placeholder_image(f"Error loading\nnode-edge data:\n{str(e)}")
        
        # Create viewer with the same settings
        viewer = napari.Viewer(ndisplay=3)
        
        # Add base labels layer
        labels_layer = viewer.add_labels(node_edge_data, name="Node-Edge Skeleton")

        # Process junctions to find centroids
        junction_mask = (node_edge_data == 4)
        junction_labels = measure.label(junction_mask, connectivity=2)
        regions = measure.regionprops(junction_labels)
        
        # Apply distance-based merging
        merged_labels = np.zeros_like(junction_labels)
        current_label = 1
        processed = set()
        
        for i, r1 in enumerate(regions):
            if i in processed:
                continue
            centroid1 = np.array(r1.centroid)
            to_merge = [i]
            
            for j, r2 in enumerate(regions[i+1:], start=i+1):
                centroid2 = np.array(r2.centroid)
                if np.linalg.norm(centroid1 - centroid2) <= 2:
                    to_merge.append(j)
            
            for idx in to_merge:
                merged_labels[junction_labels == (idx+1)] = current_label
                processed.add(idx)
            current_label += 1
        
        # Get final centroids from merged labels
        final_regions = measure.regionprops(merged_labels)
        junction_centroids = [tuple(map(int, region.centroid)) for region in final_regions]

        points = []
        # Add lone nodes (1) and edges (2) as individual pixels
        for label in [1, 2]:
            coords = np.argwhere(node_edge_data == label)
            points.extend(coords)
        # Add junction centroids
        points.extend(junction_centroids)

        points = np.array(points)
        # Track nodes across frames
        node_ids = self._track_nodes(points, frame_idx)
        
        # Add points with tracked IDs
        points_layer = viewer.add_points(
            points,
            name="Nodes",
            size=3,  # Reduced from 5
            face_color='red',
            edge_color='red',
            symbol='disc'
        )
        
        # Add smaller text labels
        text = {
            'string': [str(nid) for nid in node_ids],
            'color': 'white',
            'anchor': 'center',
            'translation': [-2, 0, 0],  # Reduced from -3
            'size': 6  # Reduced from 12
        }
        points_layer.text = text

        # Extract coordinates for edges
        edge_points = []
        for label in [3]:  # Lone nodes, nodes, junctions
            coords = np.argwhere(node_edge_data == label)
            edge_points.extend(coords)
        if edge_points:
            edge_points = np.array(edge_points)
            
            # Add points with tracked IDs
            points_layer = viewer.add_points(
                edge_points,
                name="Edges",
                size=1,  
                face_color= 'green', # green
                edge_color= 'green', # component colors
                symbol='disc'
            )

        
        # component_volume = self._create_component_labeled_volume(node_edge_data, self.component_edge_pixels)
        # component_labels_layer = viewer.add_labels(
        #     component_volume,
        #     name="Component Edges",
        #     opacity=0.8
        # )

        # Set camera view
        if self.captured_view:
            viewer.camera.angles = self.captured_view[0]
            viewer.camera.center = self.captured_view[1]
            if self.zoom_level:
                viewer.camera.zoom = self.zoom_level

        # # for debugging display napari viewer
        # napari.run()

        # Capture screenshot
        time.sleep(0.5)
        img = viewer.screenshot(canvas_only=True, scale=1.5)
        viewer.close()
        
        return img


    def _create_component_labeled_volume(self, node_edge_data, component_edge_pixels):
        """Replace edge pixels with component IDs"""
        component_volume = node_edge_data.copy()
        for comp_id, pixels in component_edge_pixels.items():
            for (z,y,x) in pixels:
                if component_volume[z,y,x] == 3:  # Only modify edge pixels
                    component_volume[z,y,x] = comp_id
        return component_volume

    def _create_placeholder_image(self, text, width=800, height=400, bg_color=(30, 30, 30), text_color=(200, 200, 200)):
        """
        Create a placeholder image with text.
        
        Parameters:
        -----------
        text : str
            Text to display on the placeholder
        width : int, optional
            Width of the placeholder image
        height : int, optional
            Height of the placeholder image
        bg_color : tuple, optional
            Background color (R, G, B)
        text_color : tuple, optional
            Text color (R, G, B)
        
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

    def _add_timestamp(self, image, timepoint):
        """Add a timestamp to the image"""
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(np.uint8(image))
        else:
            pil_image = image
            
        # Create a drawing context
        draw = ImageDraw.Draw(pil_image)
        
        # Try to get a font (use default if not available)
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

    def _create_gif(self, frame_files, output_file, duration=200):
        """Create a GIF animation with improved quality"""
        print(f"Creating GIF animation from {len(frame_files)} frames...")
        
        # Load all frames
        frames = []
        for filename in frame_files:
            frames.append(imageio.imread(filename))
        
        # Save as GIF with higher quality
        iio_imwrite(output_file, frames, extension='.gif', plugin='pillow', 
                   loop=0, duration=duration/1000.0, optimize=False, quality=100)
        print(f"GIF animation saved to {output_file}")

    def parse_timepoints_string(self, timepoints_str):
        """
        Parse a timepoints string from command line args.
        Format examples: "0,5,10-20,30" or "all"
        Returns a list of timepoint indices.
        """
        if timepoints_str.lower() == 'all':
            return list(range(self.num_timepoints))
        
        selected_timepoints = []
        try:
            for part in timepoints_str.split(','):
                if '-' in part:
                    # Handle range
                    start, end = map(int, part.split('-'))
                    if start < 0 or end >= self.num_timepoints:
                        print(f"Warning: Range {start}-{end} outside valid timepoints (0-{self.num_timepoints-1})")
                        # Clip to valid range
                        start = max(0, start)
                        end = min(self.num_timepoints-1, end)
                    selected_timepoints.extend(range(start, end + 1))
                else:
                    # Handle individual timepoint
                    t = int(part)
                    if t < 0 or t >= self.num_timepoints:
                        print(f"Warning: Timepoint {t} outside valid range (0-{self.num_timepoints-1})")
                        continue
                    selected_timepoints.append(t)
            
            # Remove duplicates and sort
            selected_timepoints = sorted(set(selected_timepoints))
            
            if not selected_timepoints:
                print("No valid timepoints selected. Using all timepoints.")
                return list(range(self.num_timepoints))
            
            print(f"Selected {len(selected_timepoints)} timepoints: {selected_timepoints}")
            return selected_timepoints
        except ValueError as e:
            print(f"Error parsing timepoints string: {e}. Using all timepoints.")
            return list(range(self.num_timepoints))

    def save_viewpoint_config(self, config_file):
        """Save viewpoint and ROI parameters to a JSON file"""
        config = {
            'viewpoint_angles': self.captured_view[0].tolist() if self.captured_view else None,
            'viewpoint_center': self.captured_view[1].tolist() if self.captured_view else None,
            'zoom_level': self.zoom_level,
            'viewport_size': list(self.viewport_size) if self.viewport_size else None,
            'roi_polygon': self.roi_polygon.tolist() if self.roi_polygon is not None else None
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Viewpoint configuration saved to {config_file}")

    def load_viewpoint_config(self, config_file):
        """Load viewpoint and ROI parameters from a JSON file"""
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        if config['viewpoint_angles'] and config['viewpoint_center']:
            self.captured_view = (
                np.array(config['viewpoint_angles']),
                np.array(config['viewpoint_center'])
            )
        self.zoom_level = config['zoom_level']
        self.viewport_size = tuple(config['viewport_size']) if config['viewport_size'] else None
        self.roi_polygon = np.array(config['roi_polygon']) if config['roi_polygon'] else None
        print("Loaded viewpoint configuration:")
        print(f"- Angles: {self.captured_view[0] if self.captured_view else None}")
        print(f"- Center: {self.captured_view[1] if self.captured_view else None}")
        print(f"- Zoom: {self.zoom_level}")
        print(f"- Viewport size: {self.viewport_size}")
        print(f"- ROI polygon: {self.roi_polygon.shape if self.roi_polygon is not None else None} points")

    def _interpolate_missing_frames(self):
        """Fill in missing node positions through linear interpolation"""
        for node_id in self.node_registry:
            frames = sorted(self.node_registry[node_id].keys())
            for i in range(1, len(frames)):
                prev_frame = frames[i-1]
                curr_frame = frames[i]
                
                # Fill gaps between frames
                if curr_frame - prev_frame > 1:
                    start_pos = np.array(self.node_registry[node_id][prev_frame])
                    end_pos = np.array(self.node_registry[node_id][curr_frame])
                    
                    for f in range(prev_frame+1, curr_frame):
                        alpha = (f - prev_frame) / (curr_frame - prev_frame)
                        interp_pos = (1 - alpha) * start_pos + alpha * end_pos
                        self.node_registry[node_id][f] = interp_pos.tolist()

    def _capture_labels_screenshot(self, label_data):
        """Capture screenshot of object labels with random colors"""
        viewer = napari.Viewer(ndisplay=3)
        layer = viewer.add_labels(label_data, name="Object Labels")
        
        # Random colors for all labels except background
        unique_labels = np.unique(label_data)
        colors = {label: np.random.random(3) for label in unique_labels}
        colors[0] = [0, 0, 0]  # Keep background black
        layer.colormap = DirectLabelColormap(color_dict=colors)
        
        # Set camera view
        if self.captured_view:
            viewer.camera.angles = self.captured_view[0]
            viewer.camera.center = self.captured_view[1]
            if self.zoom_level:
                viewer.camera.zoom = self.zoom_level
        
        time.sleep(0.5)
        img = viewer.screenshot(canvas_only=True, scale=1.5)
        viewer.close()
        return img

    

    
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process 3D/4D volumes and create visualizations')
    parser.add_argument('--base-dir', type=str, 
                        default=r"/Users/gabrielsturm/Documents/GitHub/Nellie_MitoGraph/event1_2024-10-22_13-14-25_/crop1_snout/crop1_nellie_out/nellie_necessities",
                        help='Base directory containing the input files')
    parser.add_argument('--skeleton-file', type=str, default="crop1.ome-im_skel.ome.tif",
                        help='Filename of the skeleton volume (relative to base-dir)')
    parser.add_argument('--branch-file', type=str, default="crop1.ome-im_branch_label_reassigned.ome.tif",
                        help='Filename of the branch volume (relative to base-dir)')
    parser.add_argument('--original-file', type=str, default="crop1.ome.ome.tif",
                        help='Filename of the original volume (relative to base-dir)')
    parser.add_argument('--node-edge-file', type=str, default="crop1.ome-im_pixel_class.ome.tif",
                        help='Filename of the node-edge labeled skeleton (relative to base-dir)')
    parser.add_argument('--obj-label-file', type=str, default="crop1.ome-im_obj_label_reassigned.ome.tif",
                        help='Filename of the object label file for coloring (relative to base-dir)')
    parser.add_argument('--output-file', type=str, default="final_view.png",
                        help='Filename for output (relative to base-dir)')
    parser.add_argument('--display', action='store_true',
                        help='Display the final image after processing')
    parser.add_argument('--timepoints', type=str, 
                        help='Timepoints to process (e.g., "0,5,10-20,30"). If not provided, will prompt interactively.')
    parser.add_argument('--intensity-percentile', type=int, default=50,
                        help='Percentile cutoff for intensity thresholding (0-100, default: 50). Higher values show less background.')
    
    args = parser.parse_args()
    
    # Construct full paths
    base_dir = args.base_dir
    skeleton_file = os.path.join(base_dir, args.skeleton_file)
    branch_file = os.path.join(base_dir, args.branch_file)
    original_file = os.path.join(base_dir, args.original_file)
    output_file = os.path.join(base_dir, args.output_file)
    
    # Node-edge and object label files
    node_edge_file = os.path.join(base_dir, args.node_edge_file) if args.node_edge_file else None
    obj_label_file = os.path.join(base_dir, args.obj_label_file) if args.obj_label_file else None
    
    # For a T,Z,Y,X volume (using the first time point), use projection_axis=0 (change if needed)
    selector = ViewpointSelector(skeleton_file, output_file, projection_axis=0, 
                                intensity_percentile=args.intensity_percentile)
    output_file = selector.run_workflow(branch_file, original_file, timepoints=args.timepoints,
                                       node_edge_file=node_edge_file, obj_label_file=obj_label_file)
    
    print("Workflow completed. Output saved as:", output_file)
    
    if args.display:
        try:
            img = imageio.imread(output_file)
            plt.figure(figsize=(12, 10))  # Larger figure for the expanded layout
            plt.imshow(img)
            plt.title("Final 3D View with Node-Edge Skeleton")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print("Error displaying the final image:", e)
    
    return output_file

if __name__ == "__main__":
    main() 