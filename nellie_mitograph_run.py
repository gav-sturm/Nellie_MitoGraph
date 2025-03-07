"""
Nellie_MitoGraph_run Module

This module provides the main entry point for the Nellie MitoGraph visualization application.
It combines the modular components to create a complete workflow for visualizing 3D/4D
mitochondrial structures with multiple data modalities.
"""

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from volume_loader import VolumeLoader
from viewpoint_selector import ViewpointSelector
from roi_selector import ROISelector
from screenshot_manager import ScreenshotManager
from timepoint_manager import TimepointManager
from composite_image_creator import CompositeImageCreator
from gif_generator import GIFGenerator
from node_tracker import NodeTrackerManager
from node_edge_processor import NodeEdgeProcessor
from skimage.transform import resize
from debug_visualizer import DebugVisualizer
from skimage.draw import polygon as sk_polygon

# Create a class-level debug visualizer
debug_visualizer = DebugVisualizer()

class ModularViewpointSelector:
    """
    A modular system for selecting viewpoints and creating visualizations of 3D volumes.
    
    This class serves as the main entry point for selecting viewpoints, capturing screenshots,
    and creating visualizations. It coordinates the other specialized classes.
    """
    
    def __init__(self, volume_file, output_file='final_view.png', intensity_percentile=50):
        """
        Initialize with the path to a volume file and an output filename.
        
        Parameters:
        -----------
        volume_file : str
            Path to the volume file
        output_file : str, optional
            Path to save the output file (default: 'final_view.png')
        intensity_percentile : int, optional
            Percentile cutoff for intensity thresholding (0-100, default: 50)
        """
        self.output_file = output_file
        self.output_dir = os.path.dirname(os.path.abspath(output_file))
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize component objects
        self.volume_loader = VolumeLoader(volume_file, intensity_percentile)
        self.viewpoint_selector = ViewpointSelector()
        self.roi_selector = ROISelector()
        self.screenshot_manager = ScreenshotManager(self.viewpoint_selector)
        self.timepoint_manager = TimepointManager(self.volume_loader.num_timepoints)
        self.node_tracker_manager = NodeTrackerManager()
        
        # Create node edge processor
        self.node_edge_processor = NodeEdgeProcessor(self.viewpoint_selector, self.node_tracker_manager)
        
        # Create GIF generator for timeseries
        gif_filename = os.path.splitext(os.path.basename(output_file))[0] + "_timeseries.gif"
        self.gif_generator = GIFGenerator(self.output_dir, output_file=gif_filename)
        
        # Store common bounding box for all images
        self.common_bbox = None
        self.image_quality = 95
        self.image_DPI = 300
    
    def run_workflow(self, skeleton_file, branch_file, original_file, node_edge_file=None, obj_label_file=None, timepoints=None):
        """
        Run the workflow:
          1. Create a MIP of the skeleton volume and let the user draw an ROI.
          2. Apply the ROI to the 3D volume.
          3. Select viewpoint on the filtered volume.
          4. If timeseries, let the user select timepoints to process.
          5. Capture composite screenshots of all modalities.
        
        Parameters:
        -----------
        skeleton_file : str
            Path to the skeleton volume file
        branch_file : str
            Path to the branch volume file
        original_file : str
            Path to the original volume file
        node_edge_file : str, optional
            Path to the node-edge labeled skeleton file
        obj_label_file : str, optional
            Path to the object label file for coloring the node-edge skeleton
        timepoints : str or list, optional
            Timepoints to process (e.g., "0,5,10-20,30" or [0, 5, 10, 15, 20]). If not provided, will prompt interactively.
            
        Returns:
        --------
        str
            Path to the saved composite image or GIF file
        """
        # First, create a MIP of the skeleton volume
        skeleton_volume = self.volume_loader.volume
        mip = np.max(skeleton_volume, axis=0)
        
        # Let the user draw an ROI on the MIP
        print("Draw an ROI to define the region of interest...")
        self.roi_selector.select_roi(mip)
        
        # Debug print to verify ROI mask was created
        if hasattr(self.roi_selector, 'roi_mask') and self.roi_selector.roi_mask is not None:
            print(f"ROI mask created successfully with shape: {self.roi_selector.roi_mask.shape}")
            # Visualize the ROI mask for debugging
            debug_visualizer.visualize(self.roi_selector.roi_mask, "ROI Mask", save=True)
        else:
            print("Warning: ROI mask was not created!")
        
        # Apply the ROI to the 3D volume
        filtered_skeleton = self._apply_roi_to_volume(skeleton_volume)
        
        # Now select the viewpoint on the filtered volume
        print("Select viewpoint for the 3D visualization...")
        self.viewpoint_selector.select_viewpoint(filtered_skeleton)
        
        # If timepoints were provided, parse them
        if timepoints is not None:
            self.timepoint_manager.parse_timepoints_string(timepoints)
        # Otherwise use interactive selection if it's a timeseries
        elif self.volume_loader.is_timeseries and self.volume_loader.num_timepoints > 1:
            self.timepoint_manager.select_timepoints()
        
        # Initialize the node edge processor's topological graph
        self.node_edge_processor.initialize_topological_graph()
        
        # Capture composite modalities
        if self.volume_loader.is_timeseries:
            output_file = self._process_timeseries(skeleton_file, branch_file, original_file, node_edge_file, obj_label_file)
        else:
            output_file = self._process_single_timepoint(skeleton_file, branch_file, original_file, node_edge_file, obj_label_file)
        
        return output_file
    
    def _apply_roi_to_volume(self, volume):
        """
        Apply the ROI mask to a 3D volume.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume data
            
        Returns:
        --------
        ndarray
            Filtered volume
        """
        if self.roi_selector.roi_mask is None:
            print("No ROI mask available.")
            return volume
        
        # Get the volume shape
        z_dim, y_dim, x_dim = volume.shape
        
        # The ROI mask should already match the y,x dimensions of the volume
        # since it was created on the MIP
        roi_mask_2d = self.roi_selector.roi_mask
        
        # Broadcast the 2D mask to 3D
        mask_3d = np.broadcast_to(roi_mask_2d, volume.shape)
        
        # Apply the mask to the volume
        filtered_volume = np.where(mask_3d, volume, 0)
        
        return filtered_volume
    
    def _process_single_timepoint(self, skeleton_file, branch_file, original_file, node_edge_file=None, obj_label_file=None):
        """
        Process a single timepoint and create a composite image.
        
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
            Path to the saved composite image
        """
        # Load additional volumes
        branch_loader = VolumeLoader(branch_file)
        original_loader = VolumeLoader(original_file)
        obj_label_loader = VolumeLoader(obj_label_file)
        node_edge_loader = VolumeLoader(node_edge_file)
        
        # Apply ROI to all volumes before capturing screenshots
        filtered_branch_data = self._apply_roi_to_volume(branch_loader.volume)
        filtered_original_data = self._apply_roi_to_volume(original_loader.volume)
        filtered_obj_label_data = self._apply_roi_to_volume(obj_label_loader.volume)
        filtered_node_edge_data = self._apply_roi_to_volume(node_edge_loader.volume)
        
        # Capture screenshots of the filtered volumes
        img_branch_label = self.screenshot_manager.capture_screenshot(filtered_branch_data, layer_type='labels')
        img_original = self.screenshot_manager.capture_screenshot(filtered_original_data)
        img_depth = self.screenshot_manager.generate_depth_encoded_image(filtered_original_data)
        img_obj_label = self.screenshot_manager.capture_screenshot(filtered_obj_label_data, layer_type='labels')
        img_node_edge = self.screenshot_manager.capture_screenshot(filtered_node_edge_data, layer_type='labels')
        
        # Process node-edge data using the dedicated NodeEdgeProcessor
        img_node_edge2 = self.node_edge_processor.capture_screenshot(filtered_node_edge_data)
        
        # Process node-edge data and capture screenshots for topological graphs
        img_topo_graph_projected, img_topo_graph_concentric = self.node_edge_processor.get_topological_graph_images(filtered_node_edge_data)
        
        # Collect all images
        all_images = [img_branch_label, img_obj_label, img_original, img_depth, img_node_edge, img_node_edge2]
        
        # No need to apply ROI filtering to images since we already filtered the volumes
        filtered_images = all_images
        
        # Use image 0 as a reference size
        reference_size = filtered_images[0].shape
        
        # Resize the topological graphs to match the other images
        img_topo_graph_projected = resize(img_topo_graph_projected, reference_size, 
                                         preserve_range=True, anti_aliasing=True)
        img_topo_graph_concentric = resize(img_topo_graph_concentric, reference_size, 
                                          preserve_range=True, anti_aliasing=True)
        
        # Add the topological graphs to the filtered images
        filtered_images.append(img_topo_graph_projected)
        filtered_images.append(img_topo_graph_concentric)
        
        # Create composite image
        labels = ["Branch Labels", "Object Labels", "Original", 
                 "Depth Encoded", "Node-Edge", "Node-Edge2", "Projected Graph", "Concentric Graph"]
        
        output_file = os.path.join(self.output_dir, "composite_view.png")
        composite_img = CompositeImageCreator.create_labeled_composite(
            filtered_images, labels, layout='grid', output_file=output_file, title="Composite 3D View"
        )
        
        return output_file
    
    def _process_timeseries(self, skeleton_file, branch_file, original_file, node_edge_file, obj_label_file):
        """
        Process a timeseries of volumes and create a GIF.
        
        Parameters:
        -----------
        skeleton_file : str
            Path to the skeleton volume file
        branch_file : str
            Path to the branch volume file
        original_file : str
            Path to the original volume file
        node_edge_file : str
            Path to the node-edge labeled skeleton file
        obj_label_file : str
            Path to the object label file
        
        Returns:
        --------
        str
            Path to the output GIF file
        """
        # Load all volumes
        skeleton_loader = VolumeLoader(skeleton_file)
        branch_loader = VolumeLoader(branch_file)
        original_loader = VolumeLoader(original_file)
        node_edge_loader = VolumeLoader(node_edge_file)
        obj_label_loader = VolumeLoader(obj_label_file)
        
        # Determine which timepoints to process
        if hasattr(self.timepoint_manager, 'selected_timepoints') and self.timepoint_manager.selected_timepoints:
            selected_timepoints = self.timepoint_manager.selected_timepoints
        else:
            selected_timepoints = range(self.volume_loader.num_timepoints)
            
        print(f"Processing {len(selected_timepoints)} timepoints...")
        
        # Create output directory for frames
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Process each timepoint
        frame_files = []
        padding_params = None
        
        for t_idx, t in enumerate(selected_timepoints):
            print(f"Processing timepoint {t} ({t_idx+1}/{len(selected_timepoints)})...")
            
            # Skip if timepoint is out of range
            if t < 0 or t >= self.volume_loader.num_timepoints:
                print(f"Warning: Timepoint {t} is out of range (0-{self.volume_loader.num_timepoints-1}). Skipping.")
                continue
            
            # Get volume data for this timepoint
            skeleton_data = skeleton_loader.load_volume_for_timepoint(t)
            branch_data = branch_loader.load_volume_for_timepoint(t)
            original_data = original_loader.load_volume_for_timepoint(t)
            obj_label_data = obj_label_loader.load_volume_for_timepoint(t) 
            node_edge_data = node_edge_loader.load_volume_for_timepoint(t) 
            
            # Apply ROI to all volumes before capturing screenshots
            filtered_skeleton_data = self._apply_roi_to_volume(skeleton_data)
            filtered_branch_data = self._apply_roi_to_volume(branch_data)
            filtered_original_data = self._apply_roi_to_volume(original_data)
            filtered_obj_label_data = self._apply_roi_to_volume(obj_label_data)
            filtered_node_edge_data = self._apply_roi_to_volume(node_edge_data)
            
            # Capture screenshots of the filtered volumes
            img_branch = self.screenshot_manager.capture_screenshot(filtered_branch_data, layer_type='branch', timepoint=t)
            img_original = self.screenshot_manager.capture_screenshot(filtered_original_data, layer_type='original', timepoint=t)
            img_depth = self.screenshot_manager.generate_depth_encoded_image(filtered_skeleton_data, timepoint=t)
            img_obj_label = self.screenshot_manager.capture_screenshot(filtered_obj_label_data, layer_type='obj_label', timepoint=t)
            img_node_edge = self.screenshot_manager.capture_screenshot(filtered_node_edge_data, layer_type='node_edge', timepoint=t)
            
            # Process node-edge data and capture screenshots for topological graphs and node-edge visualization
            img_topo_graph_projected, img_topo_graph_concentric, img_node_edge2, _, _ = self.node_edge_processor.get_topological_graph_images(filtered_node_edge_data, frame_idx=t)
            
            # Collect all images
            all_images = [img_branch, img_obj_label, img_original, img_depth, img_node_edge, img_node_edge2]
            
            # No need to apply ROI filtering to images since we already filtered the volumes
            filtered_images = all_images

            # Use image 0 as a reference size
            reference_size = filtered_images[0].shape

            # Resize the topological graphs to match the other images
            img_topo_graph_projected = resize(img_topo_graph_projected, reference_size, 
                                             preserve_range=True, anti_aliasing=True)
            img_topo_graph_concentric = resize(img_topo_graph_concentric, reference_size, 
                                              preserve_range=True, anti_aliasing=True)

            # Add the topological graphs to the filtered images
            filtered_images.append(img_topo_graph_projected)
            filtered_images.append(img_topo_graph_concentric)

            # Create frame with all modalities arranged in a grid
            labels = ["Raw Intensity", "Object labels", "Branch labels", "Depth-encoded",
                       "Node-Edge labels", "Network components", "Projected graph", "Concentric graph"]
            
            # Check that labels and filtered_images are the same length
            if len(labels) != len(filtered_images):
                raise ValueError("Labels and filtered images must be the same length")
            
            # Create the composite image with consistent padding
            composite_img, new_padding_params = CompositeImageCreator.create_composite_image_with_consistent_padding(
                filtered_images, labels, layout='2x4', padding_params=padding_params, timepoint=t
            )
            
            # Store padding parameters from the first timepoint
            if padding_params is None:
                padding_params = new_padding_params
                # print("Stored padding parameters from first timepoint for consistency")
            
            # Save frame
            frame_file = self.gif_generator.save_frame(composite_img, t)
            frame_files.append(frame_file)
        
        # Create GIF from frames
        gif_file = self.gif_generator.create_gif_from_files(frame_files, frame_rate=5)
        
        return gif_file


def main():
    """
    Main entry point for the Nellie MitoGraph application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process 3D/4D volumes and create visualizations')
    parser.add_argument('--base-dir', type=str, 
                        default=r"/Users/gabrielsturm/Documents/GitHub/Nellie_MG/event1_2024-10-22_13-14-25_/crop1_snout/crop1_nellie_out/nellie_necessities",
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
    parser.add_argument('--timepoints', type=str, default='0-100',
                        help='Timepoints to process (e.g., "0,5,10-20,30"). If not provided, will prompt interactively.')
    parser.add_argument('--intensity-percentile', type=int, default=50,
                        help='Percentile cutoff for intensity thresholding (0-2, default: 50). Higher values show less background.')
    
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
    
    # Initialize the ModularViewpointSelector
    selector = ModularViewpointSelector(skeleton_file, output_file, 
                                      intensity_percentile=args.intensity_percentile)
    
    # Run the workflow
    output_file = selector.run_workflow(skeleton_file, branch_file, original_file, 
                                       node_edge_file=node_edge_file, 
                                       obj_label_file=obj_label_file,
                                       timepoints=args.timepoints)
    
    print("Workflow completed. Output saved as:", output_file)
    
    # Display the final image if requested
    if args.display and output_file:
        try:
            import imageio.v2 as imageio
            img = imageio.imread(output_file)
            plt.figure(figsize=(12, 10))
            plt.imshow(img)
            plt.title("Nellie MitoGraph - Final 3D View")
            plt.axis('off')
            plt.show()
        except Exception as e:
            print("Error displaying the final image:", e)
    
    return output_file


if __name__ == "__main__":
    main()