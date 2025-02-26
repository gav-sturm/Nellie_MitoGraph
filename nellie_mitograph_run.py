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
    
    def run_workflow(self, branch_file, original_file, node_edge_file=None, obj_label_file=None, timepoints=None):
        """
        Run the workflow:
          1. Select viewpoint on the skeleton volume.
          2. Let the user draw an ROI on the captured skeleton screenshot.
          3. If timeseries, let the user select timepoints to process.
          4. Capture composite screenshots of all modalities.
        
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
        timepoints : str or list, optional
            Timepoints to process (e.g., "0,5,10-20,30" or [0, 5, 10, 15, 20]). If not provided, will prompt interactively.
            
        Returns:
        --------
        str
            Path to the saved composite image or GIF file
        """
        # First, select the viewpoint from the skeleton
        print("Select viewpoint for the 3D visualization...")
        self.viewpoint_selector.select_viewpoint(self.volume_loader.volume)
        
        # Then, let the user draw an ROI on the captured skeleton screenshot
        if hasattr(self.viewpoint_selector, 'screenshot_img'):
            print("Draw an ROI to define the region of interest...")
            self.roi_selector.select_roi(self.viewpoint_selector.screenshot_img)
        
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
            output_file = self._process_timeseries(branch_file, original_file, node_edge_file, obj_label_file)
        else:
            output_file = self._process_single_timepoint(branch_file, original_file, node_edge_file, obj_label_file)
        
        return output_file
    
    def _process_single_timepoint(self, branch_file, original_file, node_edge_file=None, obj_label_file=None):
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
        obj_label_loader = VolumeLoader(obj_label_file) if obj_label_file else None
        node_edge_loader = VolumeLoader(node_edge_file) if node_edge_file else None
        
        # Capture screenshots with the exact same viewpoint
        print("Capturing Skeleton screenshot...")
        img_skel = self.screenshot_manager.capture_screenshot(self.volume_loader.volume)
        
        print("Capturing Branch Labels screenshot...")
        img_branch = self.screenshot_manager.capture_screenshot(branch_loader.volume, layer_type='labels')
        
        print("Capturing Original Volume screenshot...")
        img_original = self.screenshot_manager.capture_screenshot(original_loader.volume)
        
        print("Generating Depth-Encoded image...")
        img_depth = self.screenshot_manager.generate_depth_encoded_image(self.volume_loader.volume)
        
        print("Capturing Object Labels screenshot...")
        img_obj_label = None
        if obj_label_loader:
            img_obj_label = self.screenshot_manager.capture_screenshot(obj_label_loader.volume, layer_type='labels')
        else:
            img_obj_label = self.screenshot_manager.create_placeholder_image("Object Labels\nNot Available")
        
        # Process node-edge data using the dedicated NodeEdgeProcessor
        print("Processing Node-Edge data...")
        img_node_edge = None
        img_topo_graph = None
        if node_edge_loader:
            # Process node-edge data and capture screenshots
            img_node_edge = self.node_edge_processor.capture_screenshot(node_edge_loader.volume, frame_idx=0)
            img_topo_graph = self.node_edge_processor.get_topological_graph_image(node_edge_loader.volume, frame_idx=0)
        else:
            img_node_edge = self.screenshot_manager.create_placeholder_image("Node-Edge\nNot Available")
            img_topo_graph = self.screenshot_manager.create_placeholder_image("Topological Graph\nNot Available")
        
        # Apply ROI filtering and cropping consistently to all images
        print("Applying ROI and creating composite image...")
        all_images = [img_skel, img_branch, img_obj_label, img_original, img_depth, img_node_edge, img_topo_graph]
        
        # First, apply ROI to each image and get all bounding boxes
        filtered_images = []
        all_bboxes = []
        for img in all_images:
            filtered_img, bbox = self.roi_selector.apply_roi_to_image(img)
            filtered_images.append(filtered_img)
            all_bboxes.append(bbox)
        
        # Find the union of all bounding boxes to ensure consistent cropping
        min_row = min(bbox[0] for bbox in all_bboxes)
        max_row = max(bbox[1] for bbox in all_bboxes)
        min_col = min(bbox[2] for bbox in all_bboxes)
        max_col = max(bbox[3] for bbox in all_bboxes)
        common_bbox = (min_row, max_row, min_col, max_col)
        self.common_bbox = common_bbox
        
        # Crop all images with the common bounding box
        cropped_images = [self.roi_selector.crop_to_bbox(img, common_bbox) for img in filtered_images]
        
        # Ensure all images have the same dimensions (for consistent display)
        final_images = self.roi_selector.ensure_consistent_dimensions(cropped_images)
        
        # Create composite image
        labels = ["Skeleton", "Branch Labels", "Object Labels", "Original", 
                 "Depth Encoded", "Node-Edge", "Topological Graph"]
        
        output_file = os.path.join(self.output_dir, "composite_view.png")
        composite_img = CompositeImageCreator.create_labeled_composite(
            final_images, labels, layout='grid', output_file=output_file, title="Composite 3D View"
        )
        
        return output_file
    
    def _process_timeseries(self, branch_file, original_file, node_edge_file, obj_label_file):
        """
        Process a timeseries and create a GIF animation.
        
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
            Path to the saved GIF file
        """
        # Load all volumes
        branch_loader = VolumeLoader(branch_file)
        original_loader = VolumeLoader(original_file)
        node_edge_loader = VolumeLoader(node_edge_file)
        obj_label_loader = VolumeLoader(obj_label_file)
        
        # Determine which timepoints to process
        if hasattr(self.timepoint_manager, 'selected_timepoints') and self.timepoint_manager.selected_timepoints:
            timepoints = self.timepoint_manager.selected_timepoints
        else:
            timepoints = range(self.volume_loader.num_timepoints)
            
        print(f"Processing {len(timepoints)} timepoints...")
        
        # Create output directory for frames
        frames_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
            
        # Find a common bounding box for the first frame to ensure consistency across frames
        if self.common_bbox is None and len(timepoints) > 0:
            first_timepoint = timepoints[0]
            # Get volume data for first timepoint
            skeleton_data = self.volume_loader.load_volume_for_timepoint(first_timepoint)
            img_skel = self.screenshot_manager.capture_screenshot(skeleton_data)
            filtered_img, bbox = self.roi_selector.apply_roi_to_image(img_skel)
            self.common_bbox = bbox
        
        # Process each selected timepoint
        frame_files = []
        for t_idx, t in enumerate(timepoints):
            print(f"Processing timepoint {t} ({t_idx+1}/{len(timepoints)})...")
            
            # Skip if timepoint is out of range
            if t < 0 or t >= self.volume_loader.num_timepoints:
                print(f"Warning: Timepoint {t} is out of range (0-{self.volume_loader.num_timepoints-1}). Skipping.")
                continue
            
            # Get volume data for this timepoint
            skeleton_data = self.volume_loader.load_volume_for_timepoint(t)
            branch_data = branch_loader.load_volume_for_timepoint(t)
            original_data = original_loader.load_volume_for_timepoint(t)
            obj_label_data = obj_label_loader.load_volume_for_timepoint(t) if obj_label_loader else None
            node_edge_data = node_edge_loader.load_volume_for_timepoint(t) if node_edge_loader else None
            
            # Capture screenshots using the same viewpoint for all
            img_skel = self.screenshot_manager.capture_screenshot(skeleton_data)
            img_branch = self.screenshot_manager.capture_screenshot(branch_data, layer_type='labels')
            img_original = self.screenshot_manager.capture_screenshot(original_data)
            img_depth = self.screenshot_manager.generate_depth_encoded_image(skeleton_data)
            
            # Handle optional volumes
            img_obj_label = None
            if obj_label_data is not None:
                img_obj_label = self.screenshot_manager.capture_screenshot(obj_label_data, layer_type='labels')
            else:
                img_obj_label = self.screenshot_manager.create_placeholder_image("Object Labels\nNot Available")
            
            # Process node-edge data using the dedicated NodeEdgeProcessor
            img_node_edge = None
            img_topo_graph = None
            if node_edge_data is not None:
                # Process node-edge data and capture screenshots
                img_node_edge = self.node_edge_processor.capture_screenshot(node_edge_data, frame_idx=t)
                img_topo_graph = self.node_edge_processor.get_topological_graph_image(node_edge_data, frame_idx=t)
            else:
                img_node_edge = self.screenshot_manager.create_placeholder_image("Node-Edge\nNot Available")
                img_topo_graph = self.screenshot_manager.create_placeholder_image("Topological Graph\nNot Available")
            
            # Collect all images for consistent processing
            all_images = [img_skel, img_branch, img_obj_label, img_original, img_depth, img_node_edge, img_topo_graph]
            
            # Apply ROI filtering to all images
            filtered_images = []
            for img in all_images:
                filtered_img, _ = self.roi_selector.apply_roi_to_image(img)
                filtered_images.append(filtered_img)
            
            # Use the common bounding box from the first frame for consistency
            if self.common_bbox:
                cropped_images = [self.roi_selector.crop_to_bbox(img, self.common_bbox) for img in filtered_images]
            else:
                # If no common bbox (shouldn't happen), use individual bboxes
                cropped_images = []
                for img in filtered_images:
                    _, bbox = self.roi_selector.apply_roi_to_image(img)
                    cropped_img = self.roi_selector.crop_to_bbox(img, bbox)
                    cropped_images.append(cropped_img)
            
            # Ensure all images have the same dimensions
            final_images = self.roi_selector.ensure_consistent_dimensions(cropped_images)
            
            # Create frame with all modalities arranged in a grid
            labels = ["Skeleton", "Branch Labels", "Object Labels", "Original", 
                     "Depth Encoded", "Node-Edge", "Topological Graph"]
            
            composite_img = CompositeImageCreator.create_labeled_composite(
                final_images, labels, layout='2x4'
            )
            
            # Add timestamp to the composite image
            composite_with_timestamp = self.screenshot_manager.add_timestamp(composite_img, t)
            
            # Save frame
            frame_file = self.gif_generator.save_frame(composite_with_timestamp, t)
            frame_files.append(frame_file)
        
        if not frame_files:
            print("No frames were generated. Check your timepoint selection.")
            return None
        
        # Create GIF from frames
        gif_file = self.gif_generator.create_gif_from_files(frame_files)
        
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
    parser.add_argument('--timepoints', type=str, default="0,1,2",
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
    
    # Initialize the ModularViewpointSelector
    selector = ModularViewpointSelector(skeleton_file, output_file, 
                                      intensity_percentile=args.intensity_percentile)
    
    # Run the workflow
    output_file = selector.run_workflow(branch_file, original_file, 
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