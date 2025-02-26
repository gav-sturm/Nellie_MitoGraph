import sys
import os
import time
import napari
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from tifffile import imread, imwrite
import imageio.v2 as imageio
from pathlib import Path
import glob
import pickle
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize
from skimage.measure import regionprops, label
from matplotlib.colors import ListedColormap, BoundaryNorm
from ViewpointSelector import ViewpointSelector
import traceback

# Import the GraphBuilder_localized functionality
try:
    from GraphBuilder_localized import GraphBuilder3D
    HAS_GRAPHBUILDER = True
except ImportError:
    print("Error: Could not import GraphBuilder_localized. This module is required.")
    print("Please ensure GraphBuilder_localized.py is in the same directory or PYTHONPATH.")
    sys.exit(1)

# Import functions from GraphBuilder_localized or define them here
def transform_coordinate(coord, azim, center):
    """Transform 3D coordinate to 2D for graph visualization, accounting for view angle."""
    # Convert azimuth to radians
    azim_rad = np.radians(azim)
    
    # Shift relative to center point
    rel_x = coord[0] - center[0]
    rel_y = coord[1] - center[1]
    
    # Apply rotation based on view angle
    new_x = rel_x * np.cos(azim_rad) - rel_y * np.sin(azim_rad)
    new_y = rel_x * np.sin(azim_rad) + rel_y * np.cos(azim_rad)
    
    return (new_x, new_y)

def adjust_layout_to_bbox(pos, target_bbox=(-10, 10, -10, 10)):
    """Scale and shift node positions to fit within the target bounding box."""
    if not pos:
        return pos
        
    # Extract current bounds
    xs, ys = zip(*pos.values())
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Target bounds
    target_min_x, target_max_x, target_min_y, target_max_y = target_bbox
    
    # Calculate scaling factors
    x_scale = (target_max_x - target_min_x) / max(1e-10, (max_x - min_x))
    y_scale = (target_max_y - target_min_y) / max(1e-10, (max_y - min_y))
    scale = min(x_scale, y_scale)
    
    # Apply scaling and shifting
    new_pos = {}
    for node, (x, y) in pos.items():
        # Scale
        scaled_x = (x - min_x) * scale
        scaled_y = (y - min_y) * scale
        
        # Shift to target bbox
        new_x = target_min_x + scaled_x
        new_y = target_min_y + scaled_y
        
        new_pos[node] = (new_x, new_y)
    
    return new_pos

class TopologicalViewpointAnalyzer:
    def __init__(self, base_dir=None, output_dir=None):
        """
        Initialize with the base directory containing skeleton, branch, and original files.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing the MitoGraph output files. If None, uses hardcoded path.
        output_dir : str, optional
            Directory for output files. If None, uses base_dir.
        """
        # Use provided base_dir or default to hardcoded path
        if base_dir is None:
            self.base_dir = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2025-02-13_yeast_mitographs\event1_2024-10-22_13-14-25_\crop1_snout\crop1_nellie_out\nellie_necessities"
        else:
            self.base_dir = base_dir
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "output_graphs")
        else:
            self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize ViewpointSelector properties
        self.viewpoint_selector = None
        self.composite_image = None
        self.captured_view = None
        
        # Initialize graph properties
        self.graph = None
        self.components = None
        self.node_color_map = {}
        self.global_ids = {}
        self.view_azim = 0  # Default view azimuth
        self.view_center = (0, 0, 0)  # Default view center
        
    def process_all_frames(self, time_points=None):
        """
        Process all time points, or a specific list of time points.
        Workflow:
        1. Select a single viewpoint and ROI using the first timepoint
        2. Apply that viewpoint and ROI to all timepoints
        3. Generate graphs and visualizations for each timepoint
        
        Parameters:
        -----------
        time_points : list, optional
            List of time points to process. If None, processes all available timepoints.
        
        Returns:
        --------
        dict
            Mapping of time points to output files
        """
        # Set up file paths using the patterns from the original code
        base_path = Path(self.base_dir)
        
        # Get the base filename - in this case, for a single file with all timepoints
        base_filename = "crop1"
        
        # Define paths to the files containing all timepoints
        skeleton_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_skel.ome.tif")
        branch_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_branch_label_reassigned.ome.tif")
        original_file = os.path.join(self.base_dir, f"{base_filename}.ome.ome.tif")
        
        # Try alternative branch file pattern if needed
        if not os.path.exists(branch_file):
            branch_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_branches.ome.tif")
            
        # Verify the files exist
        if not os.path.exists(skeleton_file):
            print(f"Error: Skeleton file not found: {skeleton_file}")
            return {}
        if not os.path.exists(branch_file):
            print(f"Error: Branch file not found: {branch_file}")
            return {}
        if not os.path.exists(original_file):
            print(f"Error: Original file not found: {original_file}")
            return {}
        
        # Load files to get timepoint information
        print(f"Loading skeleton file to determine number of timepoints: {skeleton_file}")
        skeleton_data = imread(skeleton_file)
        
        # Determine total number of available timepoints
        if skeleton_data.ndim == 4:  # Data format is [T, Z, Y, X]
            num_timepoints = skeleton_data.shape[0]
            print(f"Note: Data is {skeleton_data.ndim}D, treating as a single timepoint")
        else:
            # Handle 3D data as a single timepoint
            num_timepoints = 1
            print(f"Note: Data is {skeleton_data.ndim}D, treating as a single timepoint")
        
        # Determine which timepoints to process
        if time_points is None:
            # Process all available timepoints
            time_points = range(num_timepoints)
            print(f"Will process all {num_timepoints} timepoints")
        else:
            # Filter to valid timepoints only
            time_points = [t for t in time_points if 0 <= t < num_timepoints]
            if len(time_points) == 0:
                print(f"Error: No valid timepoints in range 0-{num_timepoints-1}")
                return {}
                
            print(f"Will process {len(time_points)} selected timepoints: {time_points}")
        
        if not time_points:
            print("No valid timepoints to process.")
            return {}
        
        # Initialize ViewpointSelector with the first timepoint for selection
        reference_timepoint = time_points[0]
        reference_output_file = os.path.join(self.output_dir, f"timepoint_{reference_timepoint:04d}_view.png")
        
        # Extract the first timepoint for reference
        if skeleton_data.ndim == 4:
            reference_skeleton_data = skeleton_data[reference_timepoint]
        else:
            reference_skeleton_data = skeleton_data

        # Create a temporary file for the reference timepoint
        reference_skeleton_temp = os.path.join(self.output_dir, f"temp_reference_skeleton_t{reference_timepoint}.tif")
        imwrite(reference_skeleton_temp, reference_skeleton_data)
        
        try:
            self.viewpoint_selector = ViewpointSelector(
                volume_file=reference_skeleton_temp,
                output_file=reference_output_file
            )
            
            # Select viewpoint (opens interactive viewer)
            print(f"\nSelecting viewpoint using timepoint {reference_timepoint}...")
            self.captured_view = self.viewpoint_selector.select_viewpoint(reference_skeleton_temp)
            
            # Select ROI if needed (opens interactive viewer)
            print("Selecting ROI (close the viewer when finished)...")
            self.viewpoint_selector.select_roi()
            
            # Store the ROI for re-use with other timepoints
            reference_roi = self.viewpoint_selector.roi_polygon
            print(f"ROI selected with {len(reference_roi) if reference_roi is not None else 0} points")
            
        finally:
            # Clean up temporary file
            if os.path.exists(reference_skeleton_temp):
                os.remove(reference_skeleton_temp)
        
        results = {}
        
        # Process each timepoint with the same viewpoint and ROI
        for idx, t in enumerate(time_points):
            print(f"\nProcessing timepoint {t} ({idx+1}/{len(time_points)})...")
            
            # Check if timepoint is valid
            if t >= num_timepoints:
                print(f"Error: Timepoint {t} exceeds available timepoints. Skipping.")
                continue
                
            # Extract data for current timepoint
            if skeleton_data.ndim == 4:
                current_skeleton_data = skeleton_data[t]
            else:
                current_skeleton_data = skeleton_data

            # Load branch and original data for this timepoint
            branch_data = imread(branch_file)
            if branch_data.ndim == 4:
                current_branch_data = branch_data[t]
            else:
                current_branch_data = branch_data
                
            original_data = imread(original_file)
            if original_data.ndim == 4:
                current_original_data = original_data[t]
            else:
                current_original_data = original_data

            # Create temporary files for this timepoint
            temp_skeleton = os.path.join(self.output_dir, f"temp_skeleton_t{t}.tif")
            temp_branch = os.path.join(self.output_dir, f"temp_branch_t{t}.tif")
            temp_original = os.path.join(self.output_dir, f"temp_original_t{t}.tif")
            
            # Write temporary files
            imwrite(temp_skeleton, current_skeleton_data)
            imwrite(temp_branch, current_branch_data)
            imwrite(temp_original, current_original_data)
            
            # Create final output filename
            output_file = os.path.join(self.output_dir, f"timepoint_{t:04d}_view.png")
            
            # Create ViewpointSelector for this timepoint with reference viewpoint and ROI
            current_viewpoint_selector = ViewpointSelector(
                volume_file=temp_skeleton,
                output_file=output_file
            )
            
            # Set the captured viewpoint and ROI
            current_viewpoint_selector.captured_view = self.captured_view
            current_viewpoint_selector.roi_polygon = reference_roi
            
            try:
                # Capture screenshots with the consistent viewpoint and ROI
                print(f"Capturing modalities for timepoint {t}...")
                try:
                    # Capture modalities but don't save the composite yet
                    composite_image_path = current_viewpoint_selector.capture_modalities_with_preset_view(
                        temp_branch,
                        temp_original,
                        output_file=output_file
                    )
                except Exception as e:
                    print(f"Error capturing modalities: {e}")
                    traceback.print_exc()
                    raise
                
                # Try to load graph from GraphBuilder_localized output
                graph_loaded = self._load_existing_graph(t)
                
                # If graph not found, generate one (fallback)
                if not graph_loaded:
                    print("No pre-existing graph found. Generating a new one...")
                    self._generate_topological_graph(temp_branch)
                    self._match_component_colors()
                
                # Create visualization with node overlay and graph
                final_vis_file = self._create_visualization(t, composite_image_path)
                
                # Store only the final visualization file in results
                results[t] = final_vis_file
                print(f"Completed time point {t}, output saved to {final_vis_file}")
                
                # Clean up intermediate files
                if os.path.exists(composite_image_path):
                    os.remove(composite_image_path)
                
                # Remove any possible debug/intermediate files
                intermediate_files = [
                    os.path.splitext(output_file)[0] + "_composite.png",
                    os.path.splitext(output_file)[0] + "_depth_encoded.png",
                    os.path.splitext(output_file)[0] + "_depth_debug.png"
                ]
                for file_path in intermediate_files:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            
            except Exception as e:
                print(f"Error processing timepoint {t}: {e}")
                traceback.print_exc()
    
        return results
    
    def _find_files(self, pattern):
        """Find files matching the given pattern in base_dir, sorted by name."""
        import glob
        files = sorted(glob.glob(os.path.join(self.base_dir, pattern)))
        return files
    
    def _run_viewpoint_selection(self, skeleton_file, branch_file, original_file):
        """Run the ViewpointSelector workflow on the given files."""
        # Select viewpoint (opens interactive viewer)
        self.captured_view = self.viewpoint_selector.select_viewpoint(skeleton_file)
        
        # Select ROI if needed (opens interactive viewer)
        self.viewpoint_selector.select_roi()
        
        # Capture modalities and create composite image
        self.composite_image = self.viewpoint_selector.capture_modalities(branch_file, original_file)
        
        return self.composite_image
    
    def _generate_topological_graph(self, branch_file):
        """
        Generate a topological graph from the branch file using GraphBuilder_localized
        functionality.
        
        Parameters:
        -----------
        branch_file : str
            Branch file path
        """
        print(f"Generating graph from {branch_file}")
        
        # Load branch data
        branch_data = imread(branch_file)
        if branch_data.ndim == 4:  # [T, Z, Y, X]
            branch_data = branch_data[0]  # Take first timepoint if 4D
        
        try:
            print("Generating topological graph from branch data...")
            # Get the base filename pattern
            base_filename = "crop1"
            
            # Define file paths required by GraphBuilder3D
            pixel_class_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_pixel_class.ome.tif")
            branch_label_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_branch_label_reassigned.ome.tif")
            skeleton_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_skel.ome.tif")
            raw_file = os.path.join(self.base_dir, f"{base_filename}.ome.ome.tif")
            
            # Check if branch_label_file exists, if not try alternative
            if not os.path.exists(branch_label_file):
                branch_label_file = os.path.join(self.base_dir, f"{base_filename}.ome-im_branches.ome.tif")
            
            # Require pixel class file - no fallback
            if not os.path.exists(pixel_class_file):
                raise FileNotFoundError(f"Required pixel class file not found: {pixel_class_file}")
                
            pixel_class_data = imread(pixel_class_file)
            if pixel_class_data.ndim == 4:  # [T, Z, Y, X]
                pixel_class_data = pixel_class_data[0]  # Take first timepoint if 4D
            print(f"Loaded pixel class data with shape {pixel_class_data.shape}")
            
            # Create a new graph
            self.graph = nx.Graph()
            
            # Define pixel class values (from GraphBuilder_localized)
            TERMINAL_CLASS = 2
            EDGE_CLASS = 3
            JUNCTION_CLASS = 4
            
            # Find junctions (class 4)
            junction_mask = (pixel_class_data == JUNCTION_CLASS)
            junction_labels = label(junction_mask, connectivity=3)
            junction_props = regionprops(junction_labels)
            
            # Create mapping from junction label to node ID
            junction_mapping = {}
            node_counter = 0
            
            # Add junction nodes
            for prop in junction_props:
                rep = tuple(map(int, np.round(prop.centroid)))
                junction_mapping[prop.label] = node_counter
                self.graph.add_node(node_counter, coord=rep, type='junction')
                node_counter += 1
            
            # Find terminal points (class 2)
            terminal_coords = [tuple(coord) for coord in np.argwhere(pixel_class_data == TERMINAL_CLASS)]
            terminal_mapping = {}
            
            # Add terminal nodes
            for coord in terminal_coords:
                if coord not in terminal_mapping:
                    terminal_mapping[coord] = node_counter
                    self.graph.add_node(node_counter, coord=coord, type='terminal')
                    node_counter += 1
            
            # Helper function to get node ID from coordinate
            def get_node_id(coord):
                if pixel_class_data[coord] == TERMINAL_CLASS:
                    return terminal_mapping.get(coord, None)
                elif pixel_class_data[coord] == JUNCTION_CLASS:
                    lab = junction_labels[coord]
                    return junction_mapping.get(lab, None)
                return None
            
            # Helper function to get 3D neighbors
            def get_neighbors_3d(coord, shape):
                z, y, x = coord
                neighbors = []
                for dz in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dz == 0 and dy == 0 and dx == 0:
                                continue
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                                neighbors.append((nz, ny, nx))
                return neighbors
            
            # Track visited pixels
            visited = np.zeros(pixel_class_data.shape, dtype=bool)
            
            # Get all node coordinates
            all_node_coords = [self.graph.nodes[node]['coord'] for node in self.graph.nodes()]
            
            # Trace paths between nodes
            for start_coord in all_node_coords:
                start_id = get_node_id(start_coord)
                if start_id is None:
                    continue
                
                # Check each neighbor of the starting node
                for nb in get_neighbors_3d(start_coord, pixel_class_data.shape):
                    # If neighbor is an edge pixel and not visited
                    if pixel_class_data[nb] == EDGE_CLASS and not visited[nb]:
                        # Start tracing a path
                        path = [start_coord, nb]
                        visited[nb] = True
                        current = nb
                        reached_id = None
                        
                        # Continue tracing until we reach another node or dead end
                        while True:
                            # Find candidates for next step (neighbors not in path)
                            candidates = [n for n in get_neighbors_3d(current, pixel_class_data.shape) 
                                         if n not in path]
                            
                            if not candidates:
                                break
                                
                            # Look for next pixel (edge, terminal, or junction)
                            next_pixel = None
                            for candidate in candidates:
                                if pixel_class_data[candidate] in (EDGE_CLASS, TERMINAL_CLASS, JUNCTION_CLASS):
                                    next_pixel = candidate
                                    break
                                    
                            if next_pixel is None:
                                break
                                
                            # Add to path and mark as visited
                            path.append(next_pixel)
                            visited[next_pixel] = True
                            
                            # Check if we've reached another node
                            node_id = get_node_id(next_pixel)
                            if node_id is not None and node_id != start_id:
                                reached_id = node_id
                                break
                                
                            # Continue from this pixel
                            current = next_pixel
                        
                        # If we reached another node, add an edge
                        if reached_id is not None:
                            if not self.graph.has_edge(start_id, reached_id):
                                self.graph.add_edge(start_id, reached_id, weight=len(path), path=path)
            
            print(f"Generated graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            
            # Identify connected components
            self.components = list(nx.connected_components(self.graph))
            
        except Exception as e:
            print(f"Error generating topological graph: {e}")
            traceback.print_exc()
            raise RuntimeError("Failed to generate topological graph from branch data.")
        
    def _match_component_colors(self):
        """
        Assign consistent colors to connected components in the graph.
        """
        # Create a colormap with enough distinct colors
        num_components = len(self.components)
        
        # Use a perceptually uniform colormap
        cmap = plt.cm.get_cmap('tab10', num_components)
        
        # Assign colors to components
        for i, component in enumerate(self.components):
            color = cmap(i)
            
            # Assign to all nodes in the component
            for node in component:
                self.node_color_map[node] = to_rgba(color)
    
    def _load_existing_graph(self, time_point):
        """
        Try to load an existing graph from GraphBuilder_localized output.
        
        Returns:
        --------
        bool
            True if graph was successfully loaded, False otherwise
        """
        # Look for graph files in the output_graphs directory
        graph_dir = os.path.join(self.base_dir, "output_graphs")
        
        # Try different potential filenames
        potential_files = [
            os.path.join(graph_dir, f"graph_data_frame_{time_point}.pkl"),
            os.path.join(graph_dir, f"graph_data_t{time_point}.pkl"),
            os.path.join(graph_dir, f"graph_frame_{time_point}.pkl")
        ]
        
        for file_path in potential_files:
            if os.path.exists(file_path):
                try:
                    print(f"Loading graph from {file_path}")
                    with open(file_path, 'rb') as f:
                        graph_data = pickle.load(f)
                    
                    # Extract graph and attributes from the loaded data
                    if isinstance(graph_data, dict):
                        # Expected format from GraphBuilder_localized
                        self.graph = graph_data.get('graph', None)
                        self.node_color_map = graph_data.get('node_colors', {})
                        self.global_ids = graph_data.get('global_ids', {})
                        self.view_azim = graph_data.get('view_azim', 0)
                        self.view_center = graph_data.get('view_center', (0, 0, 0))
                        
                        # If no graph was found in the dict, return False
                        if self.graph is None:
                            print("No graph found in the loaded data")
                            return False
                            
                        # Extract components if not already present
                        if self.components is None:
                            self.components = list(nx.connected_components(self.graph))
                        
                        return True
                    elif isinstance(graph_data, nx.Graph):
                        # Just the graph was saved
                        self.graph = graph_data
                        # Generate default colors
                        self._match_component_colors()
                        self.components = list(nx.connected_components(self.graph))
                        return True
                    else:
                        print(f"Unrecognized graph data format: {type(graph_data)}")
                except Exception as e:
                    print(f"Error loading graph from {file_path}: {e}")
        
        print("No valid graph file found")
        return False
    
    def _create_visualization(self, time_point, composite_image_path):
        """
        Create a visualization that combines:
        1. The composite image from ViewpointSelector
        2. Topological graph with matched component colors
        
        Returns:
        --------
        str
            Path to the saved visualization file
        """
        # Load the composite image
        composite = plt.imread(composite_image_path)
        
        # Create figure with similar layout to GraphBuilder_localized
        fig = plt.figure(figsize=(20, 12), facecolor='black')
        
        # Create a GridSpec layout
        gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[2, 1], 
                         hspace=0.2, wspace=0.1)
        
        # Top span: Composite image (spans both columns)
        ax1 = fig.add_subplot(gs[0, :])
        # Bottom Right: Graph view
        ax_graph = fig.add_subplot(gs[1, 1])
        # Bottom Left: Extra space for potential future use
        ax_extra = fig.add_subplot(gs[1, 0])
        ax_extra.axis('off')
        
        # Display composite image at the top
        ax1.imshow(composite)
        ax1.set_title("3D Mitochondrial View with ROI", color='white', fontsize=18)
        ax1.axis('off')
        
        # Add bottom caption to describe modalities
        ax1.text(0.5, -0.05, "From left to right: Original, Branches, Skeleton, Depth-Encoded", 
                transform=ax1.transAxes, color='white', fontsize=12,
                horizontalalignment='center')
        
        # Draw the graph using the same approach as in GraphBuilder_localized
        if self.graph and len(self.graph.nodes()) > 0:
            pos = {}
            
            # Use graph's node coordinates if available, otherwise use centroids
            for node in self.graph.nodes():
                if 'coord' in self.graph.nodes[node]:
                    coord = self.graph.nodes[node]['coord']
                elif 'centroid_3d' in self.graph.nodes[node]:
                    coord = self.graph.nodes[node]['centroid_3d']
                else:
                    # Fallback to position attribute if present
                    pos_attr = self.graph.nodes[node].get('pos', (0, 0))
                    coord = (pos_attr[0], pos_attr[1], 0)  # Add Z=0 if missing
                
                # Transform 3D coordinate to 2D based on view azimuth
                pos[node] = transform_coordinate(coord, self.view_azim, self.view_center)
            
            # Adjust layout to fit within standard bounding box
            pos = adjust_layout_to_bbox(pos, target_bbox=(-10, 10, -10, 10))
            
            # Get node colors from the map or use fallback colormap
            if not self.node_color_map:
                # Create a colormap with distinct colors
                cmap = plt.cm.get_cmap('tab10', len(self.components) if self.components else 10)
                for i, component in enumerate(self.components or []):
                    color = cmap(i)
                    for node in component:
                        self.node_color_map[node] = to_rgba(color)
            
            node_colors = [self.node_color_map.get(node, "#000000") for node in self.graph.nodes()]
            
            # Get edge colors (use source node color)
            edge_colors = [self.node_color_map.get(u, "#000000") for u, v in self.graph.edges()]
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, ax=ax_graph, edge_color=edge_colors, width=2)
            
            # Draw nodes
            # Adjust node size based on type if available
            node_sizes = []
            for node in self.graph.nodes():
                node_type = self.graph.nodes[node].get('type', 'branch')
                if node_type == 'node':
                    node_sizes.append(400)  # Larger for endpoints
                elif node_type == 'junction':
                    node_sizes.append(350)  # Medium for junctions
                else:
                    node_sizes.append(250)  # Smaller for regular branches
            
            nx.draw_networkx_nodes(self.graph, pos, ax=ax_graph, 
                                  node_color=node_colors, 
                                  node_size=node_sizes)
            
            # Create labels
            if self.global_ids:
                labels = self.global_ids
            else:
                # Create sequential labels
                labels = {node: str(i) for i, node in enumerate(self.graph.nodes())}
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, labels=labels, ax=ax_graph, 
                                   font_color='white', font_size=10)
            
            ax_graph.set_title(f"Graph View (Frame {time_point})", fontsize=18, color='white')
            ax_graph.axis('off')
            ax_graph.set_facecolor('black')
            
            # Add help text in bottom left
            ax_extra.text(0.5, 0.5, 
                         "Topological Graph represents\nthe connectivity of mitochondrial segments\n"
                         "with nodes colored by connected component.",
                         ha='center', va='center', color='white', fontsize=14)
            ax_extra.set_facecolor('black')
            
            # Save figure
            # Use a more descriptive name for the final output
            output_file = os.path.join(self.output_dir, f"timepoint_{time_point:04d}_final.png")
            fig.savefig(output_file, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            
            return output_file
        else:
            print("No graph available to visualize")
            # Create a simple error message in the graph area
            ax_graph.text(0.5, 0.5, "No graph data available", 
                         ha='center', va='center', color='white', fontsize=14)
            ax_graph.set_facecolor('black')
            ax_graph.axis('off')
            
            # Save figure anyway
            output_file = os.path.join(self.output_dir, f"timepoint_{time_point:04d}_final.png")
            fig.savefig(output_file, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            
            return output_file

def main():
    """Main function to run the TopologicalViewpointAnalyzer."""
    import argparse
    
    # Create argument parser with optional dir argument
    parser = argparse.ArgumentParser(
        description='Analyze mitochondrial topology with interactive viewpoint selection')
    parser.add_argument('--dir', type=str, help='Base directory containing MitoGraph output files')
    parser.add_argument('--out', type=str, help='Output directory (defaults to base directory)')
    parser.add_argument('--timepoints', type=str, help='Comma-separated list of time points to process (e.g., "0,1,5")')
    
    args = parser.parse_args()
    
    # Get base directory (use None to trigger default hardcoded path)
    base_dir = args.dir
    
    # Process time points if specified
    time_points = None
    if args.timepoints:
        time_points = [int(t) for t in args.timepoints.split(',')]
    
    # Create analyzer
    analyzer = TopologicalViewpointAnalyzer(base_dir, args.out)
    
    # Process frames
    results = analyzer.process_all_frames(time_points)
    
    print(f"\nProcessed {len(results)} time points.")
    print(f"Output files saved to {analyzer.output_dir}")
    
    return results

if __name__ == "__main__":
    main() 