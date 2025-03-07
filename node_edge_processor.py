"""
NodeEdgeProcessor Module

This module provides the NodeEdgeProcessor class for processing and visualizing
node-edge skeletons in 3D volumes. It handles the identification of nodes and edges,
merging of junction regions, and visualization with consistent coloring.
"""

import numpy as np
import napari
import time
from skimage import measure
from TopologicalGraph import TopologicalGraph
from PIL import Image, ImageDraw
from screenshot_manager import ScreenshotManager

class NodeEdgeProcessor:
    """
    Processes and visualizes node-edge skeletons in 3D volumes.
    
    This class is responsible for identifying nodes and edges in a 3D skeleton,
    merging junction regions, visualizing with consistent coloring, and tracking
    nodes across frames in timeseries data.
    """
    
    def __init__(self, viewpoint_selector, node_tracker_manager):
        """
        Initialize the NodeEdgeProcessor.
        
        Parameters:
        -----------
        viewpoint_selector : ViewpointSelector
            ViewpointSelector instance with captured view parameters
        node_tracker_manager : NodeTrackerManager
            NodeTrackerManager instance for tracking nodes across frames
        """
        self.viewpoint_selector = viewpoint_selector
        self.node_tracker_manager = node_tracker_manager
        self.topological_graph = None
        self.screenshot_manager = ScreenshotManager(self.viewpoint_selector)
        
    def initialize_topological_graph(self):
        """
        Initialize the TopologicalGraph with viewpoint parameters.
        """
        if self.viewpoint_selector.has_view:
            self.topological_graph = TopologicalGraph(
                self.viewpoint_selector.captured_view,
                self.viewpoint_selector.viewport_size,
                self.viewpoint_selector.zoom_level,
                self.node_tracker_manager
            )
        return self.topological_graph
    
    def process_node_edge_data(self, node_edge_data, frame_idx=0):
        """
        Process node-edge data to build a graph and assign colors.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge labeled skeleton data
        frame_idx : int, optional
            Frame index for timeseries data (default: 0)
            
        Returns:
        --------
        tuple
            (graph, component_colors) - The built graph and assigned component colors
        """
        if self.topological_graph is None:
            self.initialize_topological_graph()
            
        if self.topological_graph is None:
            print("Error: Failed to initialize TopologicalGraph")
            return None, None
            
        # Build graph
        G, component_edge_pixels = self.topological_graph.build_graph_for_frame(node_edge_data)
        
        # Assign colors
        component_colors = self.topological_graph.assign_component_colors(G)
        
        return G, component_colors
    
    def capture_screenshot(self, node_edge_data, G, component_colors=None, frame_idx=0, use_stored_params=False):
        """
        Capture a screenshot of the node-edge skeleton with component-matched colors.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data to visualize
        G : networkx.Graph
            Graph representation of the node-edge data
        component_colors : dict, optional
            Mapping of components to colors (default: None)
        frame_idx : int, optional
            Frame index (default: 0)
        use_stored_params : bool, optional
            Whether to use stored parameters (default: False)
            
        Returns:
        --------
        ndarray
            Screenshot image
        """
        try:
            # Create viewer with the same settings
            viewer = napari.Viewer(ndisplay=3)
            
            # Process junction regions
            junction_mask, junction_centroids = self._process_junctions(node_edge_data)
            
            # Collect points for nodes
            points = self._collect_node_points(node_edge_data, junction_centroids)
            
            if len(points) > 0:
                # Map points to graph node IDs
                node_ids = []
                node_colors = []
                
                # Create a list of all mapped points for nearest-neighbor search
                mapped_points = []
                mapped_node_ids = {}
                mapped_colors = {}
                
                if hasattr(self.topological_graph, 'pixel_to_node'):
                    for pixel, node_id in self.topological_graph.pixel_to_node.items():
                        mapped_points.append(np.array(pixel))
                        mapped_node_ids[pixel] = node_id
                        
                        # Get component color for this node
                        if node_id in G.nodes and 'component' in G.nodes[node_id]:
                            comp_id = G.nodes[node_id]['component']
                            if comp_id in component_colors:
                                mapped_colors[pixel] = component_colors[comp_id]
                
                # Convert to numpy array for efficient distance calculation
                if mapped_points:
                    mapped_points_array = np.array(mapped_points)
                
                for point in points:
                    point_tuple = tuple(point)
                    # Default values
                    node_id = -1
                    color = 'red'
                    
                    # Try to find this point in the pixel_to_node mapping
                    if hasattr(self.topological_graph, 'pixel_to_node') and point_tuple in self.topological_graph.pixel_to_node:
                        # Get the graph node ID
                        graph_node_id = self.topological_graph.pixel_to_node[point_tuple]
                        node_id = graph_node_id  # Use the graph node ID
                        
                        # Get the component color
                        if G.nodes[graph_node_id].get('component') in component_colors:
                            color = component_colors[G.nodes[graph_node_id]['component']]
                    
                    # If we couldn't find a direct mapping, search for the nearest mapped point
                    if node_id == -1 and mapped_points:
                        # Calculate distances to all mapped points
                        distances = np.linalg.norm(mapped_points_array - np.array(point), axis=1)
                        nearest_idx = np.argmin(distances)
                        nearest_point = tuple(mapped_points[nearest_idx])
                        
                        # Use the node ID and color of the nearest point
                        if nearest_point in mapped_node_ids:
                            node_id = mapped_node_ids[nearest_point]
                            
                            if nearest_point in mapped_colors:
                                color = mapped_colors[nearest_point]
                    
                    # If we still couldn't find a mapping, use the point index as the node ID
                    if node_id == -1:
                        node_id = len(node_ids)
                    
                    node_ids.append(node_id)
                    node_colors.append(color)
                
                # Add points with graph node IDs and colors
                points_layer = viewer.add_points(
                    points,
                    name="Nodes",
                    size=4,
                    face_color=node_colors,
                    edge_color=node_colors,
                    symbol='disc'
                )
                
                # Add text labels using graph node IDs
                text = {
                    'string': [str(nid) for nid in node_ids],
                    'color': 'white',
                    'anchor': 'center',
                    'translation': [-2, 0, 0],
                    'size': 8
                }
                points_layer.text = text
            
            # Extract and add edge points with component colors
            self._add_edge_points(viewer, node_edge_data, G, component_colors)
            
            # Set camera view
            self._set_camera_view(viewer)
            
            # Capture screenshot
            time.sleep(0.5)
            img = viewer.screenshot(canvas_only=True, scale=1.5)
            viewer.close()

            # Remove black borders
            use_stored_params = frame_idx is not None and frame_idx > 0
            img = self.screenshot_manager._crop_black_borders(img, image_type="node_edge2", use_stored_params=use_stored_params)
            
            return img
        except Exception as e:
            print(f"Error capturing node-edge screenshot: {e}")
            import traceback
            traceback.print_exc()
            return self._create_placeholder_image(f"Error capturing\nnode-edge data:\n{str(e)}")
    
    def _process_junctions(self, node_edge_data):
        """
        Process junction regions in the node-edge data.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data
            
        Returns:
        --------
        tuple
            (junction_mask, junction_centroids) - Binary mask of junctions and their centroids
        """
        # Create a mask for junctions (class 4)
        junction_mask = (node_edge_data == 4)
        
        # Label connected components in the junction mask
        junction_labels = measure.label(junction_mask, connectivity=2)
        
        # Get region properties
        regions = measure.regionprops(junction_labels)
        
        # Extract centroids
        junction_centroids = [region.centroid for region in regions]
        
        return junction_mask, junction_centroids
    
    def _collect_node_points(self, node_edge_data, junction_centroids):
        """
        Collect points for nodes from the node-edge data.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data
        junction_centroids : list
            List of junction centroids
            
        Returns:
        --------
        ndarray
            Array of node points
        """
        # Collect terminal points (class 2)
        terminal_mask = (node_edge_data == 2)
        terminal_points = np.argwhere(terminal_mask)
        
        # Combine with junction centroids
        all_points = []
        
        # Add terminal points
        if len(terminal_points) > 0:
            all_points.extend(terminal_points)
        
        # Add junction centroids
        if junction_centroids:
            all_points.extend([np.round(centroid).astype(int) for centroid in junction_centroids])
        
        return np.array(all_points) if all_points else np.empty((0, 3), dtype=int)
    
    def _add_edge_points(self, viewer, node_edge_data, G=None, component_colors=None):
        """
        Extract edge points from the node-edge data and add them to the viewer.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            Napari viewer instance
        node_edge_data : ndarray
            Node-edge data
        G : networkx.Graph, optional
            Graph representation of the node-edge data
        component_colors : dict, optional
            Mapping of components to colors
        """
        # Default color if component_colors is not available
        default_color = 'green'
        
        # Extract coordinates for edges
        edge_mask = (node_edge_data == 3)
        z_coords, y_coords, x_coords = np.where(edge_mask)
        
        if len(z_coords) == 0:
            return
        
        # Don't sample points - use all edge pixels for complete visualization
        edge_points = np.column_stack((z_coords, y_coords, x_coords))
        
        # Create a list of all mapped points for nearest-neighbor search
        mapped_points = []
        mapped_colors = {}
        
        if hasattr(self.topological_graph, 'pixel_to_color') and self.topological_graph.pixel_to_color:
            for pixel, color in self.topological_graph.pixel_to_color.items():
                mapped_points.append(np.array(pixel))
                mapped_colors[pixel] = color
        
        # Convert to numpy array for efficient distance calculation
        if mapped_points:
            mapped_points_array = np.array(mapped_points)
        
        # Use the pixel_to_color mapping if available
        if hasattr(self.topological_graph, 'pixel_to_color') and self.topological_graph.pixel_to_color:
            edge_colors = []
            for point in edge_points:
                point_tuple = tuple(point)
                # Try direct mapping first
                if point_tuple in self.topological_graph.pixel_to_color:
                    edge_colors.append(self.topological_graph.pixel_to_color[point_tuple])
                elif hasattr(self.topological_graph, 'pixel_to_component') and point_tuple in self.topological_graph.pixel_to_component:
                    comp_id = self.topological_graph.pixel_to_component[point_tuple]
                    if comp_id in component_colors:
                        edge_colors.append(component_colors[comp_id])
                    else:
                        edge_colors.append(default_color)
                # If no direct mapping, try nearest-neighbor search
                elif mapped_points:
                    # Calculate distances to all mapped points
                    distances = np.linalg.norm(mapped_points_array - np.array(point), axis=1)
                    nearest_idx = np.argmin(distances)
                    nearest_point = tuple(mapped_points[nearest_idx])
                    
                    # Use the color of the nearest point
                    if nearest_point in mapped_colors:
                        edge_colors.append(mapped_colors[nearest_point])
                    else:
                        edge_colors.append(default_color)
                else:
                    edge_colors.append(default_color)
        else:
            edge_colors = default_color
        
        # Add points for edges
        viewer.add_points(
            edge_points,
            name="Edges",
            size=1,
            face_color=edge_colors,
            edge_color=edge_colors,
            symbol='disc'
        )
    
    def _set_camera_view(self, viewer):
        """
        Set the camera view to match the captured viewpoint.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            Napari viewer instance
        """
        if self.viewpoint_selector.has_view:
            viewer.camera.angles = self.viewpoint_selector.captured_view[0]
            viewer.camera.center = self.viewpoint_selector.captured_view[1]
            if self.viewpoint_selector.zoom_level:
                viewer.camera.zoom = self.viewpoint_selector.zoom_level
    
    def _create_placeholder_image(self, text, width=400, height=400):
        """
        Create a placeholder image with text.
        
        Parameters:
        -----------
        text : str
            Text to display on the placeholder
        width : int, optional
            Width of the placeholder (default: 400)
        height : int, optional
            Height of the placeholder (default: 400)
            
        Returns:
        --------
        ndarray
            Placeholder image
        """
        # Create a blank image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :] = [50, 50, 50]  # Dark gray background
        
        # Convert to PIL Image for text drawing
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Get a font
        try:
            from composite_image_creator import CompositeImageCreator
            font = CompositeImageCreator.get_universal_font(20)
        except:
            # Use default font if CompositeImageCreator is not available
            try:
                from PIL import ImageFont
                font = ImageFont.load_default()
            except:
                font = None
        
        # Draw text
        text_lines = text.split('\n')
        y_offset = height // 2 - len(text_lines) * 15
        for line in text_lines:
            # Calculate text width to center it
            if font:
                try:
                    text_width = draw.textlength(line, font=font)
                except:
                    text_width = len(line) * 10  # Approximate width
            else:
                text_width = len(line) * 10  # Approximate width
            
            x_offset = (width - text_width) // 2
            draw.text((x_offset, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 30
        
        # Convert back to numpy array
        return np.array(pil_img)
    
    def get_topological_graph_images(self, node_edge_data, frame_idx=0):
        """
        Generate two topological graph visualizations for the node-edge data:
        1. Projected layout based on 3D coordinates
        2. Concentric circle layout
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data
        frame_idx : int, optional
            Frame index (default: 0)
            
        Returns:
        --------
        tuple
            (projected_graph_img, concentric_graph_img, node_edge_img, G, component_colors)
            - Two graph visualizations, node-edge screenshot, graph, and component colors
        """
        try:
            # Ensure node-edge data is valid
            if node_edge_data is None or node_edge_data.size == 0:
                raise ValueError("Empty or None node-edge data")
            
            # Ensure topological graph is initialized
            if self.topological_graph is None:
                self.initialize_topological_graph()
            
            if self.topological_graph is None:
                raise RuntimeError("Failed to initialize topological graph")
            
            # Apply ROI to node-edge data if ROI is available
            if hasattr(self.viewpoint_selector, 'roi_selector') and self.viewpoint_selector.roi_selector is not None:
                # Get the ROI mask
                roi_mask = self.viewpoint_selector.roi_selector.roi_mask
                if roi_mask is not None:
                    # Create a 3D mask by broadcasting the 2D ROI mask
                    roi_mask_3d = np.broadcast_to(roi_mask[..., np.newaxis], node_edge_data.shape)
                    # Apply the mask to the node-edge data
                    filtered_node_edge_data = np.where(roi_mask_3d, node_edge_data, 0)
                    # Use the filtered data for graph building
                    node_edge_data = filtered_node_edge_data
            
            # Build graph with frame index for consistent node tracking
            G, component_edge_pixels = self.topological_graph.build_graph_for_frame(node_edge_data, frame_idx)
            
            # Store the graph in the topological graph object for reference
            self.topological_graph.G = G
            
            # Ensure the graph has component information propagated to nodes
            if hasattr(self.topological_graph, '_propagate_component_info_to_nodes'):
                self.topological_graph._propagate_component_info_to_nodes(G)
            
            # Check if graph is empty
            if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
                raise ValueError("Empty graph: No nodes or edges found")
            
            # Assign colors
            component_colors = self.topological_graph.assign_component_colors(G)
            
            # Print some debug info
            node_with_component = sum(1 for _, data in G.nodes(data=True) if 'component' in data)
            print(f"Nodes with component info: {node_with_component} out of {G.number_of_nodes()}")
            
            # Capture graph visualizations with both layouts
            img_topo_graph_projected = self.topological_graph._capture_topological_graph(G, component_colors)
            img_topo_graph_concentric = self.topological_graph.capture_topological_graph_concentric(G, component_colors)

            # Capture node-edge screenshot with the same frame index for consistent node IDs
            node_edge_img = self.capture_screenshot(node_edge_data, G, component_colors, frame_idx)
            
            # Ensure all images have consistent format (RGB)
            for img in [img_topo_graph_projected, img_topo_graph_concentric, node_edge_img]:
                # Ensure the image has at least 3 color channels for consistency
                if img.ndim == 2:
                    img = np.stack([img]*3, axis=-1)
                elif img.shape[2] == 1:
                    img = np.repeat(img, 3, axis=2)
                elif img.shape[2] == 4:
                    img = img[:,:,:3]  # Drop alpha channel if present
                
                # Normalize pixel values if needed
                if img.dtype != np.uint8:
                    img = (img / img.max() * 255).astype(np.uint8)
            
            return img_topo_graph_projected, img_topo_graph_concentric, node_edge_img, G, component_colors
            
        except Exception as e:
            print(f"Error generating topological graphs: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            
            # Create a more informative placeholder
            placeholder_text = f"Error:\n{type(e).__name__}\n{str(e)}"
            placeholder = self._create_placeholder_image(
                placeholder_text, 
                width=800, 
                height=800
            )
            
            # Ensure placeholder is 3-channel
            if placeholder.ndim == 2:
                placeholder = np.stack([placeholder]*3, axis=-1)
            elif placeholder.shape[2] == 4:
                placeholder = placeholder[:,:,:3]
            
            # Return placeholders for all images
            return placeholder, placeholder, placeholder, None, None