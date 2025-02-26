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
        
    def initialize_topological_graph(self):
        """
        Initialize the TopologicalGraph with viewpoint parameters.
        """
        if self.viewpoint_selector.has_view:
            self.topological_graph = TopologicalGraph(
                self.viewpoint_selector.captured_view,
                self.viewpoint_selector.viewport_size,
                self.viewpoint_selector.zoom_level
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
    
    def capture_screenshot(self, node_edge_data, frame_idx=0, component_colors=None):
        """
        Capture a screenshot of the node-edge skeleton with component-matched colors.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data to visualize
        frame_idx : int, optional
            Frame index (default: 0)
        component_colors : dict, optional
            Mapping of components to colors (default: None)
            
        Returns:
        --------
        ndarray
            Screenshot image
        """
        # Process data if component_colors not provided
        if component_colors is None:
            G, component_colors = self.process_node_edge_data(node_edge_data, frame_idx)
            if G is None or component_colors is None:
                return self._create_placeholder_image("Error processing\nnode-edge data")
        
        try:
            # Create viewer with the same settings
            viewer = napari.Viewer(ndisplay=3)
            
            # Add base labels layer
            labels_layer = viewer.add_labels(node_edge_data, name="Node-Edge Skeleton")
            
            # Process junction regions
            junction_mask, junction_centroids = self._process_junctions(node_edge_data)
            
            # Collect points for nodes
            points = self._collect_node_points(node_edge_data, junction_centroids)
            
            if len(points) > 0:
                # Track nodes across frames
                node_ids = self.node_tracker_manager.track_nodes(points, frame_idx)
                
                # Add points with tracked IDs
                points_layer = viewer.add_points(
                    points,
                    name="Nodes",
                    size=3,
                    face_color='red',
                    edge_color='red',
                    symbol='disc'
                )
                
                # Add text labels
                text = {
                    'string': [str(nid) for nid in node_ids],
                    'color': 'white',
                    'anchor': 'center',
                    'translation': [-2, 0, 0],
                    'size': 6
                }
                points_layer.text = text
            
            # Extract and add edge points
            self._add_edge_points(viewer, node_edge_data)
            
            # Set camera view - CRUCIAL for consistent visualization
            self._set_camera_view(viewer)
            
            # Capture screenshot
            time.sleep(0.5)
            img = viewer.screenshot(canvas_only=True, scale=1.5)
            viewer.close()
            
            return img
            
        except Exception as e:
            print(f"Error capturing node-edge screenshot: {e}")
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
            (junction_mask, junction_centroids) - Binary mask and list of centroids
        """
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
        
        return junction_mask, junction_centroids
    
    def _collect_node_points(self, node_edge_data, junction_centroids):
        """
        Collect points for nodes from the node-edge data.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data
        junction_centroids : list
            List of junction centroid coordinates
            
        Returns:
        --------
        ndarray
            Array of node points
        """
        points = []
        
        # Add lone nodes (1) and terminals (2) as individual pixels
        for label in [1, 2]:
            coords = np.argwhere(node_edge_data == label)
            points.extend(coords)
        
        # Add junction centroids
        points.extend(junction_centroids)
        
        if len(points) > 0:
            return np.array(points)
        else:
            return np.empty((0, 3), dtype=np.int32)
    
    def _add_edge_points(self, viewer, node_edge_data):
        """
        Extract edge points from the node-edge data and add them to the viewer.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            Napari viewer instance
        node_edge_data : ndarray
            Node-edge data
        """
        # Extract coordinates for edges
        edge_points = []
        for label in [3]:  # Edges
            coords = np.argwhere(node_edge_data == label)
            edge_points.extend(coords)
            
        if edge_points and len(edge_points) > 0:
            edge_points = np.array(edge_points)
            
            # Add points for edges
            viewer.add_points(
                edge_points,
                name="Edges",
                size=1,  
                face_color='green',
                edge_color='green',
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
    
    def _create_placeholder_image(self, text, width=800, height=400):
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
            
        Returns:
        --------
        ndarray
            Placeholder image
        """
        from PIL import Image, ImageDraw, ImageFont
        import os
        
        # Create a blank image
        img = Image.new('RGB', (width, height), color=(30, 30, 30))
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
            draw.text(position, line, font=font, fill=(200, 200, 200))
            y += line_heights[i]
        
        return np.array(img)
    
    def get_topological_graph_image(self, node_edge_data, frame_idx=0):
        """
        Generate a topological graph visualization for the node-edge data.
        
        Parameters:
        -----------
        node_edge_data : ndarray
            Node-edge data
        frame_idx : int, optional
            Frame index (default: 0)
            
        Returns:
        --------
        ndarray
            Topological graph visualization image
        """
        if self.topological_graph is None:
            self.initialize_topological_graph()
            
        if self.topological_graph is None:
            return self._create_placeholder_image("Error: Cannot initialize\nTopological Graph")
            
        try:
            # Build graph
            G, component_edge_pixels = self.topological_graph.build_graph_for_frame(node_edge_data)
            
            # Assign colors
            component_colors = self.topological_graph.assign_component_colors(G)
            
            # Capture graph visualization
            img_topo_graph = self.topological_graph._capture_topological_graph(G, component_colors)
            
            return img_topo_graph
        except Exception as e:
            print(f"Error generating topological graph: {e}")
            return self._create_placeholder_image(f"Error generating\ntopological graph:\n{str(e)}")
