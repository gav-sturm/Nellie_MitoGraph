import networkx as nx
import numpy as np
from skimage import measure
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import itertools

class TopologicalGraph:
    def __init__(self, captured_view=None, viewport_size=None, zoom_level=None, node_tracker_manager=None):
        """
        Initialize the TopologicalGraph.
        
        Parameters:
        -----------
        captured_view : tuple, optional
            Tuple containing camera parameters (angles, center)
        viewport_size : tuple, optional
            Size of the viewport (width, height)
        zoom_level : float, optional
            Zoom level
        node_tracker_manager : NodeTrackerManager, optional
            Manager for tracking nodes across timepoints
        """
        self.component_edge_pixels = defaultdict(list)
        self.current_component_id = 1000
        self.prev_components = []
        self.color_index = 0
        self.G = None  # Initialize the graph as None
        self.global_nodes = {}  # {global_id: (z,y,x)}
        self.next_global_node_id = 1
        self.global_threshold = 2.0  # Pixels

        # capture view parameters
        self.captured_view = captured_view
        self.angles, self.center = self.captured_view if captured_view else (None, None)
        self.zoom_level = zoom_level
        self.viewport_size = viewport_size
        self.node_tracker_manager = node_tracker_manager
        self.frame_idx = 0  # Initialize the frame index
        self.node_size = 2000
        self.edge_width = 6
        self.node_font_size = 24

    def build_graph_for_frame(self, pixel_class_vol, frame_idx=0):
        """
        Enhanced junction merging with expanded connectivity and distance checks.
        Uses NodeTrackerManager for consistent node IDs across frames.
        
        Parameters:
        -----------
        pixel_class_vol : ndarray
            Volume with pixel classifications
        frame_idx : int, optional
            Frame index for tracking nodes across timepoints (default: 0)
            
        Returns:
        --------
        tuple
            (G, component_to_pixels) - The built graph and component pixel mappings
        """
        import networkx as nx  # Add this import to ensure nx is available
        from collections import defaultdict  # Also ensure defaultdict is available
        
        junction_mask = (pixel_class_vol == 4)
        
        # Use full 3D connectivity (26-connectivity) for initial labeling
        junction_labels = measure.label(junction_mask, connectivity=3)  # Increased connectivity
        regions = measure.regionprops(junction_labels)
        
        # Merge regions within a larger distance threshold
        merged_labels = np.zeros_like(junction_labels)
        current_label = 1
        processed = set()
        
        # Increased merging distance threshold
        merge_distance_threshold = 2.0  # Increased from 2.0
        
        for i, r1 in enumerate(regions):
            if i in processed:
                continue
            # Find nearby regions using centroid distances
            centroid1 = np.array(r1.centroid)
            to_merge = [i]
            
            for j, r2 in enumerate(regions[i+1:], start=i+1):
                centroid2 = np.array(r2.centroid)
                if np.linalg.norm(centroid1 - centroid2) <= merge_distance_threshold:
                    to_merge.append(j)
            
            # Merge all selected regions
            for idx in to_merge:
                merged_labels[junction_labels == (idx+1)] = current_label
                processed.add(idx)
            current_label += 1
        
        # Update junction labels with merged regions
        junction_labels = merged_labels
        junction_props = measure.regionprops(junction_labels)
        
        G = nx.Graph()
        
        # Create a mapping from pixel coordinates to node IDs
        pixel_to_node = {}
        
        # Collect all node points for tracking
        all_node_points = []
        node_types = []  # 'junction' or 'terminal'
        node_coords = []  # Store original coordinates
        
        # Add junctions
        for prop in junction_props:
            coord = tuple(map(int, np.round(prop.centroid)))
            all_node_points.append(np.array(coord))
            node_types.append('junction')
            node_coords.append(coord)
        
        # Add terminals
        terminal_coords = [tuple(coord) for coord in np.argwhere(pixel_class_vol == 2)]
        for coord in terminal_coords:
            all_node_points.append(np.array(coord))
            node_types.append('terminal')
            node_coords.append(coord)
        
        # Use NodeTrackerManager to get consistent node IDs across frames
        if hasattr(self, 'node_tracker_manager') and self.node_tracker_manager is not None:
            node_ids = self.node_tracker_manager.track_nodes(all_node_points, frame_idx)
        else:
            # Fallback to sequential IDs if no tracker is available
            node_ids = list(range(1, len(all_node_points) + 1))
        
        # Now add nodes to the graph with tracked IDs
        for i, (point, node_type, node_id) in enumerate(zip(node_coords, node_types, node_ids)):
            G.add_node(node_id, coord=point, type=node_type)
            
            # For junctions, map all pixels in the junction to this node ID
            if node_type == 'junction':
                prop_idx = i
                if prop_idx < len(junction_props):
                    for pixel_coord in junction_props[prop_idx].coords:
                        pixel_to_node[tuple(pixel_coord)] = node_id
            else:
                # For terminals, just map the terminal pixel
                pixel_to_node[point] = node_id
        
        # Process edges with improved connectivity detection
        visited = np.zeros_like(pixel_class_vol, dtype=bool)
        all_node_coords = [G.nodes[node]['coord'] for node in G.nodes()]
        edge_paths = {}  # Store paths for each edge
        
        # Map all edge pixels (class 3) to their coordinates for later assignment
        all_edge_pixels = [tuple(coord) for coord in np.argwhere(pixel_class_vol == 3)]
        
        # Create a node lookup dictionary for faster access
        node_lookup = {G.nodes[node]['coord']: node for node in G.nodes()}
        
        # Increased search radius for edge connectivity
        edge_search_radius = 3  # Increased from default
        
        for start_coord in all_node_coords:
            start_id = node_lookup.get(start_coord)
            if start_id is None:
                continue
            
            # Get neighbors within the increased search radius
            neighbors = []
            z, y, x = start_coord
            for dz in range(-edge_search_radius, edge_search_radius + 1):
                for dy in range(-edge_search_radius, edge_search_radius + 1):
                    for dx in range(-edge_search_radius, edge_search_radius + 1):
                        if dz == 0 and dy == 0 and dx == 0:
                            continue
                        nz, ny, nt = z + dz, y + dy, x + dx
                        if (0 <= nz < pixel_class_vol.shape[0] and 
                            0 <= ny < pixel_class_vol.shape[1] and 
                            0 <= nt < pixel_class_vol.shape[2]):
                            neighbors.append((nz, ny, nt))
            
            for nb in neighbors:
                if pixel_class_vol[nb] == 3 and not visited[nb]:
                    path = [start_coord, nb]
                    visited[nb] = True
                    current = nb
                    reached_id = None
                    
                    while True:
                        # Get all neighbors of the current pixel
                        candidates = []
                        cz, cy, cx = current
                        for dz in range(-1, 2):
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    if dz == 0 and dy == 0 and dx == 0:
                                        continue
                                    nz, ny, nt = cz + dz, cy + dy, cx + dx
                                    if (0 <= nz < pixel_class_vol.shape[0] and 
                                        0 <= ny < pixel_class_vol.shape[1] and 
                                        0 <= nt < pixel_class_vol.shape[2] and
                                        (nz, ny, nt) not in path and 
                                        pixel_class_vol[nz, ny, nt] in (2, 3, 4)):
                                        candidates.append((nz, ny, nt))
                        
                        if not candidates:
                            break
                            
                        next_pixel = None
                        # First check for node pixels (terminal or junction)
                        for candidate in candidates:
                            if pixel_class_vol[candidate] in (2, 4):
                                next_pixel = candidate
                                # Check if this is a node
                                if candidate in node_lookup:
                                    reached_id = node_lookup[candidate]
                                    if reached_id != start_id:
                                        break
                                break
                        
                        # If no node pixel found, use an unvisited edge pixel
                        if next_pixel is None:
                            for candidate in candidates:
                                if pixel_class_vol[candidate] == 3 and not visited[candidate]:
                                    next_pixel = candidate
                                    break
                        
                        if next_pixel is None:
                            break
                            
                        path.append(next_pixel)
                        visited[next_pixel] = True
                        current = next_pixel
                        
                        # Check if reached another node
                        if current in node_lookup:
                            node_id = node_lookup[current]
                            if node_id != start_id:
                                reached_id = node_id
                                break

                    if reached_id is not None and not G.has_edge(start_id, reached_id):
                        # Add edge with path information
                        G.add_edge(start_id, reached_id, weight=len(path), path=path)
                        edge_paths[(start_id, reached_id)] = path

        # Now identify connected components
        components = list(nx.connected_components(G))
        
        # Initialize component mapping
        component_to_pixels = defaultdict(list)
        pixel_to_component = {}
        
        # Assign component IDs and map pixels to components
        for comp_idx, component in enumerate(components, start=1):
            # Get all edges in this component
            component_edges = [(u, v) for u, v in G.edges() if u in component and v in component]
            
            # Collect all pixels in the component's edges
            component_pixels = []
            for u, v in component_edges:
                if (u, v) in edge_paths:
                    component_pixels.extend(edge_paths[(u, v)])
                elif (v, u) in edge_paths:
                    component_pixels.extend(edge_paths[(v, u)])
            
            # Add all node pixels to the component
            for node in component:
                node_coord = G.nodes[node]['coord']
                component_pixels.append(node_coord)
            
            # Store component pixels
            component_to_pixels[comp_idx].extend(component_pixels)
            
            # Map each pixel to this component
            for pixel in component_pixels:
                pixel_to_component[pixel] = comp_idx
            
            # Assign component ID to nodes
            for node in component:
                G.nodes[node]['component'] = comp_idx
            
            # Assign component ID to edges
            for u, v in component_edges:
                G[u][v]['component'] = comp_idx
        
        # Check for any unmapped edge pixels and assign them to the nearest component
        unmapped_edge_pixels = [p for p in all_edge_pixels if p not in pixel_to_component]
        if unmapped_edge_pixels:
            # print(f"Found {len(unmapped_edge_pixels)} unmapped edge pixels. Assigning to nearest component...")
            
            # For each unmapped edge pixel, find the nearest mapped pixel
            for pixel in unmapped_edge_pixels:
                min_dist = float('inf')
                nearest_comp = None
                
                for comp_idx, pixels in component_to_pixels.items():
                    for comp_pixel in pixels:
                        dist = np.linalg.norm(np.array(pixel) - np.array(comp_pixel))
                        if dist < min_dist:
                            min_dist = dist
                            nearest_comp = comp_idx
                
                if nearest_comp is not None:
                    pixel_to_component[pixel] = nearest_comp
                    component_to_pixels[nearest_comp].append(pixel)

        # Assign colors to components
        component_colors = self.assign_component_colors(G)
        
        # Create a mapping from pixel coordinates to colors
        pixel_to_color = {}
        for pixel, comp_id in pixel_to_component.items():
            if comp_id in component_colors:
                pixel_to_color[pixel] = component_colors[comp_id]
        
        # Store these mappings for later use
        self.pixel_to_node = pixel_to_node
        self.pixel_to_component = pixel_to_component
        self.pixel_to_color = pixel_to_color
        self.component_colors = component_colors
        
        # Print some statistics
        # print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        # print(f"Found {len(components)} connected components")
        # print(f"Mapped {len(pixel_to_node)} pixels to nodes")
        # print(f"Mapped {len(pixel_to_component)} pixels to components")
        # print(f"Mapped {len(pixel_to_color)} pixels to colors")
        
        return G, component_to_pixels
    
    def _propagate_component_info_to_nodes(self, G):
        """
        Propagate component information from edges to nodes.
        
        Parameters:
        -----------
        G : networkx.Graph
            The graph to update
        """
        # For each node, find all its edges and assign the most common component
        for node in G.nodes():
            # Get all edges connected to this node
            edges = list(G.edges(node, data=True))
            
            if not edges:
                continue
            
            # Count components
            component_counts = {}
            for _, _, data in edges:
                if 'component' in data:
                    component = data['component']
                    component_counts[component] = component_counts.get(component, 0) + 1
            
            # Assign the most common component to the node
            if component_counts:
                most_common_component = max(component_counts.items(), key=lambda x: x[1])[0]
                G.nodes[node]['component'] = most_common_component
                # print(f"Assigned component {most_common_component} to node {node}")

    # Path finding and edge creation
    def get_neighbors_3d(self, coord, shape):
        z, y, x = coord
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz = z + dz
                    ny = y + dy
                    nx = x + dx
                    if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                        neighbors.append((nz, ny, nx))
        return neighbors

    def assign_component_colors(self, G):
        """
        Assign colors to components in the graph with temporal consistency.
        Ensures no two components in the current frame have the same color.
        
        Parameters:
        -----------
        G : networkx.Graph
            The graph to color
            
        Returns:
        --------
        dict
            Mapping of component IDs to colors
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Get connected components
        components = list(nx.connected_components(G))
        
        # Calculate component features for matching
        component_features = []
        for comp_idx, comp in enumerate(components, start=1):
            # Get all nodes in this component
            nodes = [G.nodes[n]['coord'] for n in comp if 'coord' in G.nodes[n]]
            
            if nodes:
                # Calculate centroid and size
                centroid = np.mean(nodes, axis=0)
                size = len(nodes)
                
                # Store component info
                component_features.append((comp_idx, comp, centroid, size))
        
        # Create a larger color palette by combining multiple colormaps
        num_components = len(component_features)
        
        # Generate a large palette of distinct colors
        color_palette = []
        
        # Add colors from tab20
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.tab20(range(20))])
        
        # Add colors from tab20b
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.tab20b(range(20))])
        
        # Add colors from tab20c
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.tab20c(range(20))])
        
        # Add colors from Set1
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.Set1(range(9))])
        
        # Add colors from Set2
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.Set2(range(8))])
        
        # Add colors from Set3
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.Set3(range(12))])
        
        # Add colors from Paired
        color_palette.extend([mcolors.to_hex(c) for c in plt.cm.Paired(range(12))])
        
        # If we still need more colors, create variations by adjusting brightness
        if num_components > len(color_palette):
            base_colors = color_palette.copy()
            for color in base_colors:
                # Create a darker version
                rgb = mcolors.to_rgb(color)
                darker = tuple(max(0, c * 0.7) for c in rgb)
                color_palette.append(mcolors.to_hex(darker))
                
                # Create a lighter version
                lighter = tuple(min(1, c * 1.3) for c in rgb)
                color_palette.append(mcolors.to_hex(lighter))
        
        # Assign colors to components with temporal consistency
        component_colors = {}
        
        # If we have previous components, try to match them
        if hasattr(self, 'prev_components') and self.prev_components:
            for comp_idx, comp, centroid, size in component_features:
                best_match = None
                best_score = 0
                
                for prev_comp_idx, prev_comp, prev_centroid, prev_size, prev_color in self.prev_components:
                    # Calculate overlap of nodes (using global IDs for consistency)
                    current_globals = set(self.get_or_create_global_id(G.nodes[n]['coord']) 
                                         for n in comp if 'coord' in G.nodes[n])
                    prev_globals = set(prev_comp)
                    
                    overlap = len(current_globals.intersection(prev_globals))
                    
                    # Calculate distance between centroids
                    distance = np.linalg.norm(centroid - prev_centroid)
                    
                    # Calculate size similarity
                    size_similarity = 1 - abs(size - prev_size) / max(size, prev_size)
                    
                    # Combined score (higher is better)
                    score = overlap + (1 - distance/100) + size_similarity
                    
                    if score > best_score:
                        best_score = score
                        best_match = prev_color
                
                # If we found a good match, use that color
                if best_match and best_score > 1.5:
                    component_colors[comp_idx] = best_match
                else:
                    # Otherwise, assign a new color
                    color_idx = len(component_colors) % len(color_palette)
                    component_colors[comp_idx] = color_palette[color_idx]
        else:
            # First frame, just assign colors sequentially
            for i, (comp_idx, comp, centroid, size) in enumerate(component_features):
                color_idx = i % len(color_palette)
                component_colors[comp_idx] = color_palette[color_idx]

        # check if any 2 components have the same color
        color_conflicts = {}
        for comp_idx, color in component_colors.items():
            if list(component_colors.values()).count(color) > 1:
                if color not in color_conflicts:
                    color_conflicts[color] = []
                color_conflicts[color].append((comp_idx, next(size for idx, _, _, size in component_features if idx == comp_idx)))
        
        # Resolve color conflicts - larger components keep their color, smaller ones get new colors
        for color, conflicts in color_conflicts.items():
            # Sort conflicts by component size (descending)
            conflicts.sort(key=lambda x: x[1], reverse=True)
            
            # Keep the color for the largest component
            largest_comp_idx = conflicts[0][0]
            
            # Assign new colors to smaller components
            for comp_idx, _ in conflicts[1:]:
                # Find a new color that's not already used
                for color_idx in range(len(color_palette)):
                    candidate_color = color_palette[color_idx]
                    if candidate_color not in component_colors.values():
                        component_colors[comp_idx] = candidate_color
                        break
                else:
                    # If we've exhausted the palette, create a random color
                    r, g, b = np.random.random(3)
                    component_colors[comp_idx] = mcolors.to_hex((r, g, b))
        
        # Store current components for next frame
        self.prev_components = []
        for comp_idx, comp, centroid, size in component_features:
            # Store global IDs of nodes for better matching
            global_ids = [self.get_or_create_global_id(G.nodes[n]['coord']) 
                         for n in comp if 'coord' in G.nodes[n]]
            
            # Store the color that was assigned
            color = component_colors[comp_idx]
            
            # Add to previous components
            self.prev_components.append((comp_idx, global_ids, centroid, size, color))
        
        # print(f"Assigned colors to {len(component_colors)} components")
        return component_colors
    
    def _get_node_id(self, coord, G, pixel_class_vol):
        """Updated to use global node registry"""
        # Find node with matching coordinate
        for node in G.nodes():
            if np.linalg.norm(np.array(G.nodes[node]['coord']) - np.array(coord)) < 2:
                return node
        return None

    def _capture_topological_graph(self, G, component_colors, viewpoint_selector=None):
        """
        Capture a screenshot of the topological graph with node and edge coloring by component.
        Applies viewpoint transformations directly to the graph positions.
        
        Parameters:
        -----------
        G : networkx.Graph
            Graph to visualize
        component_colors : dict
            Mapping of component IDs to colors
        viewpoint_selector : ViewpointSelector, optional
            ViewpointSelector instance to get transformations from (default: None)
            
        Returns:
        --------
        ndarray
            Graph visualization image
        """
        if G is None or len(G.nodes) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Use enhanced layout to prevent component overlap
        pos = self._create_non_overlapping_layout(G)
        
        # Apply transformations from viewpoint if available
        if viewpoint_selector is not None and viewpoint_selector.has_view:
            # Get the transformations needed to match the 3D viewpoint
            transformations = viewpoint_selector.determine_2d_transformations()
            
            if transformations:
                # print("Applying transformations to graph positions...")
                # Apply transformations to the node positions
                pos = self._apply_transformations_to_positions(pos, transformations)
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Group edges by component for coloring
        component_edge_groups = {}
        for u, v, data in G.edges(data=True):
            if 'component' in data and u in pos and v in pos:
                component = data['component']
                if component not in component_edge_groups:
                    component_edge_groups[component] = []
                component_edge_groups[component].append((u, v))
        
        # Draw edges by component with appropriate colors - increased width
        for component, edges in component_edge_groups.items():
            if component in component_colors:
                color = component_colors[component]
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=self.edge_width, alpha=0.8, 
                                      edge_color=color, ax=ax)  # Increased width
        
        # Group nodes by component for coloring
        component_node_groups = {}
        for node, data in G.nodes(data=True):
            if 'component' in data and node in pos:
                component = data['component']
                if component not in component_node_groups:
                    component_node_groups[component] = []
                component_node_groups[component].append(node)
        
        # Draw nodes by component with appropriate colors - increased size
        for component, nodes in component_node_groups.items():
            if component in component_colors:
                color = component_colors[component]
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, 
                                      node_size=self.node_size, alpha=0.8, ax=ax)  # Increased size
        
        # Draw any nodes without component info
        nodes_without_component = [n for n in G.nodes() if n in pos and 'component' not in G.nodes[n]]
        if nodes_without_component:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_without_component, 
                                  node_color='gray', node_size=self.node_size, alpha=0.8, ax=ax)  # Increased size
        
        # Add node labels
        nx.draw_networkx_labels(G, pos, font_size=self.node_font_size, font_color='black', ax=ax)
        
        # Remove axis
        ax.axis('off')
        
        # Convert the figure to an image - use a safer approach that works with different backends
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure canvas
        w, h = fig.canvas.get_width_height()
        
        # Try different methods to get the buffer depending on the canvas type
        try:
            # For Agg backend
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
        except AttributeError:
            try:
                # For Qt backend
                buf = fig.canvas.tostring_argb()
                img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                
                # Convert ARGB to RGBA
                img = np.roll(img, 3, axis=2)
            except AttributeError:
                # Fallback method
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf.reshape(h, w, 3)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img

    def _apply_transformations_to_positions(self, pos, transformations):
        """
        Apply 2D transformations to graph node positions.
        
        Parameters:
        -----------
        pos : dict
            Dictionary of node positions {node: (x, y)}
        transformations : list
            List of transformation operations from determine_2d_transformations()
            
        Returns:
        --------
        dict
            Transformed node positions
        """
        import numpy as np
        
        # Convert positions to numpy arrays
        pos_array = {node: np.array(position, dtype=float) for node, position in pos.items()}
        
        # Get the bounding box of the graph
        positions = np.array(list(pos_array.values()))
        min_x, min_y = positions.min(axis=0)
        max_x, max_y = positions.max(axis=0)
        
        # Calculate the center of the graph
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Apply each transformation in sequence
        for op, param in transformations:
            if op == 'rotate':
                # Convert angle to radians
                angle_rad = np.radians(param)
                cos_theta = np.cos(angle_rad)
                sin_theta = np.sin(angle_rad)
                
                # Rotate around the center
                for node in pos_array:
                    x, y = pos_array[node]
                    # Translate to origin
                    x_centered = x - center_x
                    y_centered = y - center_y
                    
                    # Rotate
                    x_rotated = x_centered * cos_theta - y_centered * sin_theta
                    y_rotated = x_centered * sin_theta + y_centered * cos_theta
                    
                    # Translate back
                    pos_array[node] = np.array([x_rotated + center_x, y_rotated + center_y])
                    
            elif op == 'flip_h':
                # Flip horizontally around the center
                for node in pos_array:
                    x, y = pos_array[node]
                    pos_array[node] = np.array([2 * center_x - x, y])
                    
            elif op == 'flip_v':
                # Flip vertically around the center
                for node in pos_array:
                    x, y = pos_array[node]
                    pos_array[node] = np.array([x, 2 * center_y - y])
                    
            elif op == 'transpose':
                # Swap x and y coordinates
                for node in pos_array:
                    x, y = pos_array[node]
                    pos_array[node] = np.array([y, x])
        
        return pos_array

    def _create_non_overlapping_layout(self, G):
        """
        Create a layout where components don't overlap but stay close together.
        
        Parameters:
        -----------
        G : networkx.Graph
            Graph to layout
        
        Returns:
        --------
        dict
            Node positions {node: (x, y)}
        """
        # Get connected components
        components = list(nx.connected_components(G))
        components = sorted(components, key=len, reverse=True)
        
        # Initial positions based on 3D coordinates
        initial_pos = {}
        for node, data in G.nodes(data=True):
            if 'coord' in data:
                z, y, x = data['coord']
                initial_pos[node] = np.array([x, y])  # Use x, y for 2D projection
        
        # Create a layout for each component
        component_layouts = {}
        for i, component in enumerate(components):
            # Extract the subgraph for this component
            subgraph = G.subgraph(component)
            
            # Get initial positions for this component
            component_pos = {n: initial_pos[n] for n in component if n in initial_pos}
            
            # If we don't have positions for all nodes, use spring layout
            if len(component_pos) < len(component):
                # Use spring layout with the initial positions as a starting point
                component_pos = nx.spring_layout(subgraph, pos=component_pos, seed=i+42)
            
            # Calculate the center of this component
            center = np.mean([pos for pos in component_pos.values()], axis=0)
            
            # Calculate the radius (maximum distance from center)
            radius = max([np.linalg.norm(pos - center) for pos in component_pos.values()]) if component_pos else 0
            
            # Store the layout, center, and radius
            component_layouts[i] = {
                'pos': component_pos,
                'center': center,
                'radius': radius,
                'component': component
            }
        
        # Adjust component positions to prevent overlap
        adjusted_layouts = self._adjust_component_positions(component_layouts)
        
        # Combine all layouts into one
        final_pos = {}
        for layout in adjusted_layouts.values():
            final_pos.update(layout['pos'])
        
        return final_pos

    def _adjust_component_positions(self, component_layouts):
        """
        Adjust component positions to prevent overlap while keeping them close.
        
        Parameters:
        -----------
        component_layouts : dict
            Dictionary of component layouts
        
        Returns:
        --------
        dict
            Adjusted component layouts
        """
        # If we only have one component, no adjustment needed
        if len(component_layouts) <= 1:
            return component_layouts
        
        # Create a deep copy of the layouts to adjust
        adjusted_layouts = {}
        for i, layout in component_layouts.items():
            adjusted_layouts[i] = {
                'pos': {node: np.array(pos, dtype=float) for node, pos in layout['pos'].items()},
                'center': np.array(layout['center'], dtype=float),
                'radius': float(layout['radius']),
                'component': layout['component']
            }
        
        # Define repulsion parameters
        repulsion_strength = 1.2  # Strength of repulsion (higher = more separation)
        min_distance = 0.1  # Minimum distance between components
        
        # Iteratively adjust positions to reduce overlap
        max_iterations = 50
        for iteration in range(max_iterations):
            # Calculate forces between components
            forces = {i: np.zeros(2, dtype=float) for i in adjusted_layouts}
            
            # Check all pairs of components
            for i in range(len(adjusted_layouts)):
                for j in range(i+1, len(adjusted_layouts)):
                    # Get the centers and radii
                    center_i = adjusted_layouts[i]['center']
                    center_j = adjusted_layouts[j]['center']
                    radius_i = adjusted_layouts[i]['radius']
                    radius_j = adjusted_layouts[j]['radius']
                    
                    # Calculate distance between centers
                    distance_vector = center_j - center_i
                    distance = np.linalg.norm(distance_vector)
                    
                    # Calculate the minimum distance needed to prevent overlap
                    min_required_distance = (radius_i + radius_j) * repulsion_strength + min_distance
                    
                    # If components are too close, apply repulsion
                    if distance < min_required_distance and distance > 0:
                        # Calculate repulsion force (inversely proportional to distance)
                        force_magnitude = (min_required_distance - distance) / min_required_distance
                        force_direction = distance_vector / distance  # Normalize
                        force = force_direction * force_magnitude
                        
                        # Apply forces in opposite directions
                        forces[i] -= force
                        forces[j] += force
            
            # Apply forces to adjust component positions
            max_force = 0
            for i, force in forces.items():
                force_magnitude = np.linalg.norm(force)
                max_force = max(max_force, force_magnitude)
                
                if force_magnitude > 0:
                    # Move the component center
                    adjusted_layouts[i]['center'] = adjusted_layouts[i]['center'] + force
                    
                    # Update all node positions in this component
                    for node in adjusted_layouts[i]['pos']:
                        adjusted_layouts[i]['pos'][node] = adjusted_layouts[i]['pos'][node] + force
            
            # If forces are small enough, we've reached equilibrium
            if max_force < 0.01:
                break
        
        return adjusted_layouts

    def get_or_create_global_id(self, coord):
        """Match GBL's node ID persistence logic"""
        rounded = tuple(round(c, 1) for c in coord)
        best_id = None
        best_distance = float('inf')
        
        for gid, existing in self.global_nodes.items():
            dist = np.linalg.norm(np.array(rounded) - np.array(existing))
            if dist < self.global_threshold and dist < best_distance:
                best_distance = dist
                best_id = gid
                
        if best_id is not None:
            return best_id
            
        new_id = self.next_global_node_id
        self.global_nodes[new_id] = rounded
        self.next_global_node_id += 1
        return new_id

    def capture_topological_graph_concentric(self, G, component_colors):
        """
        Capture a screenshot of the topological graph with concentric layout.
        Ensures consistent edge lengths across components and maximizes component sizes
        to fill available space while preventing overlap.
        Maintains consistent circle layout across timepoints.
        
        Parameters:
        -----------
        G : networkx.Graph
            Graph to visualize
        component_colors : dict
            Mapping of component IDs to colors
            
        Returns:
        --------
        ndarray
            Graph visualization image
        """
        if G is None or len(G.nodes) == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Get connected components
        components = list(nx.connected_components(G))
        components = sorted(components, key=len, reverse=True)
        
        # Store the number of components from the first frame for consistency
        if not hasattr(self, 'initial_num_components') or self.frame_idx == 0:
            self.initial_num_components = len(components)
            self.initial_components_per_ring = 8  # Maximum components per ring
            
            # Calculate the actual number of rings needed for the first frame
            # The first component is placed in the center, so we only need rings for the remaining components
            remaining_components = len(components) - 1  # Subtract 1 for the center component
            
            # Initialize with 1 ring
            self.initial_num_rings = 1
            components_in_first_ring = min(self.initial_components_per_ring, remaining_components)
            remaining_after_first = remaining_components - components_in_first_ring
            
            # If we still have components left, calculate additional rings needed
            if remaining_after_first > 0:
                # For each additional ring, we can fit more components
                ring_idx = 1
                while remaining_after_first > 0:
                    components_in_this_ring = min(self.initial_components_per_ring + ring_idx * 2, remaining_after_first)
                    remaining_after_first -= components_in_this_ring
                    if remaining_after_first > 0:
                        self.initial_num_rings += 1
                    ring_idx += 1
            
            self.initial_ring_radius = 0.5  # Start with a reasonable distance from center
            self.initial_ring_spacing = 0.7  # Increase radius for next ring
            print(f"Initial frame has {self.initial_num_components} components, using {self.initial_num_rings} rings")
        
        # Calculate target edge length based on the largest component
        target_edge_length = 0.1  # Default value
        
        if components and len(components[0]) > 1:
            largest_component = G.subgraph(components[0])
            # Calculate average edge length in the largest component
            if largest_component.number_of_edges() > 0:
                # Create a temporary layout to measure edge lengths
                temp_layout = nx.spring_layout(largest_component, seed=42)
                edge_lengths = []
                for u, v in largest_component.edges():
                    if u in temp_layout and v in temp_layout:
                        edge_lengths.append(np.linalg.norm(np.array(temp_layout[u]) - np.array(temp_layout[v])))
                if edge_lengths:
                    target_edge_length = np.mean(edge_lengths)
        
        # Create a concentric layout with consistent edge lengths
        pos = {}
        component_layouts = []
        
        # Place the largest component in the center with normalized edge lengths
        if components:
            center_component = components[0]
            center_subgraph = G.subgraph(center_component)
            
            # Use spring layout for the center component
            center_pos = nx.spring_layout(center_subgraph, seed=42)
            
            # Calculate the center and radius of this component
            center_coords = np.array([center_pos[node] for node in center_pos])
            center = np.mean(center_coords, axis=0) if len(center_coords) > 0 else np.array([0, 0])
            radius = max([np.linalg.norm(pos - center) for pos in center_coords]) if len(center_coords) > 0 else 0.1
            
            # Store the component layout
            component_layouts.append({
                'component': center_component,
                'pos': center_pos,
                'center': center,
                'radius': radius,
                'scale': 1.0  # Initial scale factor
            })
        
        # Place other components in concentric rings with consistent structure
        if len(components) > 1:
            # Use the number of rings and components per ring from the first frame
            ring_radius = self.initial_ring_radius if hasattr(self, 'initial_ring_radius') else 0.5
            components_per_ring = self.initial_components_per_ring if hasattr(self, 'initial_components_per_ring') else 8
            num_rings = self.initial_num_rings if hasattr(self, 'initial_num_rings') else 1
            ring_spacing = self.initial_ring_spacing if hasattr(self, 'initial_ring_spacing') else 0.7
            
            # debug print these values
            print(f"Using {num_rings} rings with {components_per_ring} components per ring")
            
            remaining_components = components[1:]  # Skip the center component
            component_idx = 0
            
            # Strictly enforce the number of rings
            for ring_idx in range(num_rings):
                # Calculate ring radius
                current_radius = ring_radius + ring_idx * ring_spacing
                
                # Calculate how many components to place on this ring
                # If this is the last ring, place all remaining components
                if ring_idx == num_rings - 1:
                    current_ring_capacity = len(remaining_components) - component_idx
                else:
                    current_ring_capacity = min(components_per_ring + ring_idx * 2, 
                                               len(remaining_components) - component_idx)
                
                if current_ring_capacity <= 0:
                    break  # No more components to place
                
                print(f"Ring {ring_idx+1}: placing {current_ring_capacity} components")
                
                # Place components evenly around the ring
                for i in range(current_ring_capacity):
                    if component_idx >= len(remaining_components):
                        break
                        
                    component = remaining_components[component_idx]
                    
                    # Calculate angle for this component
                    angle = 2 * np.pi * i / current_ring_capacity
                    
                    # Calculate component center
                    center_x = current_radius * np.cos(angle)
                    center_y = current_radius * np.sin(angle)
                    
                    # Create a mini-layout for this component
                    component_subgraph = G.subgraph(component)
                    component_pos = nx.spring_layout(component_subgraph, seed=component_idx+42)
                    
                    # Calculate the center and radius of this component
                    comp_coords = np.array([component_pos[node] for node in component_pos])
                    comp_center = np.mean(comp_coords, axis=0) if len(comp_coords) > 0 else np.array([0, 0])
                    comp_radius = max([np.linalg.norm(np.array(pos) - comp_center) for pos in comp_coords]) if len(comp_coords) > 0 else 0.1
                    
                    # Store the component layout
                    component_layouts.append({
                        'component': component,
                        'pos': component_pos,
                        'center': comp_center,
                        'radius': comp_radius,
                        'base_pos': np.array([center_x, center_y]),
                        'scale': 1.0  # Initial scale factor
                    })
                    
                    component_idx += 1
        
        # STEP 1: Normalize edge lengths across components
        for i, layout in enumerate(component_layouts):
            component = layout['component']
            component_subgraph = G.subgraph(component)
            
            if component_subgraph.number_of_edges() > 0:
                # Calculate average edge length in this component
                edge_lengths = []
                for u, v in component_subgraph.edges():
                    if u in layout['pos'] and v in layout['pos']:
                        edge_lengths.append(np.linalg.norm(np.array(layout['pos'][u]) - np.array(layout['pos'][v])))
                
                if edge_lengths:
                    avg_edge_length = np.mean(edge_lengths)
                    # Scale factor to normalize edge lengths
                    scale_factor = target_edge_length / max(avg_edge_length, 0.001)
                    layout['scale'] = scale_factor
        
        # STEP 2: Position components with normalized edge lengths
        for i, layout in enumerate(component_layouts):
            if i == 0:  # Center component
                # Scale the center component
                center_scale = layout['scale'] * 0.4  # Base scale
                for node, coords in layout['pos'].items():
                    pos[node] = coords * center_scale
            else:
                # Position and scale other components
                base_pos = layout['base_pos']
                component_scale = layout['scale'] * 0.2  # Base scale for other components
                
                for node, coords in layout['pos'].items():
                    pos[node] = base_pos + coords * component_scale
        
        # STEP 3: Maximize overall scale until components would start to overlap
        # Calculate the current positions and sizes of all components
        component_info = []
        for i, layout in enumerate(component_layouts):
            component = layout['component']
            
            # Get positions for this component
            component_pos = [pos[node] for node in component if node in pos]
            if not component_pos:
                continue
            
            # Calculate center and radius
            component_pos_array = np.array(component_pos)
            center = np.mean(component_pos_array, axis=0)
            
            # Calculate radius as maximum distance from center to any node
            radius = max([np.linalg.norm(p - center) for p in component_pos]) + 0.05  # Small buffer
            
            component_info.append({
                'index': i,
                'component': component,
                'center': center,
                'radius': radius
            })
        
        # Find the maximum scale factor that prevents overlap
        max_scale = 1.0
        scale_increment = 0.1
        max_iterations = 20
        
        for _ in range(max_iterations):
            # Try increasing the scale
            test_scale = max_scale + scale_increment
            
            # Check if this scale would cause overlap
            overlap = False
            
            # Scale all component positions and check for overlap
            scaled_centers = [info['center'] * test_scale for info in component_info]
            scaled_radii = [info['radius'] * test_scale for info in component_info]
            
            # Check all pairs of components for overlap
            for i in range(len(component_info)):
                for j in range(i+1, len(component_info)):
                    center_i = scaled_centers[i]
                    center_j = scaled_centers[j]
                    radius_i = scaled_radii[i]
                    radius_j = scaled_radii[j]
                    
                    # Calculate distance between centers
                    distance = np.linalg.norm(center_j - center_i)
                    
                    # Check if components would overlap
                    if distance < (radius_i + radius_j):
                        overlap = True
                        break
            
            if not overlap:
                max_scale = test_scale
            else:
                # We found an overlap, so stop increasing the scale
                break
        
        # Apply the maximum scale to all positions
        scaled_pos = {node: np.array(position) * max_scale for node, position in pos.items()}
        
        # STEP 4: Apply overlap prevention for final adjustments
        final_pos = scaled_pos # self._simple_overlap_prevention(G, scaled_pos, components)
        
        # Store the axis limits from the first frame for consistency
        if not hasattr(self, 'initial_axis_limits') or self.frame_idx == 0:
            positions = np.array(list(final_pos.values()))
            min_x, min_y = positions.min(axis=0) - 0.1
            max_x, max_y = positions.max(axis=0) + 0.1
            
            # Ensure the plot is square
            width = max(max_x - min_x, max_y - min_y)
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            
            self.initial_axis_limits = {
                'x_min': center_x - width/2,
                'x_max': center_x + width/2,
                'y_min': center_y - width/2,
                'y_max': center_y + width/2
            }
        
        # Group edges by component for coloring
        component_edge_groups = {}
        for u, v, data in G.edges(data=True):
            if 'component' in data and u in final_pos and v in final_pos:
                component = data['component']
                if component not in component_edge_groups:
                    component_edge_groups[component] = []
                component_edge_groups[component].append((u, v))
        
        # Draw edges by component with appropriate colors
        for component, edges in component_edge_groups.items():
            if component in component_colors:
                color = component_colors[component]
                nx.draw_networkx_edges(G, final_pos, ax=ax, edgelist=edges, width=self.edge_width, alpha=0.8, edge_color=color)
        
        # Group nodes by component for coloring
        component_node_groups = {}
        for node, data in G.nodes(data=True):
            if 'component' in data and node in final_pos:
                component = data['component']
                if component not in component_node_groups:
                    component_node_groups[component] = []
                component_node_groups[component].append(node)
        
        # Draw nodes by component with appropriate colors
        for component, nodes in component_node_groups.items():
            if component in component_colors:
                color = component_colors[component]
                nx.draw_networkx_nodes(G, final_pos, nodelist=nodes, node_color=color, 
                                      node_size=(self.node_size-500), alpha=0.8, ax=ax)
        
        # Draw any nodes without component info
        nodes_without_component = [n for n in G.nodes() if n in final_pos and 'component' not in G.nodes[n]]
        if nodes_without_component:
            nx.draw_networkx_nodes(G, final_pos, nodelist=nodes_without_component, 
                                  node_color='gray', node_size=(self.node_size-500), alpha=0.8, ax=ax)
        
        # Add node labels
        nx.draw_networkx_labels(G, final_pos, font_size=self.node_font_size, font_color='black', ax=ax)
        
        # Remove axis and set tight layout
        ax.axis('off')
        plt.tight_layout(pad=0)
        
        # Use consistent axis limits across frames
        if hasattr(self, 'initial_axis_limits'):
            ax.set_xlim(self.initial_axis_limits['x_min'], self.initial_axis_limits['x_max'])
            ax.set_ylim(self.initial_axis_limits['y_min'], self.initial_axis_limits['y_max'])
        else:
            # Fallback if initial limits aren't set
            if final_pos:
                positions = np.array(list(final_pos.values()))
                min_x, min_y = positions.min(axis=0) - 0.1
                max_x, max_y = positions.max(axis=0) + 0.1
                
                # Ensure the plot is square
                width = max(max_x - min_x, max_y - min_y)
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                
                ax.set_xlim(center_x - width/2, center_x + width/2)
                ax.set_ylim(center_y - width/2, center_y + width/2)
            else:
                ax.set_xlim(-1.0, 1.0)
                ax.set_ylim(-1.0, 1.0)
        
        # Convert the figure to an image
        fig.canvas.draw()
        
        # Get the RGBA buffer from the figure canvas
        w, h = fig.canvas.get_width_height()
        
        # Try different methods to get the buffer depending on the canvas type
        try:
            # For Agg backend
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
        except AttributeError:
            try:
                # For Qt backend
                buf = fig.canvas.tostring_argb()
                img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                
                # Convert ARGB to RGBA
                img = np.roll(img, 3, axis=2)
            except AttributeError:
                # Fallback method
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = buf.reshape(h, w, 3)
        
        # Close the figure to free memory
        plt.close(fig)
        
        return img

    def _simple_overlap_prevention(self, G, pos, components):
        """
        Simple algorithm to prevent component overlap by adjusting component positions.
        
        Parameters:
        -----------
        G : networkx.Graph
            Graph to adjust
        pos : dict
            Dictionary of node positions {node: (x, y)}
        components : list
            List of connected components
            
        Returns:
        --------
        dict
            Adjusted node positions
        """
        # Convert positions to numpy arrays
        pos_array = {node: np.array(position, dtype=float) for node, position in pos.items()}
        
        # Calculate component centers and bounding boxes
        component_info = []
        for i, component in enumerate(components):
            if not component:
                continue
            
            # Get positions for this component
            component_pos = [pos_array[node] for node in component if node in pos_array]
            if not component_pos:
                continue
            
            # Calculate center and radius
            component_pos_array = np.array(component_pos)
            center = np.mean(component_pos_array, axis=0)
            
            # Calculate radius as maximum distance from center to any node
            radius = max([np.linalg.norm(pos - center) for pos in component_pos]) + 0.1
            
            component_info.append({
                'index': i,
                'component': component,
                'center': center,
                'radius': radius
            })
        
        # Check for overlaps and adjust positions
        max_iterations = 20
        for _ in range(max_iterations):
            overlap_found = False
            
            # Check all pairs of components for overlap
            for i in range(len(component_info)):
                for j in range(i+1, len(component_info)):
                    comp_i = component_info[i]
                    comp_j = component_info[j]
                    
                    # Calculate distance between centers
                    center_i = comp_i['center']
                    center_j = comp_j['center']
                    distance = np.linalg.norm(center_j - center_i)
                    
                    # Calculate minimum required distance
                    min_distance = comp_i['radius'] + comp_j['radius']
                    
                    # If components overlap, move them apart
                    if distance < min_distance:
                        overlap_found = True
                        
                        # Calculate direction vector
                        direction = (center_j - center_i) / max(distance, 0.001)
                        
                        # Calculate how much to move each component
                        move_distance = (min_distance - distance) / 2
                        
                        # Move components in opposite directions
                        move_i = -direction * move_distance
                        move_j = direction * move_distance
                        
                        # Update component centers
                        comp_i['center'] += move_i
                        comp_j['center'] += move_j
                        
                        # Update node positions for each component
                        for node in comp_i['component']:
                            if node in pos_array:
                                pos_array[node] += move_i
                        
                        for node in comp_j['component']:
                            if node in pos_array:
                                pos_array[node] += move_j
            
            # If no overlaps found, we're done
            if not overlap_found:
                break
        
        return pos_array

    