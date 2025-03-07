



    # def layout_by_projected_centroid(self, G):
    #     """Project 3D coordinates to 2D using captured viewpoint"""
    #     pos = {}
    #     for node in G.nodes():
    #         coord = G.nodes[node]['coord']
    #         # Transform using captured view parameters
    #         pos[node] = self._transform_3d_to_2d(coord)
    #     return self._adjust_layout(pos)



    # def layout_by_concentric_circles(self, G, ideal_edge_length=1.0, base_distance=3.0):
    #     """
    #     Compute a layout that arranges disconnected components in concentric rings.
        
    #     Each connected component is laid out using a spring layout and then scaled so that
    #     the average edge length in that component equals ideal_edge_length.
    #     The largest component is placed at the center and smaller ones are arranged in
    #     concentric rings outward.
        
    #     Parameters:
    #     -----------
    #     G : networkx.Graph
    #         The graph to layout
    #     ideal_edge_length : float, optional
    #         Target edge length for each component (default: 1.0)
    #     base_distance : float, optional
    #         Base distance between concentric rings (default: 3.0)
            
    #     Returns:
    #     --------
    #     dict
    #         Dictionary mapping node IDs to (x, y) positions
    #     """
    #     import math

    #     components = list(nx.connected_components(G))
    #     components = sorted(components, key=lambda comp: len(comp), reverse=True)
    #     overall_pos = {}

    #     def scale_layout(subG, pos, ideal_length):
    #         if subG.number_of_nodes() <= 1:
    #             return pos
    #         edges = list(subG.edges())
    #         lengths = [np.linalg.norm(pos[u] - pos[v]) for u, v in edges]
    #         avg_length = np.mean(lengths) if lengths else 1.0
    #         scale = ideal_length / avg_length
    #         for node in pos:
    #             pos[node] = pos[node] * scale
    #         return pos

    #     # Handle empty graph case
    #     if len(components) == 0:
    #         return {}
        
    #     # Place largest component in center
    #     central_comp = components[0]
    #     central_subG = G.subgraph(central_comp)
    #     central_pos = nx.spring_layout(central_subG, k=ideal_edge_length, seed=42)
    #     central_pos = scale_layout(central_subG, central_pos, ideal_edge_length)
    #     for node, pos in central_pos.items():
    #         overall_pos[node] = pos

    #     # Arrange remaining components in concentric rings
    #     remaining = components[1:]
    #     n = len(remaining)
    #     if n > 0:
    #         comps_per_ring = math.ceil(math.sqrt(n))
    #         for idx, comp in enumerate(remaining, start=1):
    #             ring_index = math.ceil(idx / comps_per_ring)
    #             pos_in_ring = (idx - 1) % comps_per_ring
    #             angle = 2 * math.pi * pos_in_ring / comps_per_ring
    #             offset = np.array([
    #                 base_distance * ring_index * math.cos(angle),
    #                 base_distance * ring_index * math.sin(angle)
    #             ])
    #             subG = G.subgraph(comp)
    #             subpos = nx.spring_layout(subG, k=ideal_edge_length, seed=42)
    #             subpos = scale_layout(subG, subpos, ideal_edge_length)
    #             for node, pos in subpos.items():
    #                 overall_pos[node] = pos + offset
                
    #     # Adjust layout to fit viewport
    #     return self._adjust_layout(overall_pos)

    # def layout_disconnected_components(self, G, ideal_edge_length=1.0, base_distance=2.0):
    #     """
    #     Compute a layout that arranges disconnected components in concentric rings.
    #     Each connected component is laid out using a spring layout and then scaled so that
    #     the average edge length in that component equals ideal_edge_length.
    #     The largest component is placed at the center and smaller ones are arranged in
    #     concentric rings outward, minimizing white space.
        
    #     Parameters:
    #     -----------
    #     G : networkx.Graph
    #         Graph to layout
    #     ideal_edge_length : float, optional
    #         Ideal length for edges within components (default: 1.0)
    #     base_distance : float, optional
    #         Base distance between components (default: 2.0)
            
    #     Returns:
    #     --------
    #     dict
    #         Mapping of nodes to positions {node: (x, y)}
    #     """
    #     import math

    #     components = list(nx.connected_components(G))
    #     components = sorted(components, key=lambda comp: len(comp), reverse=True)
    #     overall_pos = {}

    #     def scale_layout(subG, pos, ideal_length):
    #         """Scale the layout to have the ideal average edge length"""
    #         if subG.number_of_nodes() <= 1:
    #             return pos
    #         edges = list(subG.edges())
    #         lengths = [np.linalg.norm(pos[u] - pos[v]) for u, v in edges]
    #         avg_length = np.mean(lengths) if lengths else 1.0
    #         scale = ideal_length / avg_length
    #         for node in pos:
    #             pos[node] = pos[node] * scale
    #         return pos
        
    #     def get_component_radius(subG, pos):
    #         """Get the radius of a component based on its layout"""
    #         if not pos:
    #             return 0
    #         # Calculate the center of the component
    #         center = np.mean([pos[n] for n in pos], axis=0)
    #         # Calculate the maximum distance from the center
    #         max_dist = max([np.linalg.norm(pos[n] - center) for n in pos]) if pos else 0
    #         return max_dist
        
    #     # If there are no components, return empty layout
    #     if not components:
    #         return {}
        
    #     # Layout the central (largest) component
    #     central_comp = components[0]
    #     central_subG = G.subgraph(central_comp)
    #     central_pos = nx.spring_layout(central_subG, k=ideal_edge_length, seed=42)
    #     central_pos = scale_layout(central_subG, central_pos, ideal_edge_length)
        
    #     # Get the radius of the central component
    #     central_radius = get_component_radius(central_subG, central_pos)
        
    #     # Add central component to overall layout
    #     for node, pos in central_pos.items():
    #         overall_pos[node] = pos

    #     # Layout remaining components in concentric rings with adaptive spacing
    #     remaining = components[1:]
    #     if remaining:
    #         # Sort remaining components by size (largest first)
    #         remaining = sorted(remaining, key=len, reverse=True)
            
    #         # Calculate how many components to place in each ring
    #         # Use a more adaptive approach based on component sizes
    #         ring_assignments = []  # [(ring_index, angle, component), ...]
    #         current_ring = 1
    #         angle_step = 2 * math.pi / 8  # Start with 8 positions in first ring
    #         current_angle = 0
            
    #         for comp in remaining:
    #             # Create a subgraph for this component
    #             subG = G.subgraph(comp)
    #             subpos = nx.spring_layout(subG, k=ideal_edge_length, seed=42)
    #             subpos = scale_layout(subG, subpos, ideal_edge_length)
                
    #             # Get the radius of this component
    #             comp_radius = get_component_radius(subG, subpos)
                
    #             # Assign to current ring and angle
    #             ring_assignments.append((current_ring, current_angle, comp, subpos, comp_radius))
                
    #             # Update angle for next component
    #             current_angle += angle_step
                
    #             # If we've gone all the way around, move to next ring
    #             if current_angle >= 2 * math.pi:
    #                 current_ring += 1
    #                 # Increase number of positions in next ring
    #                 angle_step = 2 * math.pi / (8 * current_ring)
    #                 current_angle = 0
            
    #         # Now place components based on assignments, adjusting distances to minimize white space
    #         for ring_idx, angle, comp, subpos, comp_radius in ring_assignments:
    #             # Calculate distance from center based on ring index and component sizes
    #             # Use a more compact layout by considering component sizes
    #             ring_distance = central_radius + comp_radius + base_distance * ring_idx
                
    #             # Calculate offset based on angle and distance
    #             offset = np.array([
    #                 ring_distance * math.cos(angle),
    #                 ring_distance * math.sin(angle)
    #             ])
                
    #             # Add component with offset
    #             for node, pos in subpos.items():
    #                 overall_pos[node] = pos + offset
        
    #     return overall_pos


   # def create_projected_graph_image(self, viewpoint_selector=None, node_labels=True, edge_labels=False, figsize=(10, 10)):
    #     """
    #     Create a projected graph visualization that matches the user's selected viewpoint.
        
    #     Parameters:
    #     -----------
    #     viewpoint_selector : ViewpointSelector, optional
    #         ViewpointSelector instance to get transformations from (default: None)
    #     node_labels : bool, optional
    #         Whether to show node labels (default: True)
    #     edge_labels : bool, optional
    #         Whether to show edge labels (default: False)
    #     figsize : tuple, optional
    #         Figure size (default: (10, 10))
            
    #     Returns:
    #     --------
    #     ndarray
    #         Graph visualization image
    #     """
    #     if self.G is None or len(self.G.nodes) == 0:
    #         return np.zeros((100, 100, 3), dtype=np.uint8)
        
    #     # Create a new figure
    #     fig, ax = plt.subplots(figsize=figsize)
        
    #     # Get node positions (use 2D projection of 3D coordinates)
    #     pos = {}
    #     for node, data in self.G.nodes(data=True):
    #         if 'pos' in data:
    #             z, y, x = data['pos']
    #             pos[node] = (x, y)  # Use x, y for 2D projection
        
    #     # Draw the graph
    #     nx.draw_networkx_edges(self.G, pos, ax=ax, width=self.edge_width, alpha=0.7)
        
    #     # Draw nodes with different colors based on type
    #     terminal_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'terminal']
    #     junction_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'junction']
        
    #     nx.draw_networkx_nodes(self.G, pos, nodelist=terminal_nodes, 
    #                           node_color='blue', node_size=800, alpha=0.8, ax=ax)
    #     nx.draw_networkx_nodes(self.G, pos, nodelist=junction_nodes, 
    #                           node_color='red', node_size=800, alpha=0.8, ax=ax)
        
    #     # Add node labels if requested
    #     if node_labels:
    #         nx.draw_networkx_labels(self.G, pos, font_size=50, font_color='black', ax=ax)
        
    #     # Add edge labels if requested
    #     if edge_labels and nx.get_edge_attributes(self.G, 'weight'):
    #         edge_labels = nx.get_edge_attributes(self.G, 'weight')
    #         nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=ax)
        
    #     # Remove axis
    #     ax.axis('off')
        
    #     # Convert the figure to an image - use a safer approach that works with different backends
    #     fig.canvas.draw()
        
    #     # Get the RGBA buffer from the figure canvas
    #     w, h = fig.canvas.get_width_height()
        
    #     # Try different methods to get the buffer depending on the canvas type
    #     try:
    #         # For Agg backend
    #         buf = fig.canvas.buffer_rgba()
    #         img = np.asarray(buf)
    #     except AttributeError:
    #         try:
    #             # For Qt backend
    #             buf = fig.canvas.tostring_argb()
    #             img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                
    #             # Convert ARGB to RGBA
    #             img = np.roll(img, 3, axis=2)
    #         except AttributeError:
    #             # Fallback method
    #             buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             img = buf.reshape(h, w, 3)
        
    #     # Close the figure to free memory
    #     plt.close(fig)
        
    #     # Apply transformations from viewpoint if available
    #     if viewpoint_selector is not None and viewpoint_selector.has_view:
    #         # Get the transformations needed to match the 3D viewpoint
    #         transformations = viewpoint_selector.determine_2d_transformations()
            
    #         # Apply the transformations to the image
    #         if transformations:
    #             print("Applying transformations to projected graph image...")
    #             img = viewpoint_selector.apply_transformations_to_image(img, transformations)
        
    #     return img

    # def create_concentric_graph_image(self, node_labels=True, edge_labels=False, figsize=(10, 10)):
    #     """
    #     Create a concentric graph visualization.
        
    #     Parameters:
    #     -----------
    #     node_labels : bool, optional
    #         Whether to show node labels (default: True)
    #     edge_labels : bool, optional
    #         Whether to show edge labels (default: False)
    #     figsize : tuple, optional
    #         Figure size (default: (10, 10))
            
    #     Returns:
    #     --------
    #     ndarray
    #         Graph visualization image
    #     """
    #     if self.G is None or len(self.G.nodes) == 0:
    #         return np.zeros((100, 100, 3), dtype=np.uint8)
        
    #     # Create a new figure
    #     fig, ax = plt.subplots(figsize=figsize)
        
    #     # Create a concentric layout
    #     pos = nx.shell_layout(self.G)
        
    #     # Draw the graph
    #     nx.draw_networkx_edges(self.G, pos, ax=ax, width=self.edge_width, alpha=0.7)
        
    #     # Draw nodes with different colors based on type
    #     terminal_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'terminal']
    #     junction_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'junction']
        
    #     nx.draw_networkx_nodes(self.G, pos, nodelist=terminal_nodes, 
    #                           node_color='blue', node_size=self.node_size, alpha=0.8, ax=ax)
    #     nx.draw_networkx_nodes(self.G, pos, nodelist=junction_nodes, 
    #                           node_color='red', node_size=self.node_size, alpha=0.8, ax=ax)
        
    #     # Add node labels if requested
    #     if node_labels:
    #         nx.draw_networkx_labels(self.G, pos, font_size=self.node_font_size, font_color='black', ax=ax)
        
    #     # Add edge labels if requested
    #     if edge_labels and nx.get_edge_attributes(self.G, 'weight'):
    #         edge_labels = nx.get_edge_attributes(self.G, 'weight')
    #         nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, ax=ax)
        
    #     # Remove axis
    #     ax.axis('off')
        
    #     # Convert the figure to an image - use a safer approach that works with different backends
    #     fig.canvas.draw()
        
    #     # Get the RGBA buffer from the figure canvas
    #     w, h = fig.canvas.get_width_height()
        
    #     # Try different methods to get the buffer depending on the canvas type
    #     try:
    #         # For Agg backend
    #         buf = fig.canvas.buffer_rgba()
    #         img = np.asarray(buf)
    #     except AttributeError:
    #         try:
    #             # For Qt backend
    #             buf = fig.canvas.tostring_argb()
    #             img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                
    #             # Convert ARGB to RGBA
    #             img = np.roll(img, 3, axis=2)
    #         except AttributeError:
    #             # Fallback method
    #             buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #             img = buf.reshape(h, w, 3)
        
    #     # Close the figure to free memory
    #     plt.close(fig)
        
    #     return img

    # def update_graph(self, G, frame_idx=None):
    #     """
    #     Update the internal graph with a new graph.
        
    #     Parameters:
    #     -----------
    #     G : networkx.Graph
    #         The new graph to store
    #     frame_idx : int, optional
    #         Frame index for tracking nodes across timepoints
    #     """
    #     self.G = G
        
    #     # Store the frame index if provided
    #     if frame_idx is not None:
    #         self.frame_idx = frame_idx
        
    #     # If we have a node tracker manager, update it with the new graph
    #     if hasattr(self, 'node_tracker_manager') and self.node_tracker_manager is not None:
    #         self.node_tracker_manager.update_node_positions(G, frame_idx)
        
    #     print(f"Updated graph with {len(G.nodes())} nodes and {len(G.edges())} edges")

    # def get_node_labels_and_coordinates(self):
    #     """
    #     Get the node labels and their 3D coordinates from the graph.
        
    #     Returns:
    #     --------
    #     dict
    #         Dictionary mapping node IDs to their coordinates {node_id: (z, y, x)}
    #     """
    #     if self.G is None:
    #         return {}
        
    #     node_coords = {}
    #     for node, data in self.G.nodes(data=True):
    #         if 'coord' in data:
    #             # Use the global ID as the node label
    #             global_id = self.get_or_create_global_id(data['coord'])
    #             node_coords[global_id] = data['coord']
        
    #     return node_coords

# Add other graph-related methods as needed 



def _transform_3d_to_2d(self, coord):
    """Convert 3D coordinate to 2D projection based on captured viewpoint"""
    # Get camera parameters from captured view
    cam_azim, cam_elev, cam_roll = np.deg2rad(self.angles)
    
    # Convert coordinate to numpy array and center
    coord = np.array(coord) - self.center
    
    # Rotation matrices
    R_z = np.array([[np.cos(cam_azim), -np.sin(cam_azim), 0],
                    [np.sin(cam_azim), np.cos(cam_azim), 0],
                    [0, 0, 1]])
    
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(cam_elev), -np.sin(cam_elev)],
                    [0, np.sin(cam_elev), np.cos(cam_elev)]])
    
    # Apply rotations
    rotated = R_z @ R_x @ coord
    
    # Orthographic projection (ignore Z-axis)
    x_proj = rotated[0] * self.zoom_level
    y_proj = rotated[1] * self.zoom_level
    
    # Adjust for viewport size
    if self.viewport_size:
        x_proj += self.viewport_size[0] / 2
        y_proj += self.viewport_size[1] / 2
    
    return (x_proj, y_proj)

def _adjust_layout(self, pos):
    """Maintain original spatial relationships while fitting to viewport"""
    positions = np.array(list(pos.values()))
    if len(positions) == 0:
        return pos

    # 1. Apply the same view transformation as the node-edge skeleton
    min_coords = positions.min(axis=0)
    max_coords = positions.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1  # Prevent division by zero
    
    # 2. Normalize while maintaining aspect ratio from node-edge view
    normalized = (positions - min_coords) / range_coords
    
    # 3. Scale to match the node-edge view dimensions
    if self.viewport_size:
        # Use actual viewport size from captured view
        scaled_x = normalized[:,0] * self.viewport_size[0] * 0.9  # 90% of width
        scaled_y = normalized[:,1] * self.viewport_size[1] * 0.9  # 90% of height
    else:
        # Fallback scaling
        scaled_x = normalized[:,0] * 1000
        scaled_y = normalized[:,1] * 1000

    # 4. Center the layout while preserving relative positions
    center_offset = np.array([scaled_x.min() + (scaled_x.ptp()/2), 
                            scaled_y.min() + (scaled_y.ptp()/2)])
    scaled_x -= center_offset[0]
    scaled_y -= center_offset[1]

    # 5. Flip Y-axis to match napari's coordinate system
    # scaled_y *= -1

    return {node: (scaled_x[i], scaled_y[i]) for i, node in enumerate(pos.keys())}