import networkx as nx
import numpy as np
from skimage import measure
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class TopologicalGraph:
    def __init__(self, captured_view, viewport_size, zoom_level):
        self.component_edge_pixels = defaultdict(list)
        self.current_component_id = 1000
        self.prev_components = []
        self.color_index = 0
        self.G = nx.Graph()
        self.global_nodes = {}  # {global_id: (z,y,x)}
        self.next_global_node_id = 1
        self.global_threshold = 2.0  # Pixels

        # capture view parameters
        self.captured_view = captured_view
        self.angles, self.center = self.captured_view
        self.zoom_level = zoom_level
        self.viewport_size = viewport_size

    def build_graph_for_frame(self, pixel_class_vol):
        """Enhanced junction merging with expanded connectivity and distance checks"""
        junction_mask = (pixel_class_vol == 4)
        
        # Use full 3D connectivity (26-connectivity) for initial labeling
        junction_labels = measure.label(junction_mask, connectivity=2)
        regions = measure.regionprops(junction_labels)
        
        # Merge regions within 2 pixels of each other
        merged_labels = np.zeros_like(junction_labels)
        current_label = 1
        processed = set()

        for i, r1 in enumerate(regions):
            if i in processed:
                continue
            # Find nearby regions using centroid distances
            centroid1 = np.array(r1.centroid)
            to_merge = [i]
            
            for j, r2 in enumerate(regions[i+1:], start=i+1):
                centroid2 = np.array(r2.centroid)
                if np.linalg.norm(centroid1 - centroid2) <= 2:  # Within 2 pixels
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
        node_counter = 0
        junction_mapping = {}
        
        # Add junctions with global IDs
        for prop in junction_props:
            coord = tuple(map(int, np.round(prop.centroid)))
            global_id = self.get_or_create_global_id(coord)
            G.add_node(global_id, coord=coord, type='junction')
        
        # Add terminals with global IDs
        terminal_coords = [tuple(coord) for coord in np.argwhere(pixel_class_vol == 2)]
        for coord in terminal_coords:
            global_id = self.get_or_create_global_id(coord)
            G.add_node(global_id, coord=coord, type='terminal')

        

        visited = np.zeros_like(pixel_class_vol, dtype=bool)
        all_node_coords = [G.nodes[node]['coord'] for node in G.nodes()]
        component_id = 1
        for start_coord in all_node_coords:
            start_id = self._get_node_id(start_coord, G, pixel_class_vol)
            if start_id is None:
                continue
            
            for nb in self.get_neighbors_3d(start_coord, pixel_class_vol.shape):
                if pixel_class_vol[nb] == 3 and not visited[nb]:
                    path = [start_coord, nb]
                    visited[nb] = True
                    current = nb
                    reached_id = None
                    
                    while True:
                        candidates = [n for n in self.get_neighbors_3d(current, pixel_class_vol.shape) 
                                    if n not in path and pixel_class_vol[n] in (2,3,4)]
                        
                        if not candidates:
                            break
                            
                        next_pixel = None
                        for candidate in candidates:
                            if pixel_class_vol[candidate] in (2,4):  # Found another node
                                next_pixel = candidate
                                break
                            elif pixel_class_vol[candidate] == 3 and not visited[candidate]:
                                next_pixel = candidate
                                break
                            
                        if next_pixel is None:
                            break
                            
                        path.append(next_pixel)
                        visited[next_pixel] = True
                        current = next_pixel
                        
                        # Check if reached another node
                        node_id = self._get_node_id(current, G, pixel_class_vol)
                        if node_id is not None and node_id != start_id:
                            reached_id = node_id
                            break

                    if reached_id is not None and not G.has_edge(start_id, reached_id):
                        G.add_edge(start_id, reached_id, 
                                   weight=len(path), 
                                   path=path,
                                   component_id=component_id)  # Store component ID on edges
                        self.component_edge_pixels[component_id].extend(path)
                        component_id += 1


        return G, self.component_edge_pixels
    

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
        """Match GBL's temporal color consistency"""
        current_components = [frozenset(cc) for cc in nx.connected_components(G)]
        
        current_globals = [frozenset(self.get_or_create_global_id(G.nodes[n]['coord']) 
                            for n in comp) for comp in current_components]
        
        # Calculate component centroids
        component_features = []
        for comp in current_components:
            nodes = [G.nodes[n]['coord'] for n in comp]  # comp is now frozenset
            centroid = np.mean(nodes, axis=0)
            size = len(nodes)
            component_features.append((comp, centroid, size))

        cmap = plt.get_cmap("tab20")
        color_map = {}
        
        for comp, centroid, size in component_features:
            best_match = None
            best_score = 0
            
            if self.prev_components:
                for prev_comp, prev_centroid, prev_size, prev_color in self.prev_components:
                    # Use proper set operations with frozensets
                    overlap = len(comp & prev_comp)
                    distance = np.linalg.norm(centroid - prev_centroid)
                    size_similarity = 1 - abs(size - prev_size)/max(size, prev_size)
                    
                    score = overlap + (1 - distance/100) + size_similarity
                    
                    if score > best_score:
                        best_score = score
                        best_match = prev_color

            if best_match and best_score > 1.5:
                color_map[comp] = best_match
            else:
                color_map[comp] = cmap(self.color_index % 20)
                self.color_index += 1

        # Store as frozensets
        self.prev_components = [
            (frozenset(comp), centroid, len(comp), color)
            for comp, (_, centroid, _), color in zip(current_components, component_features, color_map.values())
        ]
        
        return color_map
    
    def _get_node_id(self, coord, G, pixel_class_vol):
        """Updated to use global node registry"""
        # Find node with matching coordinate
        for node in G.nodes():
            if np.linalg.norm(np.array(G.nodes[node]['coord']) - np.array(coord)) < 2:
                return node
        return None

    def _capture_topological_graph(self, G, component_colors):
        """Visualize the graph with consistent coloring"""
        fig = Figure(figsize=(8, 8))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        
        # Get projected positions
        pos = self.layout_by_projected_centroid(G)
        
        # Create color mappings
        node_color_map = {}
        edge_color_map = {}
        for comp, color in component_colors.items():
            for node in comp:
                node_color_map[node] = color
                # Color edges by their start node's color
                for neighbor in G.neighbors(node):
                    edge_color_map[(node, neighbor)] = color
        
        # Draw with Matplotlib with thicker edges
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=[node_color_map.get(n, "#000000") for n in G.nodes()], node_size=300)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=[edge_color_map.get(e, "#000000") for e in G.edges()], width=4.0)  # Increased from 1.0
        
        # Draw labels with white text
        label_pos = {n: (pos[n][0], pos[n][1]+5) for n in G.nodes()}  # Offset for visibility
        labels = {n: str(self.get_or_create_global_id(G.nodes[n]['coord'])) 
                for n in G.nodes()}
        
        nx.draw_networkx_labels(
            G, label_pos, ax=ax, 
            labels=labels,  # Show global IDs
            font_color='white', 
            font_size=10,
            font_weight='bold'
        )
        
        # Convert to numpy array
        canvas.draw()
        graph_img = np.array(canvas.renderer.buffer_rgba())
        
        plt.close(fig)
        return graph_img
   
   
    def layout_by_projected_centroid(self, G):
        """Project 3D coordinates to 2D using captured viewpoint"""
        pos = {}
        for node in G.nodes():
            coord = G.nodes[node]['coord']
            # Transform using captured view parameters
            pos[node] = self._transform_3d_to_2d(coord)
        return self._adjust_layout(pos)

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
        scaled_y *= -1

        return {node: (scaled_x[i], scaled_y[i]) for i, node in enumerate(pos.keys())}

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

# Add other graph-related methods as needed 