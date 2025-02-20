import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tifffile import imread
from skimage.measure import label, regionprops
from skimage.draw import polygon as sk_polygon
from tqdm import tqdm
import imageio

# Import napari for ROI selection
import napari

# For 3D plotting and widgets
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Button


def get_view_angles(skel_vol):
    """
    Display the first skeleton volume in an interactive 3D plot.
    The nonzero voxels are plotted with depth-encoded colors.
    The user can rotate the view, then click the "Capture View" button to record
    the current elevation and azimuth.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z_idxs, y_idxs, x_idxs = np.nonzero(skel_vol)
    if z_idxs.size > 0:
        norm = plt.Normalize(vmin=z_idxs.min(), vmax=z_idxs.max())
        cmap = plt.get_cmap("plasma")
        colors = cmap(norm(z_idxs))
    else:
        colors = 'gray'
    ax.scatter(x_idxs, y_idxs, z_idxs, c=colors, alpha=0.6, s=10)
    ax.set_title("Adjust view then click 'Capture View'")
    view = {}
    def capture(event):
        view['elev'] = ax.elev
        view['azim'] = ax.azim
        plt.close(fig)
    ax_button = plt.axes([0.8, 0.05, 0.15, 0.075])
    button = Button(ax_button, "Capture View")
    button.on_clicked(capture)
    plt.show()
    return view.get('elev', 20), view.get('azim', -60)


def get_neighbors_3d(coord, shape):
    """
    Return all 26-connected neighbors for a 3D coordinate.
    """
    z, y, x = coord
    neighbors = []
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dz == 0 and dy == 0 and dx == 0:
                    continue
                nz, ny, nx_ = z + dz, y + dy, x + dx
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx_ < shape[2]:
                    neighbors.append((nz, ny, nx_))
    return neighbors


def adjust_layout_to_bbox(pos, target_bbox=(-10, 10, -10, 10)):
    """
    Given a dictionary of 2D positions (node -> (x,y)), linearly scale and translate
    them so that they fit within the target bounding box.
    target_bbox: (xmin, xmax, ymin, ymax)
    """
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    current_xmin, current_xmax = min(xs), max(xs)
    current_ymin, current_ymax = min(ys), max(ys)
    target_xmin, target_xmax, target_ymin, target_ymax = target_bbox
    if current_xmax - current_xmin == 0 or current_ymax - current_ymin == 0:
        return pos
    scale_x = (target_xmax - target_xmin) / (current_xmax - current_xmin)
    scale_y = (target_ymax - target_ymin) / (current_ymax - current_ymin)
    scale = min(scale_x, scale_y)
    new_pos = {}
    for node, p in pos.items():
        new_x = (p[0] - current_xmin) * scale + target_xmin
        new_y = (p[1] - current_ymin) * scale + target_ymin
        new_pos[node] = np.array([new_x, new_y])
    return new_pos


def project_point(coord, elev, azim):
    """
    Project a 3D coordinate (stored as (z, y, x)) into 2D view-space.
    Reorder (z, y, x) into (x, y, z), then apply rotations for azim and elev.
    Return the first two coordinates.
    """
    x, y, z = coord[2], coord[1], coord[0]
    azim_rad = np.deg2rad(azim)
    elev_rad = np.deg2rad(elev)
    Rz = np.array([[np.cos(azim_rad), -np.sin(azim_rad), 0],
                   [np.sin(azim_rad),  np.cos(azim_rad), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(elev_rad), -np.sin(elev_rad)],
                   [0, np.sin(elev_rad),  np.cos(elev_rad)]])
    T = Rx @ Rz
    p = np.array([x, y, z])
    p_trans = T @ p
    return p_trans[:2]


class GraphBuilder3D:
    def __init__(self, pixel_class_file, branch_label_file, skeleton_file, output_dir,
                 dilation_iters=1, view_elev=20, view_azim=-60):
        """
        Initialize the GraphBuilder3D.
        The pixel_class image is assumed to encode:
          Terminal nodes: 2, Edge pixels: 3, Junction pixels: 4.
        """
        self.pixel_class_file = pixel_class_file
        self.branch_label_file = branch_label_file
        self.skeleton_file = skeleton_file
        self.output_dir = output_dir
        self.dilation_iters = dilation_iters
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.roi_mask = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.pixel_class_stack = imread(self.pixel_class_file)
        self.branch_label_stack = imread(self.branch_label_file)
        self.skel_stack = imread(self.skeleton_file)
        self.num_frames = self.pixel_class_stack.shape[0]
        num_z = self.skel_stack.shape[1]
        self.global_vmin = 0
        self.global_vmax = num_z - 1
        self.global_threshold = 2.0  # Adjust as needed.
        self.global_nodes = {}
        self.next_global_node_id = 0

        # For component tracking across frames.
        self.prev_components = None  # List of tuples: (global_node_set, assigned_color)
        self.next_color_index = 0   # Color index for new components.

    def get_or_create_global_id(self, coord):
        """
        For display: round the coordinate and check if a global id exists within the threshold.
        Returns the closest global id if within threshold, otherwise assigns a new one.
        """
        rounded = tuple(round(c, 1) for c in coord)
        best_id = None
        best_distance = None
        for global_id, existing in self.global_nodes.items():
            d = np.linalg.norm(np.array(rounded) - np.array(existing))
            if d < self.global_threshold:
                if best_distance is None or d < best_distance:
                    best_distance = d
                    best_id = global_id
        if best_id is not None:
            return best_id
        new_id = self.next_global_node_id
        self.global_nodes[new_id] = rounded
        self.next_global_node_id += 1
        return new_id

    def assign_component_colors(self, current_comp_globals):
        """
        Given a list of frozensets (each frozenset is the set of global node IDs for a component)
        assign colors based on the previous frame.
        For each current component, if it has significant overlap with a previous component,
        assign it the same color; otherwise, assign a new color.
        Colors are taken from the "tab20" colormap.
        """
        cmap = plt.get_cmap("tab30")
        assigned = {}
        # For each current component:
        for comp in current_comp_globals:
            best_match = None
            best_overlap = 0
            if self.prev_components is not None:
                for prev_comp, prev_color in self.prev_components:
                    overlap = len(comp.intersection(prev_comp))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = prev_color
            if best_match is not None and best_overlap > 0:
                assigned[comp] = best_match
            else:
                assigned[comp] = cmap(self.next_color_index % 20)
                self.next_color_index += 1
        # Update prev_components for next frame.
        self.prev_components = list(assigned.items())
        return assigned

    def select_roi(self):
        first_frame = self.skel_stack[0]
        mip = np.max(first_frame, axis=0)
        print("Launching Napari for ROI selection. Draw a polygon on the MIP image, then close the viewer window when done.")
        viewer = napari.Viewer()
        viewer.add_image(mip, name='MIP')
        shapes_layer = viewer.add_shapes(name='ROI', shape_type='polygon')
        napari.run()
        if len(shapes_layer.data) == 0:
            print("No ROI selected. Using full image.")
            self.roi_mask = np.ones(mip.shape, dtype=bool)
        else:
            polygon_coords = shapes_layer.data[0]
            rr, cc = sk_polygon(polygon_coords[:, 0], polygon_coords[:, 1], mip.shape)
            mask = np.zeros(mip.shape, dtype=bool)
            mask[rr, cc] = True
            self.roi_mask = mask
            print("ROI selected.")

    def build_graph_for_frame(self, frame_idx):
        pixel_class_vol = self.pixel_class_stack[frame_idx].copy()
        if self.roi_mask is not None:
            roi_mask_3d = np.broadcast_to(self.roi_mask, pixel_class_vol.shape)
            pixel_class_vol = np.where(roi_mask_3d, pixel_class_vol, 0)
        junction_mask = (pixel_class_vol == 4)
        junction_labels = label(junction_mask, connectivity=3)
        junction_props = regionprops(junction_labels)
        junction_mapping = {}
        G = nx.Graph()
        node_counter = 0
        for prop in junction_props:
            rep = tuple(map(int, np.round(prop.centroid)))
            junction_mapping[prop.label] = node_counter
            G.add_node(node_counter, coord=rep, type='junction')
            node_counter += 1
        terminal_coords = [tuple(coord) for coord in np.argwhere(pixel_class_vol == 2)]
        terminal_mapping = {}
        for coord in terminal_coords:
            if coord not in terminal_mapping:
                terminal_mapping[coord] = node_counter
                G.add_node(node_counter, coord=coord, type='terminal')
                node_counter += 1

        def get_node_id(coord):
            if pixel_class_vol[coord] == 2:
                return terminal_mapping.get(coord, None)
            elif pixel_class_vol[coord] == 4:
                lab = junction_labels[coord]
                return junction_mapping.get(lab, None)
            return None

        visited = np.zeros(pixel_class_vol.shape, dtype=bool)
        all_node_coords = [G.nodes[node]['coord'] for node in G.nodes()]
        for start_coord in all_node_coords:
            start_id = get_node_id(start_coord)
            if start_id is None:
                continue
            for nb in get_neighbors_3d(start_coord, pixel_class_vol.shape):
                if pixel_class_vol[nb] == 3 and not visited[nb]:
                    path = [start_coord, nb]
                    visited[nb] = True
                    current = nb
                    reached_id = None
                    while True:
                        candidates = [n for n in get_neighbors_3d(current, pixel_class_vol.shape) if n not in path]
                        if not candidates:
                            break
                        next_pixel = None
                        for candidate in candidates:
                            if pixel_class_vol[candidate] in (3, 2, 4):
                                next_pixel = candidate
                                break
                        if next_pixel is None:
                            break
                        path.append(next_pixel)
                        visited[next_pixel] = True
                        node_id = get_node_id(next_pixel)
                        if node_id is not None and node_id != start_id:
                            reached_id = node_id
                            break
                        current = next_pixel
                    if reached_id is not None:
                        if not G.has_edge(start_id, reached_id):
                            G.add_edge(start_id, reached_id, weight=len(path), path=path)
        return G, pixel_class_vol

    def layout_by_projected_centroid(self, G, ideal_edge_length=1.0, max_iter=100, compress_fraction=0.5):
        """
        For each connected component:
          1. Compute a local spring layout (preserving edge lengths) and rescale it so that the mean edge length equals ideal_edge_length.
          2. Project each node's 3D coordinate (using captured view angles) via project_point.
          3. Compute the desired centroid (average of projected 2D points).
          4. Shift the local layout so its centroid matches the desired centroid.
          5. Iteratively resolve overlaps between component bounding boxes by minimally pushing them apart.
          6. Apply a compression step to bring components as close together as possible without overlapping.
        """
        components = list(nx.connected_components(G))
        local_layouts = {}
        desired_centroids = {}
        for comp in components:
            subG = G.subgraph(comp)
            local = nx.spring_layout(subG, k=ideal_edge_length, seed=42)
            if subG.number_of_edges() > 0:
                lengths = [np.linalg.norm(local[u]-local[v]) for u, v in subG.edges()]
                mean_length = np.mean(lengths)
                factor = ideal_edge_length / mean_length
                local = {node: pos * factor for node, pos in local.items()}
            pts = np.array(list(local.values()))
            local_centroid = np.mean(pts, axis=0)
            local_layouts[frozenset(comp)] = local
            projected = np.array([project_point(G.nodes[node]['coord'], self.view_elev, self.view_azim)
                                   for node in comp])
            desired_centroid = np.mean(projected, axis=0)
            desired_centroids[frozenset(comp)] = desired_centroid

        adjusted_layouts = {}
        for comp in components:
            key = frozenset(comp)
            local = local_layouts[key]
            pts = np.array(list(local.values()))
            local_centroid = np.mean(pts, axis=0)
            shift = desired_centroids[key] - local_centroid
            adjusted = {node: pos + shift for node, pos in local.items()}
            adjusted_layouts[key] = adjusted

        def get_bbox(layout):
            xs = [p[0] for p in layout.values()]
            ys = [p[1] for p in layout.values()]
            return (min(xs), max(xs), min(ys), max(ys))
        def boxes_overlap(b1, b2):
            return not (b1[1] <= b2[0] or b1[0] >= b2[1] or b1[3] <= b2[2] or b1[2] >= b2[3])
        comp_keys = list(adjusted_layouts.keys())
        for _ in range(max_iter):
            overlap_found = False
            for i in range(len(comp_keys)):
                for j in range(i+1, len(comp_keys)):
                    key1 = comp_keys[i]
                    key2 = comp_keys[j]
                    bbox1 = get_bbox(adjusted_layouts[key1])
                    bbox2 = get_bbox(adjusted_layouts[key2])
                    if boxes_overlap(bbox1, bbox2):
                        c1 = np.array([(bbox1[0]+bbox1[1])/2, (bbox1[2]+bbox1[3])/2])
                        c2 = np.array([(bbox2[0]+bbox2[1])/2, (bbox2[2]+bbox2[3])/2])
                        overlap_x = min(bbox1[1], bbox2[1]) - max(bbox1[0], bbox2[0])
                        overlap_y = min(bbox1[3], bbox2[3]) - max(bbox1[2], bbox2[2])
                        if overlap_x < overlap_y:
                            if c1[0] < c2[0]:
                                disp = np.array([-overlap_x/2, 0])
                            else:
                                disp = np.array([overlap_x/2, 0])
                        else:
                            if c1[1] < c2[1]:
                                disp = np.array([0, -overlap_y/2])
                            else:
                                disp = np.array([0, overlap_y/2])
                        adjusted_layouts[key1] = {node: pos + disp for node, pos in adjusted_layouts[key1].items()}
                        adjusted_layouts[key2] = {node: pos - disp for node, pos in adjusted_layouts[key2].items()}
                        overlap_found = True
            if not overlap_found:
                break

        # Compression step: push components as close together as possible.
        overall_positions = {}
        for key, layout in adjusted_layouts.items():
            overall_positions.update(layout)
        overall_x = [p[0] for p in overall_positions.values()]
        overall_y = [p[1] for p in overall_positions.values()]
        overall_center = np.array([np.mean(overall_x), np.mean(overall_y)])
        for _ in range(max_iter):
            moved = False
            for key in comp_keys:
                bbox = get_bbox(adjusted_layouts[key])
                comp_center = np.array([(bbox[0]+bbox[1])/2, (bbox[2]+bbox[3])/2])
                vec = overall_center - comp_center
                step = 0.1 * vec
                tentative = {node: pos + step for node, pos in adjusted_layouts[key].items()}
                new_bbox = get_bbox(tentative)
                conflict = False
                for other_key in comp_keys:
                    if other_key == key:
                        continue
                    other_bbox = get_bbox(adjusted_layouts[other_key])
                    if boxes_overlap(new_bbox, other_bbox):
                        conflict = True
                        break
                if not conflict:
                    adjusted_layouts[key] = tentative
                    moved = True
            if not moved:
                break

        overall_pos = {}
        for key, layout in adjusted_layouts.items():
            overall_pos.update(layout)
        # If necessary, apply a final transpose correction:
        final_pos = {node: np.array([pos[1], pos[0]]) for node, pos in overall_pos.items()}
        overall_pos = adjust_layout_to_bbox(final_pos, target_bbox=(-10, 10, -10, 10))
        return overall_pos

    def _compute_fixed_3d_limits(self):
        first_skel = self.skel_stack[0]
        z_idxs, y_idxs, x_idxs = np.nonzero(first_skel)
        pad = 5
        self.fixed_xlim_3d = (max(x_idxs.min() - pad, 0), x_idxs.max() + pad)
        self.fixed_ylim_3d = (max(y_idxs.min() - pad, 0), y_idxs.max() + pad)
        self.fixed_zlim_3d = (max(z_idxs.min() - pad, 0), z_idxs.max() + pad)
        self.fixed_depth_norm = plt.Normalize(vmin=z_idxs.min(), vmax=z_idxs.max())

    def assign_component_colors(self, comp_globals):
        """
        comp_globals: a list of frozensets, where each frozenset is the set of global node IDs for a component.
        Compare these with the previous frame's components (stored in self.prev_components) and assign colors accordingly.
        If a component has significant overlap with a previous component, assign it that color.
        Otherwise, assign a new color.
        """
        cmap = plt.get_cmap("tab20")
        assigned = {}
        for comp in comp_globals:
            best_match = None
            best_overlap = 0
            if self.prev_components is not None:
                for prev_comp, prev_color in self.prev_components:
                    overlap = len(comp.intersection(prev_comp))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = prev_color
            if best_match is not None and best_overlap > 0:
                assigned[comp] = best_match
            else:
                assigned[comp] = cmap(self.next_color_index % 20)
                self.next_color_index += 1
        self.prev_components = list(assigned.items())
        return assigned

    def visualize_graph_and_skeleton(self, G, pixel_class_vol, frame_idx):
        if not hasattr(self, 'fixed_xlim_3d'):
            self._compute_fixed_3d_limits()
        # Compute connected components.
        comps = list(nx.connected_components(G))
        # Build global ID sets for each component.
        comp_globals = []
        for comp in comps:
            global_ids = {self.get_or_create_global_id(G.nodes[node]['coord']) for node in comp}
            comp_globals.append(frozenset(global_ids))
        # Get color mapping based on component tracking.
        comp_color_map = self.assign_component_colors(comp_globals)
        # Build a mapping from each node to its component color.
        node_color_map = {}
        for comp, color in comp_color_map.items():
            for node in G.nodes():
                gid = self.get_or_create_global_id(G.nodes[node]['coord'])
                if gid in comp:
                    node_color_map[node] = color

        fig = plt.figure(figsize=(12, 18))
        # Top: Graph View.
        ax_graph = fig.add_subplot(211)
        pos = self.layout_by_projected_centroid(G, ideal_edge_length=1.0, max_iter=100, compress_fraction=0.5)
        pos = adjust_layout_to_bbox(pos, target_bbox=(-10, 10, -10, 10))
        all_x = [p[0] for p in pos.values()]
        all_y = [p[1] for p in pos.values()]
        margin = 2
        fixed_xlim = (min(all_x)-margin, max(all_x)+margin)
        fixed_ylim = (min(all_y)-margin, max(all_y)+margin)
        node_colors = [node_color_map.get(node, "#000000") for node in G.nodes()]
        edge_colors = [node_color_map.get(u, "#000000") for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color=edge_colors, width=2)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors, node_size=300)
        global_labels = {node: self.get_or_create_global_id(G.nodes[node]['coord']) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=global_labels, ax=ax_graph, font_color='white', font_size=10)
        ax_graph.set_title(f"Graph View (Frame {frame_idx})", fontsize=18)
        ax_graph.axis('off')
        ax_graph.set_xlim(fixed_xlim)
        ax_graph.set_ylim(fixed_ylim)
        # Bottom: 3D Skeleton View.
        ax_3d = fig.add_subplot(212, projection='3d')
        skel_vol = self.skel_stack[frame_idx]
        z_idxs, y_idxs, x_idxs = np.nonzero(skel_vol)
        norm = self.fixed_depth_norm
        cmap_depth = plt.get_cmap("plasma")
        depth_colors = cmap_depth(norm(z_idxs))
        ax_3d.scatter(x_idxs, y_idxs, z_idxs, c=depth_colors, alpha=0.6, marker='o', s=10)
        text_offset = np.array([1.0, 1.0, 1.0])
        for node in G.nodes():
            coord = G.nodes[node]['coord']
            z0, y0, x0 = coord
            node_color = node_color_map.get(node, "#000000")
            ax_3d.scatter(x0, y0, z0, c=[node_color], s=100, marker='o')
            label_text = str(self.get_or_create_global_id(coord))
            ax_3d.text(x0 + text_offset[0], y0 + text_offset[1], z0 + text_offset[2],
                       label_text, color='black', fontsize=10)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f"3D Skeleton View with Graph Nodes (Frame {frame_idx})", fontsize=18)
        ax_3d.view_init(elev=self.view_elev, azim=self.view_azim)
        ax_3d.set_xlim(self.fixed_xlim_3d)
        ax_3d.set_ylim(self.fixed_ylim_3d)
        ax_3d.set_zlim(self.fixed_zlim_3d)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_depth)
        mappable.set_array([])
        fig.colorbar(mappable, ax=ax_3d, shrink=0.5, aspect=10, label='Depth (z-index)')
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"components_frame_{frame_idx}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

    def process_all_frames(self):
        if self.roi_mask is None:
            self.select_roi()
        for frame_idx in tqdm(range(self.num_frames)):
            G, pixel_class_vol = self.build_graph_for_frame(frame_idx)
            if G.number_of_nodes() == 0:
                print(f"No nodes remain in graph for frame {frame_idx}, skipping.")
                continue
            self.visualize_graph_and_skeleton(G, pixel_class_vol, frame_idx)

    def create_gif(self, frame_rate=3):
        import imageio.v2 as imageio
        png_files = sorted(glob.glob(os.path.join(self.output_dir, "components_frame_*.png")),
                           key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]))
        if not png_files:
            print("No frame images found to create GIF.")
            return
        images = []
        for filename in png_files:
            img = imageio.imread(filename)
            images.append(img)
        max_h = max(img.shape[0] for img in images)
        max_w = max(img.shape[1] for img in images)
        padded_images = []
        for img in tqdm(images):
            h, w = img.shape[:2]
            pad_h = max_h - h
            pad_w = max_w - w
            if img.ndim == 3:
                pad_width = ((0, pad_h), (0, pad_w), (0, 0))
                padded = np.pad(img, pad_width, mode='constant', constant_values=255)
            else:
                pad_width = ((0, pad_h), (0, pad_w))
                padded = np.pad(img, pad_width, mode='constant', constant_values=255)
            padded_images.append(padded)
        gif_path = os.path.join(self.output_dir, "components_all_frames.gif")
        duration = 1.0 / frame_rate
        imageio.mimsave(gif_path, padded_images, duration=duration, loop=0)
        print(f"Saved animated GIF to {gif_path}")


def main():
    global_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2025-02-13_yeast_mitographs\event1_2024-10-22_13-14-25_\crop1_snout\crop1_nellie_out\nellie_necessities"
    pixel_class_file = os.path.join(global_path, "crop1.ome-im_pixel_class.ome.tif")
    branch_label_file = os.path.join(global_path, "crop1.ome-im_branch_label_reassigned.ome.tif")
    skeleton_file = os.path.join(global_path, "crop1.ome-im_skel.ome.tif")
    output_dir = os.path.join(global_path, "output_graphs")
    dilation_iters = 1
    full_skel_stack = imread(skeleton_file)
    first_volume = full_skel_stack[0]
    print("Adjust the view for the first skeleton volume to select the desired perspective.")
    view_elev, view_azim = get_view_angles(first_volume)
    print(f"Captured view angles: elevation={view_elev}, azimuth={view_azim}")
    frame_rate = 2
    builder = GraphBuilder3D(pixel_class_file=pixel_class_file,
                               branch_label_file=branch_label_file,
                               skeleton_file=skeleton_file,
                               output_dir=output_dir,
                               dilation_iters=dilation_iters,
                               view_elev=view_elev,
                               view_azim=view_azim)
    builder.process_all_frames()
    builder.create_gif(frame_rate=frame_rate)


if __name__ == "__main__":
    main()
