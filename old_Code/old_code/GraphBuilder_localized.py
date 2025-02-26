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
import warnings

# Import napari for ROI selection
import napari

# For 3D plotting and widgets
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Button

warnings.simplefilter("ignore", FutureWarning)

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
    Project a 3D coordinate using the same view transformation as matplotlib's 3D view
    """
    z, y, x = coord
    
    # Convert angles to radians
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    
    # Create rotation matrices
    cos_elev = np.cos(elev_rad)
    sin_elev = np.sin(elev_rad)
    cos_azim = np.cos(azim_rad)
    sin_azim = np.sin(azim_rad)
    
    # Transform matrix for matplotlib's view transformation
    transform = np.array([
        [cos_azim, -sin_azim*cos_elev],
        [sin_azim, cos_azim*cos_elev]
    ])
    
    # Apply the transformation
    point = np.array([x, y])
    projected = transform @ point
    
    return projected


def get_view_angles(skel_vol):
    """
    Display the first skeleton volume for view selection
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    z_idxs, y_idxs, x_idxs = np.nonzero(skel_vol)
    
    if z_idxs.size > 0:
        norm = plt.Normalize(vmin=z_idxs.min(), vmax=z_idxs.max())
        cmap = plt.get_cmap("plasma")
        colors = cmap(norm(z_idxs))
    else:
        colors = 'gray'
    
    ax.scatter(x_idxs, y_idxs, z_idxs, c=colors, alpha=0.6, s=10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Rotate to desired viewpoint then click 'Capture View'")
    
    view = {}
    def capture(event):
        view['elev'] = ax.elev
        view['azim'] = ax.azim
        plt.close(fig)
    
    ax_button = plt.axes([0.8, 0.05, 0.15, 0.075])
    button = Button(ax_button, "Capture View")
    button.on_clicked(capture)
    plt.show()
    
    return view.get('elev', 90), view.get('azim', 0)


def get_napari_viewpoint(volume):
    """
    Launch a 3D napari viewer so the user may adjust the view.
    Press 'v' to capture the current camera state and close the viewer.
    Returns:
      view_angles: the camera angles (a tuple, e.g. (azimuth, elevation, roll))
      view_center: the camera center (a tuple, assumed to be (x, y, z))
    """
    import napari
    captured = {}
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(volume, name='Raw Volume')
    
    def capture_view(event):
        # Instead of viewer.camera.angles.copy(), convert to tuple.
        captured['angles'] = tuple(viewer.camera.angles)
        captured['center'] = tuple(viewer.camera.center)
        try:
            viewer.window.close()
        except Exception as e:
            print("Error closing viewer, proceeding anyway:", e)
    
    viewer.bind_key('v', capture_view)
    print("Adjust the 3D view in napari. Press 'v' when ready.")
    napari.run()
    # If the user didn't press v, use sensible defaults.
    return captured.get('angles', (0, 90, 0)), captured.get('center', (0, 0, 0))


def transform_coordinate(coord, angle, center):
    """
    Transform a 3D coordinate (stored as (z, y, x)) into 2D using an in-plane rotation.
    We extract x from coord[2] and y from coord[1], then rotate about the provided center.
    The angle is in degrees.
    Returns a 2D point: (transformed_x, transformed_y)
    """
    # Extract the x and y coordinates (we ignore z for 2D projection)
    x = coord[2]
    y = coord[1]
    cx, cy = center[0], center[1]  # assume center is (x,y,z)
    rad = np.deg2rad(angle)
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad),  np.cos(rad)]])
    vec = np.array([x - cx, y - cy])
    vec_rot = R.dot(vec)
    new_point = vec_rot + np.array([cx, cy])
    return new_point


def get_roi_mask_2d(mask):
    """
    Given an ROI mask, returns a 2D version.
    If the mask is 3D, returns the maximum projection along the first axis.
    Otherwise, returns the mask unchanged.
    """
    if mask.ndim == 3:
        return np.max(mask, axis=0)
    else:
        return mask


def render_3d_skeleton_with_napari(skel_vol, branch_vol, G, view_angles, view_center, output_size=(800,600)):
    """
    Render the 3D skeleton (and optionally branch edges) using napari and return a screenshot image.
    skel_vol: 3D skeleton volume (shape: [z, y, x])
    branch_vol: 3D branch volume (used for optional overlays, not implemented in this snippet)
    G: topological graph (unused here but could be used for node labels if desired)
    view_angles: tuple of camera angles e.g. (azimuth, elevation, roll)
    view_center: tuple of camera center coordinates (x, y, z)
    output_size: desired output size for the screenshot
    """
    import napari
    import time
    with napari.gui_qt():
        # Create the viewer with show=False to keep the window hidden
        viewer = napari.Viewer(ndisplay=3, show=False)
        # Extract skeleton coordinates (napari expects points in data order, here (z, y, x))
        z_idxs, y_idxs, x_idxs = np.nonzero(skel_vol)
        points = np.vstack((z_idxs, y_idxs, x_idxs)).T
        # Add a points layer for the skeleton.
        viewer.add_points(points, name='Skeleton', size=3, face_color='cyan')
        
        # (Optional) You could add branch edges and node labels here if desired.
        
        # Set the camera parameters using the captured view.
        viewer.camera.angles = view_angles
        viewer.camera.center = view_center  # Ensure center is set appropriately.
        
        # Optionally set the viewer window size.
        viewer.window.qt_viewer.canvas.native.setFixedSize(*output_size)
        
        # Allow a short pause for the viewer to update.
        time.sleep(0.5)
        img = viewer.screenshot(canvas_only=True)
        try:
            viewer.window.close()
        except Exception as e:
            print("Error closing napari viewer:", e)
    return img


class GraphBuilder3D:
    def __init__(self, pixel_class_file, branch_label_file, skeleton_file, raw_file, output_dir,
                 dilation_iters=1):
        """
        Remove view angle parameters, we'll use fixed view
        """
        self.pixel_class_file = pixel_class_file
        self.branch_label_file = branch_label_file
        self.skeleton_file = skeleton_file
        self.raw_file = raw_file
        self.output_dir = output_dir
        self.dilation_iters = dilation_iters
        self.roi_mask = None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.pixel_class_stack = imread(self.pixel_class_file)
        self.branch_label_stack = imread(self.branch_label_file)
        self.skel_stack = imread(self.skeleton_file)
        self.raw_stack = imread(self.raw_file)
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

        # Use napari to capture a 3D viewpoint from the raw volume.
        print("Please adjust the 3D view in napari. Press 'v' to capture the viewpoint.")
        self.view_angles, self.view_center = get_napari_viewpoint(self.raw_stack)
        # For our 2D rotation, we assume the first element is the azimuth.
        self.view_azim = self.view_angles[0]
        print(f"Captured view: angles={self.view_angles}, center={self.view_center}")

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

    def layout_by_projected_centroid(self, G):
        """
        Project each node's 3D coordinate to 2D using the captured viewpoint.
        """
        pos = {}
        for node in G.nodes():
            coord = G.nodes[node]['coord']
            pos[node] = transform_coordinate(coord, self.view_azim, self.view_center)
        return adjust_layout_to_bbox(pos)

    def _compute_fixed_3d_limits(self):
        first_skel = self.skel_stack[0]
        z_idxs, y_idxs, x_idxs = np.nonzero(first_skel)
        pad = 5
        self.fixed_xlim_3d = (max(x_idxs.min() - pad, 0), x_idxs.max() + pad)
        self.fixed_ylim_3d = (max(y_idxs.min() - pad, 0), y_idxs.max() + pad)  # Keep original y limits
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
        
        # Setup component colors
        comps = list(nx.connected_components(G))
        comp_globals = []
        for comp in comps:
            global_ids = {self.get_or_create_global_id(G.nodes[node]['coord']) for node in comp}
            comp_globals.append(frozenset(global_ids))
        comp_color_map = self.assign_component_colors(comp_globals)
        node_color_map = {}
        for comp, color in comp_color_map.items():
            for node in G.nodes():
                gid = self.get_or_create_global_id(G.nodes[node]['coord'])
                if gid in comp:
                    node_color_map[node] = color

        # Create a 2x2 figure layout
        fig = plt.figure(figsize=(20, 20))
        gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # Calculate cropping ROI and aspect ratio for MIP images
        if self.roi_mask is not None:
            # Create a 2D ROI mask by taking the maximum projection along z.
            roi_mask_2d = get_roi_mask_2d(self.roi_mask)
            rows, cols = np.nonzero(roi_mask_2d)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
        else:
            # Fallback if roi_mask is not set
            min_row, max_row = 0, self.raw_stack.shape[2]
            min_col, max_col = 0, self.raw_stack.shape[3]
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        aspect = width / height

        # Top Right: Raw Image MIP
        ax_raw = fig.add_subplot(gs[0, 1])
        raw_vol = self.raw_stack[frame_idx]
        if self.roi_mask is not None:
            raw_vol = np.where(np.broadcast_to(self.roi_mask, raw_vol.shape), raw_vol, 0)
        raw_mip = np.max(raw_vol, axis=0)
        # Crop to the ROI bounding box if available
        if self.roi_mask is not None:
            # Create a 2D ROI mask by taking the maximum projection along z.
            roi_mask_2d = get_roi_mask_2d(self.roi_mask)
            rows, cols = np.nonzero(roi_mask_2d)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            raw_mip = raw_mip[min_row:max_row+1, min_col:max_col+1]
        from skimage.transform import rotate as sk_rotate
        # Rotate the MIP image by -view_azim so it matches the captured view.
        raw_mip_rot = sk_rotate(raw_mip, angle=-self.view_azim, resize=True)
        # Flip vertically to fix upside down display
        raw_mip_rot = np.flipud(raw_mip_rot)
        ax_raw.imshow(raw_mip_rot, cmap='gray', aspect=aspect)
        ax_raw.set_title(f"Raw Image MIP (Frame {frame_idx})", fontsize=18)
        ax_raw.axis('off')

        # Top Left: Branch Labels MIP
        ax_branch = fig.add_subplot(gs[0, 0])
        branch_vol = self.branch_label_stack[frame_idx]
        if self.roi_mask is not None:
            branch_vol = np.where(np.broadcast_to(self.roi_mask, branch_vol.shape), branch_vol, 0)
        branch_mip = np.max(branch_vol, axis=0)
        if self.roi_mask is not None:
            # Create a 2D ROI mask by taking the maximum projection along z.
            roi_mask_2d = get_roi_mask_2d(self.roi_mask)
            rows, cols = np.nonzero(roi_mask_2d)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            branch_mip = branch_mip[min_row:max_row+1, min_col:max_col+1]
        branch_mip_rot = sk_rotate(branch_mip, angle=-self.view_azim, resize=True)
        branch_mip_rot = np.flipud(branch_mip_rot)
        ax_branch.imshow(branch_mip_rot, cmap='nipy_spectral', aspect=aspect)
        ax_branch.set_title(f"Branch Labels MIP (Frame {frame_idx})", fontsize=18)
        ax_branch.axis('off')

        # Bottom Left: 3D Skeleton View with grid and node labels (matplotlib implementation)
        ax_skel = fig.add_subplot(gs[1, 0], projection='3d')
        skel_vol = self.skel_stack[frame_idx]
        if self.roi_mask is not None:
            skel_vol = np.where(np.broadcast_to(self.roi_mask, skel_vol.shape), skel_vol, 0)
        z_idxs, y_idxs, x_idxs = np.nonzero(skel_vol)
        if len(z_idxs) > 0:
            norm = self.fixed_depth_norm
            cmap_depth = plt.get_cmap("plasma")
            depth_colors = cmap_depth(norm(z_idxs))
            # In our data, coordinates are stored as (z, y, x). For a standard 3D plot, we interpret:
            #   x = x_idxs, y = y_idxs, z = z_idxs.
            ax_skel.scatter(x_idxs, y_idxs, z_idxs, c=depth_colors, alpha=0.6, marker='o', s=10)
            # Also plot branch edges in 3D:
            branch_vol = self.branch_label_stack[frame_idx]
            unique_labels = np.unique(branch_vol[branch_vol > 0])
            for label in unique_labels:
                mask = branch_vol == label
                z, y, x = np.where(mask)
                if len(z) > 1:
                    color = plt.cm.nipy_spectral(label / unique_labels.max())
                    ax_skel.plot(x, y, z, color=color, alpha=0.3, linewidth=1)
        # Add node labels on the skeleton view.
        text_offset = np.array([2.0, 2.0, 2.0])
        for node in G.nodes():
            coord = G.nodes[node]['coord']  # (z, y, x)
            label_text = str(self.get_or_create_global_id(coord))
            ax_skel.text(coord[2] + text_offset[0], coord[1] + text_offset[1], coord[0] + text_offset[2],
                         label_text, color='black', fontsize=10)
        ax_skel.grid(True)
        ax_skel.set_xlabel('X')
        ax_skel.set_ylabel('Y')
        ax_skel.set_zlabel('Z')
        # Set a fixed top-down view to match the raw image (MIP) orientation.
        ax_skel.view_init(elev=90, azim=0)
        ax_skel.set_title(f"3D Skeleton View (Frame {frame_idx})", fontsize=18)

        # Bottom Right: Topological Graph View
        ax_graph = fig.add_subplot(gs[1, 1])
        pos = {}
        for node in G.nodes():
            coord = G.nodes[node]['coord']
            pos[node] = transform_coordinate(coord, self.view_azim, self.view_center)
        pos = adjust_layout_to_bbox(pos, target_bbox=(-10, 10, -10, 10))
        node_colors = [node_color_map.get(node, "#000000") for node in G.nodes()]
        edge_colors = [node_color_map.get(u, "#000000") for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color=edge_colors, width=2)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color=node_colors, node_size=300)
        global_labels = {node: self.get_or_create_global_id(G.nodes[node]['coord']) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=global_labels, ax=ax_graph, font_color='white', font_size=10)
        ax_graph.set_title(f"Graph View (Frame {frame_idx})", fontsize=18)
        ax_graph.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"components_frame_{frame_idx}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
    raw_file = os.path.join(global_path, "crop1.ome.ome.tif")
    output_dir = os.path.join(global_path, "output_graphs")
    dilation_iters = 1
    frame_rate = 2
    
    builder = GraphBuilder3D(pixel_class_file=pixel_class_file,
                           branch_label_file=branch_label_file,
                           skeleton_file=skeleton_file,
                           raw_file=raw_file,
                           output_dir=output_dir,
                           dilation_iters=dilation_iters)
    builder.process_all_frames()
    builder.create_gif(frame_rate=frame_rate)


if __name__ == "__main__":
    main()
