import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tifffile import imread
from skimage.measure import label, regionprops
from scipy.ndimage import binary_dilation
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


class GraphBuilder3D:
    def __init__(self, pixel_class_file, branch_label_file, skeleton_file, output_dir,
                 dilation_iters=1, view_elev=20, view_azim=-60):
        """
        Initialize the GraphBuilder3D.

        The pixel_class image is assumed to encode:
            Terminal nodes: 2
            Edge pixels: 3
            Junction pixels: 4

        Junction pixels will be merged into a single node via connected-component analysis.
        """
        self.pixel_class_file = pixel_class_file
        self.branch_label_file = branch_label_file
        self.skeleton_file = skeleton_file
        self.output_dir = output_dir
        self.dilation_iters = dilation_iters
        self.view_elev = view_elev
        self.view_azim = view_azim
        self.roi_mask = None  # 2D mask for ROI selection

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Load image stacks (each frame is a 3D volume: (frames, z, y, x))
        self.pixel_class_stack = imread(self.pixel_class_file)
        self.branch_label_stack = imread(self.branch_label_file)
        self.skel_stack = imread(self.skeleton_file)
        self.num_frames = self.pixel_class_stack.shape[0]

        num_z = self.skel_stack.shape[1]
        self.global_vmin = 0
        self.global_vmax = num_z - 1

        # Global node mapping for label consistency (for display only).
        self.global_threshold = 2.0  # distance threshold
        self.global_nodes = {}  # global_id -> rounded coordinate tuple
        self.next_global_node_id = 0

    def get_or_create_global_id(self, coord):
        """
        For display only: round the coordinate and check if a global node exists within the threshold.
        If so, return its global id; otherwise, create a new one.
        """
        rounded = tuple(round(c, 1) for c in coord)
        for global_id, existing in self.global_nodes.items():
            if np.linalg.norm(np.array(rounded) - np.array(existing)) < self.global_threshold:
                return global_id
        new_id = self.next_global_node_id
        self.global_nodes[new_id] = rounded
        self.next_global_node_id += 1
        return new_id

    def select_roi(self):
        """
        Launch Napari for ROI selection.
        """
        first_frame = self.skel_stack[0]
        mip = np.max(first_frame, axis=0)
        print(f"DEBUG: MIP shape: {mip.shape}")

        print("Launching Napari for ROI selection. Draw a polygon on the MIP image,")
        print("then close the viewer window when done.")

        viewer = napari.Viewer()
        viewer.add_image(mip, name='MIP')
        shapes_layer = viewer.add_shapes(name='ROI', shape_type='polygon')
        napari.run()

        if len(shapes_layer.data) == 0:
            print("No ROI selected. Using full image.")
            self.roi_mask = np.ones(mip.shape, dtype=bool)
        else:
            polygon_coords = shapes_layer.data[0]
            print(f"DEBUG: Polygon coordinates shape: {polygon_coords.shape}")
            rr, cc = sk_polygon(polygon_coords[:, 0], polygon_coords[:, 1], mip.shape)
            mask = np.zeros(mip.shape, dtype=bool)
            mask[rr, cc] = True
            self.roi_mask = mask
            print(f"DEBUG: ROI mask shape: {self.roi_mask.shape}")
            print("ROI selected.")

    def build_graph_for_frame(self, frame_idx):
        """
        Process a single frame (3D volume) and build a graph by tracing edges
        between terminal nodes (value 2) and merged junctions (value 4).
        Edge pixels have value 3.
        (Graph generation remains unchanged.)
        """
        pixel_class_vol = self.pixel_class_stack[frame_idx].copy()
        if self.roi_mask is not None:
            roi_mask_3d = np.broadcast_to(self.roi_mask, pixel_class_vol.shape)
            print(f"DEBUG: ROI mask 3d shape: {roi_mask_3d.shape}")
            pixel_class_vol = np.where(roi_mask_3d, pixel_class_vol, 0)
            print(f"DEBUG: Filtered pixel class volume shape: {pixel_class_vol.shape}")

        # --- Merge junction pixels (value 4) ---
        junction_mask = (pixel_class_vol == 4)
        junction_labels = label(junction_mask, connectivity=3)
        junction_props = regionprops(junction_labels)
        junction_mapping = {}  # maps junction label -> local node id
        G = nx.Graph()
        node_counter = 0
        for prop in junction_props:
            rep = tuple(map(int, np.round(prop.centroid)))
            junction_mapping[prop.label] = node_counter
            G.add_node(node_counter, coord=rep, type='junction')
            node_counter += 1

        # --- Terminal nodes (value 2) ---
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

    def layout_disconnected_components(self, G, ideal_edge_length=1.0, base_distance=3.0):
        """
        Compute a layout that arranges disconnected components in concentric rings.
        Each connected component is laid out using a spring layout and then scaled so that
        the average edge length in that component equals ideal_edge_length.
        The largest component is placed at the center and smaller ones are arranged in
        concentric rings outward.
        """
        import math

        components = list(nx.connected_components(G))
        components = sorted(components, key=lambda comp: len(comp), reverse=True)
        overall_pos = {}

        def scale_layout(subG, pos, ideal_length):
            if subG.number_of_nodes() <= 1:
                return pos
            edges = list(subG.edges())
            lengths = [np.linalg.norm(pos[u] - pos[v]) for u, v in edges]
            avg_length = np.mean(lengths) if lengths else 1.0
            scale = ideal_length / avg_length
            for node in pos:
                pos[node] = pos[node] * scale
            return pos

        central_comp = components[0]
        central_subG = G.subgraph(central_comp)
        central_pos = nx.spring_layout(central_subG, k=ideal_edge_length, seed=42)
        central_pos = scale_layout(central_subG, central_pos, ideal_edge_length)
        for node, pos in central_pos.items():
            overall_pos[node] = pos

        remaining = components[1:]
        n = len(remaining)
        if n > 0:
            comps_per_ring = math.ceil(math.sqrt(n))
            for idx, comp in enumerate(remaining, start=1):
                ring_index = math.ceil(idx / comps_per_ring)
                pos_in_ring = (idx - 1) % comps_per_ring
                angle = 2 * math.pi * pos_in_ring / comps_per_ring
                offset = np.array([
                    base_distance * ring_index * math.cos(angle),
                    base_distance * ring_index * math.sin(angle)
                ])
                subG = G.subgraph(comp)
                subpos = nx.spring_layout(subG, k=ideal_edge_length, seed=42)
                subpos = scale_layout(subG, subpos, ideal_edge_length)
                for node, pos in subpos.items():
                    overall_pos[node] = pos + offset
        return overall_pos

    def _compute_fixed_3d_limits(self):
        """
        Compute fixed axis limits from the first frame of the skeleton stack.
        """
        first_skel = self.skel_stack[0]
        z_idxs, y_idxs, x_idxs = np.nonzero(first_skel)
        pad = 5
        self.fixed_xlim_3d = (max(x_idxs.min() - pad, 0), x_idxs.max() + pad)
        self.fixed_ylim_3d = (max(y_idxs.min() - pad, 0), y_idxs.max() + pad)
        self.fixed_zlim_3d = (max(z_idxs.min() - pad, 0), z_idxs.max() + pad)

    def visualize_graph_and_skeleton(self, G, pixel_class_vol, frame_idx):
        """
        Visualize the graph and 3D skeleton view.
        The layout of the graph is computed using layout_disconnected_components,
        which preserves the original layout.
        Global node labels (consistent between frames) are computed for display only.
        A text offset is applied in the 3D view so that node labels are more visible.
        """
        if not hasattr(self, 'fixed_xlim_3d'):
            self._compute_fixed_3d_limits()

        fixed_xlim = (-10, 10)
        fixed_ylim = (-10, 10)

        fig = plt.figure(figsize=(12, 18))

        # ---- Top: Graph View ----
        ax_graph = fig.add_subplot(211)
        # Use the custom layout function to preserve the previous layout.
        pos = self.layout_disconnected_components(G, ideal_edge_length=1.0, base_distance=3.0)
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='blue', width=2)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='red', node_size=300)
        # Build a global label mapping for display.
        global_labels = {node: self.get_or_create_global_id(G.nodes[node]['coord'])
                         for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=global_labels, ax=ax_graph, font_color='white', font_size=10)
        ax_graph.set_title(f"Graph View (Frame {frame_idx})", fontsize=18)
        ax_graph.axis('off')
        ax_graph.set_xlim(fixed_xlim)
        ax_graph.set_ylim(fixed_ylim)

        # ---- Bottom: 3D Skeleton View ----
        ax_3d = fig.add_subplot(212, projection='3d')
        skel_vol = self.skel_stack[frame_idx]
        z_idxs, y_idxs, x_idxs = np.nonzero(skel_vol)
        norm = plt.Normalize(vmin=z_idxs.min(), vmax=z_idxs.max())
        cmap = plt.get_cmap("plasma")
        colors = cmap(norm(z_idxs))
        ax_3d.scatter(x_idxs, y_idxs, z_idxs, c=colors, alpha=0.6, marker='o', s=10)
        text_offset = np.array([1.0, 1.0, 1.0])
        for node in G.nodes():
            coord = G.nodes[node]['coord']
            z0, y0, x0 = coord
            ax_3d.scatter(x0, y0, z0, c='red', s=100, marker='o')
            label_text = str(self.get_or_create_global_id(coord))
            ax_3d.text(x0 + text_offset[0], y0 + text_offset[1], z0 + text_offset[2],
                       label_text, color='black', fontsize=12)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f"3D Skeleton View with Graph Nodes (Frame {frame_idx})", fontsize=18)
        ax_3d.view_init(elev=self.view_elev, azim=self.view_azim)
        ax_3d.set_xlim(self.fixed_xlim_3d)
        ax_3d.set_ylim(self.fixed_ylim_3d)
        ax_3d.set_zlim(self.fixed_zlim_3d)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"components_frame_{frame_idx}.png")
        plt.savefig(output_path, dpi=150)
        plt.close()

    def process_all_frames(self):
        """
        Process all frames: build graphs and save visualizations.
        """
        if self.roi_mask is None:
            self.select_roi()

        for frame_idx in tqdm(range(self.num_frames)):
            G, pixel_class_vol = self.build_graph_for_frame(frame_idx)
            if G.number_of_nodes() == 0:
                print(f"No nodes remain in graph for frame {frame_idx}, skipping.")
                continue
            self.visualize_graph_and_skeleton(G, pixel_class_vol, frame_idx)

    def create_gif(self, frame_rate=12):
        """
        Create an animated GIF from saved frame images.
        """
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
    # global_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2025-02-13_yeast_mitographs\event1_2024-10-22_13-14-25_\crop1_snout\crop1_nellie_out\nellie_necessities"
    # global_path = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2025-02-13_yeast_mitographs\event1_2024-10-22_13-18-01_\crop1_snout\crop1_nellie_out\nellie_necessities"
    global_path = r"/Users/gabrielsturm/Documents/GitHub/Nellie_MG/event1_2024-10-22_13-14-25_/crop1_snout/crop1_nellie_out/nellie_necessities"
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

    frame_rate = 3

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
