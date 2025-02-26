import sys
import os
import time
import napari
import numpy as np
from tifffile import imread
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class ViewpointSelector:
    def __init__(self, volume_file, output_file='final_view.png', projection_axis=0):
        """
        Initialize with the path to a volume file and an output filename.
        The volume is assumed to be a 4D OME-TIF with shape (T, Z, Y, X).
        This script uses only the first time point, resulting in a 3D volume (Z, Y, X).
        You can specify the projection_axis along which the max projection is computed.
        For a volume stored as (Z, Y, X), use projection_axis=0 to obtain an (Y, X) image.
        """
        self.volume_file = volume_file
        self.output_file = output_file
        self.projection_axis = projection_axis
        self.volume = imread(volume_file)
        # If the volume has a time dimension (T, Z, Y, X), take the first time point.
        if self.volume.ndim == 4:
            self.volume = self.volume[0]
        self.captured_view = None
        self.zoom_level = None
        self.viewport_size = None  # Store the viewport size for scaling
        self.roi_polygon = None

    def select_viewpoint(self, volume_to_view):
        """
        Launch a 3D viewer of the skeleton volume.
        Adjust the view and press 'v' to capture the view.
        The screenshot is stored in self.skeleton_img and camera parameters in self.captured_view.
        """
        captured_view = {}
    
        viewer = napari.Viewer(ndisplay=3)
        layer = viewer.add_image(self.volume, name="Skeleton")
    
        def capture_view(event):
            print("Capturing skeleton view (screenshot)...")
            time.sleep(0.5)
            # Store zoom level and viewport size for consistent scaling
            self.zoom_level = viewer.camera.zoom
            self.viewport_size = viewer.window.qt_viewer.canvas.size
            print(f"Captured zoom level: {self.zoom_level}")
            print(f"Captured viewport size: {self.viewport_size}")
            
            img = viewer.screenshot(canvas_only=True)
            imageio.imwrite(self.output_file, img)
            print(f"Skeleton view saved to {self.output_file}")
            captured_view['view'] = (viewer.camera.angles, viewer.camera.center)
            # Store the screenshot in self.skeleton_img for ROI selection.
            self.skeleton_img = img
            viewer.close()
    
        viewer.bind_key('v', capture_view)
        print("Adjust the 3D view. Press 'v' to capture the view and close the viewer automatically.")
        napari.run()
    
        self.captured_view = captured_view.get('view', None)
        if self.captured_view is None:
            print("No view was captured.")
        else:
            print("Captured view:", self.captured_view)
        return self.captured_view

    def crop_final_image(self):
        """
        Load the captured final image (self.output_file) and allow the user to draw an ROI
        in a 2D napari viewer. The drawn ROI is then used to crop the final image.
        The cropped image is saved as "<output_file>_cropped.png".
        """
        final_img = imageio.imread(self.output_file)
    
        # Create a 2D viewer for ROI selection.
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(final_img, name="Final Captured Image")
        shapes_layer = viewer.add_shapes(name="ROI", shape_type="polygon")
        print("Draw an ROI on the final captured image. When finished, close the viewer window to proceed.")
        napari.run()
    
        if len(shapes_layer.data) > 0:
            roi_polygon = shapes_layer.data[0]
            print("ROI captured on final image.")
        else:
            print("No ROI drawn on final image.")
            roi_polygon = None
    
        if roi_polygon is None:
            print("No ROI selected on final image. Using full final image.")
            cropped_img = final_img  # no cropping
        else:
            # Convert the drawn polygon (assumed in (y, x) order) to an ROI mask.
            poly = np.array(roi_polygon)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=final_img.shape[:2])
            mask = np.zeros(final_img.shape[:2], dtype=bool)
            mask[rr, cc] = True
            rows, cols = np.nonzero(mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            cropped_img = final_img[min_row:max_row + 1, min_col:max_col + 1]
            # (Optional) Display the cropped final image for confirmation.
            plt.figure()
            plt.imshow(cropped_img)
            plt.title("Cropped Final View")
            plt.axis('off')
            plt.show()
    
        # Append a legend (colorbar) to the cropped final image.
        cropped_output_file = os.path.splitext(self.output_file)[0] + "_cropped.png"
        # Use the depth range from the original 3D volume.
        Z = self.volume.shape[0]
        cmap = plt.get_cmap("plasma")
        norm_obj = plt.Normalize(vmin=0, vmax=Z-1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([])
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(cropped_img)
        ax.axis('off')
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Depth (Z)")
        fig.savefig(cropped_output_file, bbox_inches='tight')
        plt.close(fig)
        print(f"Cropped final view with legend saved to {cropped_output_file}")
        return cropped_output_file

    def composite_view(self, branch_file, original_file):
        # (This method is removed in favor of the new capture_modalities method.)
        pass

    def run_workflow(self, branch_file, original_file):
        """
        Run the workflow:
          1. Select viewpoint on the skeleton volume.
          2. Let the user draw an ROI on the captured skeleton screenshot.
          3. Capture composite screenshots of the raw (original), branch, and skeleton modalities.
          4. Generate a depth-encoded image of the skeleton.
        Returns a tuple: (composite image filename, depth encoded image filename).
        """
        # First, select the viewpoint from the skeleton:
        self.select_viewpoint(self.volume)
        # Then, let the user draw an ROI on the captured skeleton screenshot:
        self.select_roi()
        # Capture composite modalities:
        composite_file = self.capture_modalities(branch_file, original_file)
        # Capture depth-encoded image:
        depth_file = self.capture_depth_encoded()
        return composite_file, depth_file

    def select_roi(self):
        """
        Display the captured skeleton screenshot in a 2D viewer.
        Let the user draw an ROI polygon. When the viewer is closed, the ROI is saved in self.roi_polygon.
        """
        if not hasattr(self, 'skeleton_img'):
            print("No skeleton screenshot available for ROI selection.")
            return
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(self.skeleton_img, name="Skeleton Screenshot")
        shapes_layer = viewer.add_shapes(name="ROI", shape_type="polygon")
        print("Draw an ROI on the skeleton screenshot. When finished, close the viewer to proceed.")
        napari.run()
        if len(shapes_layer.data) > 0:
            self.roi_polygon = shapes_layer.data[0]
            print("ROI captured.")
        else:
            print("No ROI drawn; using full image as ROI.")
            self.roi_polygon = None

    def capture_screenshot(self, volume, layer_type='image'):
        """
        Open a 3D viewer for the given volume, set the camera to the captured view,
        and capture a screenshot. The parameter 'layer_type' controls how the volume is displayed:
         - 'image': use viewer.add_image (for raw images).
         - 'labels': use viewer.add_labels (for label images with preset color maps).
        Then close the viewer and return the image.
        """
        viewer = napari.Viewer(ndisplay=3)
        if layer_type == 'labels':
            viewer.add_labels(volume, name="Modality")
        else:
            viewer.add_image(volume, name="Modality")

        if self.captured_view is not None:
            angles, center = self.captured_view
            viewer.camera.angles = angles
            viewer.camera.center = center
            
            # Apply the exact same zoom level that was captured
            if self.zoom_level is not None:
                viewer.camera.zoom = self.zoom_level
                print(f"Using captured zoom level: {self.zoom_level}")
            
            # # Ensure the viewport size matches what was captured
            # if self.viewport_size is not None:
            #     print(f"Resizing viewport to match captured size: {self.viewport_size}")
            #     # Only attempt if viewport_size is a tuple with 2 values
            #     if isinstance(self.viewport_size, tuple) and len(self.viewport_size) == 2:
            #         try:
            #             # Fix for deprecation warning - use canvas.view instead of view
            #             # And use resize() rather than setFixedSize()
            #             viewer.window.qt_viewer.canvas.view.resize(*self.viewport_size)
            #         except Exception as e:
            #             print(f"Warning: Could not resize viewport: {e}")
        else:
            print("No captured viewpoint; using default camera.")
        time.sleep(0.5)
        img = viewer.screenshot(canvas_only=True)
        viewer.close()
        return img

    def capture_modalities(self, branch_file, original_file):
        """
        Capture screenshots for skeleton, branch, and original volumes using the captured camera view.
        Apply the same ROI (if drawn) to all screenshots and combine them side-by-side.
        Save and return the composite image filename.
        """
        # Load volumes and capture screenshots for all four modalities
        img_skel, img_branch, img_original, img_depth = self._capture_all_screenshots(branch_file, original_file)
        
        # Apply ROI filtering and cropping
        cropped_images, _ = self._apply_roi_to_images(img_skel, img_branch, img_original, img_depth)
        cropped_skel, cropped_branch, cropped_original, cropped_depth = cropped_images
        
        # Create and save composite image
        composite_file = self._create_composite_image([cropped_original, cropped_branch, cropped_skel, cropped_depth])
        
        return composite_file

    def _capture_all_screenshots(self, branch_file, original_file):
        """
        Load volumes and capture screenshots for all four modalities.
        Returns: tuple of (skeleton_img, branch_img, original_img, depth_img)
        """
        # Capture screenshot for skeleton (using raw image display)
        img_skel = self.capture_screenshot(self.volume, layer_type='image')

        # Load branch and original volumes
        branch = imread(branch_file)
        if branch.ndim == 4:
            branch = branch[0]
        original = imread(original_file)
        if original.ndim == 4:
            original = original[0]

        # Capture branch using labels so that its pre-assigned colors are preserved
        img_branch = self.capture_screenshot(branch, layer_type='labels')
        # Capture raw (original) image
        img_original = self.capture_screenshot(original, layer_type='image')
        
        # Generate and capture depth-encoded image
        img_depth = self._generate_depth_encoded_image()
        
        return img_skel, img_branch, img_original, img_depth

    def _apply_roi_to_images(self, img_skel, img_branch, img_original, img_depth):
        """
        Apply ROI filtering and cropping to all four modality images.
        Returns: 
            - tuple of cropped images (skel, branch, original, depth)
            - crop coordinates (min_row, max_row, min_col, max_col) or None if no ROI
        """
        if self.roi_polygon is None:
            print("No ROI selected; using full image for all modalities.")
            return (img_skel, img_branch, img_original, img_depth), None
        
        # Create binary mask from polygon
        poly = np.array(self.roi_polygon)  # Assumed (y, x) coordinates
        
        # Scale the ROI polygon if image dimensions don't match the original capture
        original_height, original_width = self.skeleton_img.shape[:2]
        current_height, current_width = img_skel.shape[:2]
        
        if (original_height != current_height or original_width != current_width):
            print(f"Scaling ROI from original dimensions {original_width}x{original_height} " +
                  f"to current dimensions {current_width}x{current_height}")
            
            # Calculate scaling factors
            y_scale = current_height / original_height
            x_scale = current_width / original_width
            
            # Scale the polygon
            scaled_poly = poly.copy()
            scaled_poly[:, 0] = poly[:, 0] * y_scale
            scaled_poly[:, 1] = poly[:, 1] * x_scale
            poly = scaled_poly
        
        # Check if polygon is valid
        if len(poly) >= 3:  # Need at least 3 points for a polygon
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=img_skel.shape[:2])
            mask = np.zeros(img_skel.shape[:2], dtype=np.uint8)
            mask[rr, cc] = 1
        else:
            print("Warning: Invalid ROI polygon with less than 3 points. Using full image.")
            mask = np.ones(img_skel.shape[:2], dtype=np.uint8)

        # Print image dimensions to help debug scaling issues
        print(f"Image dimensions - Skeleton: {img_skel.shape}, Branch: {img_branch.shape}, "
              f"Original: {img_original.shape}, Depth: {img_depth.shape}")

        # Resize images to match skeleton screenshot size
        target_shape = img_skel.shape
        
        # Force all images to exactly match the skeleton dimensions
        if img_branch.shape[:2] != target_shape[:2]:
            img_branch = resize(img_branch, target_shape[:2], 
                              preserve_range=True, anti_aliasing=True).astype(img_branch.dtype)
        
        if img_original.shape[:2] != target_shape[:2]:
            img_original = resize(img_original, target_shape[:2], 
                                preserve_range=True, anti_aliasing=True).astype(img_original.dtype)
        
        if img_depth.shape[:2] != target_shape[:2]:
            img_depth = resize(img_depth, target_shape[:2], 
                             preserve_range=True, anti_aliasing=True).astype(img_depth.dtype)

        # Print dimensions after resize to verify they match
        print(f"After resize - Skeleton: {img_skel.shape}, Branch: {img_branch.shape}, "
              f"Original: {img_original.shape}, Depth: {img_depth.shape}")

        # Apply mask to each modality
        filtered_images = self._apply_mask_to_images([img_skel, img_branch, img_original, img_depth], mask)
        filtered_skel, filtered_branch, filtered_original, filtered_depth = filtered_images

        # Compute bounding box from mask
        rows, cols = np.nonzero(mask)
        
        # Check if mask has any non-zero elements
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = int(rows.min()), int(rows.max())
            min_col, max_col = int(cols.min()), int(cols.max())
        else:
            # If mask is empty, use the entire image
            print("Warning: Empty mask. Using full image dimensions.")
            min_row, max_row = 0, img_skel.shape[0] - 1
            min_col, max_col = 0, img_skel.shape[1] - 1
            
        crop_coords = (min_row, max_row, min_col, max_col)

        # Crop filtered images
        cropped_skel = filtered_skel[min_row:max_row + 1, min_col:max_col + 1]
        cropped_branch = filtered_branch[min_row:max_row + 1, min_col:max_col + 1]
        cropped_original = filtered_original[min_row:max_row + 1, min_col:max_col + 1]
        cropped_depth = filtered_depth[min_row:max_row + 1, min_col:max_col + 1]
        
        return (cropped_skel, cropped_branch, cropped_original, cropped_depth), crop_coords

    def _ensure_shape(self, img, target_shape):
        """Resize image to match target shape if needed."""
        if img.shape != target_shape:
            return resize(img, target_shape, preserve_range=True, 
                         anti_aliasing=True).astype(img.dtype)
        return img

    def _apply_mask_to_images(self, images, mask):
        """Apply binary mask to a list of images, handling RGB vs grayscale."""
        filtered_images = []
        for img in images:
            if img.ndim == 3:  # RGB image
                # Create a black background and copy only pixels inside the mask
                filtered = np.zeros_like(img)
                # Only copy pixels where mask is active
                filtered[mask > 0] = img[mask > 0]
            else:  # Grayscale image
                filtered = np.zeros_like(img)
                filtered[mask > 0] = img[mask > 0]
            filtered_images.append(filtered)
        return filtered_images

    def _generate_depth_encoded_image(self):
        """
        Generate a depth-encoded image of the skeleton using points colored by Z-depth.
        Returns: depth-encoded image as a screenshot
        """
        # Create a standard napari viewer - this worked before
        viewer = napari.Viewer(ndisplay=3)
        
        # Get coordinates of skeleton voxels
        volume_mask = (self.volume > 0)
        z_coords, y_coords, x_coords = np.where(volume_mask)
        
        # Use the same colormap approach that worked before
        norm = plt.Normalize(vmin=z_coords.min(), vmax=z_coords.max())
        cmap = plt.cm.plasma
        colors = cmap(norm(z_coords))
        
        # Add points layer with the same parameters as before
        points = np.column_stack((z_coords, y_coords, x_coords))
        points_layer = viewer.add_points(
            points,
            name="Depth-Encoded Skeleton",
            size=4,  # Slightly larger for better visibility
            face_color=colors,
            edge_color="transparent",
            shading="none"  # Disable shading which might affect visibility
        )
        
        # Set camera to captured viewpoint
        if self.captured_view is not None:
            angles, center = self.captured_view
            viewer.camera.angles = angles
            viewer.camera.center = center
            
            # Important: Don't reset view as it changes the perspective
            # Just ensure we're using same scale/zoom as other modalities
            try:
                # Use the same zoom level as in capture_screenshot
                # This is a reasonable default that should match other modalities
                viewer.camera.zoom = self.zoom_level
            except:
                pass  # If zoom setting fails, use default
        
        # Wait a bit longer to ensure rendering is complete
        time.sleep(0.5)
        img_depth = viewer.screenshot(canvas_only=True)
        
        # Print depth image dimensions to help debug
        print(f"Depth image dimensions before closing viewer: {img_depth.shape}")
        
        viewer.close()
        
        # Check if the image is empty (all black)
        if np.mean(img_depth) < 0.01:  # Very dark image
            print("Warning: Depth image appears to be empty. Using alternative approach.")
            return self._create_simple_depth_image()
            
        return img_depth

    def _create_simple_depth_image(self):
        """Create a simple depth-encoded image as a fallback."""
        # Create a simple colored image based on the skeleton
        Z, Y, X = self.volume.shape
        
        # Project the skeleton to 2D, preserving Z information
        depth_img = np.zeros((Y, X, 4), dtype=np.float32)
        
        # For each Z-slice, color the skeleton voxels
        for z in range(Z):
            # Get the normalized z-value for coloring
            z_norm = z / max(1, Z-1)
            # Get color from plasma colormap
            color = plt.cm.plasma(z_norm)
            
            # Get mask for this slice
            mask = self.volume[z] > 0
            
            # Apply color to this slice's voxels
            for c in range(4):
                depth_img[:, :, c] = np.where(mask, color[c], depth_img[:, :, c])
        
        return depth_img

    def _create_composite_image(self, images):
        """
        Create a composite image by horizontally stacking the provided images.
        Save it with a black background and white title.
        Returns: path to saved composite image
        """
        # Combine images horizontally
        composite = np.hstack(images)
        
        # Create figure with black background
        composite_file = os.path.splitext(self.output_file)[0] + "_composite.png"
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
        ax.imshow(composite)
        ax.set_title("Composite 3D View", color="white", fontsize=14)
        ax.axis('off')
        
        # Save figure
        fig.savefig(composite_file, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"Composite image saved to {composite_file}")
        
        return composite_file

    def depth_encoded_skeleton(self):
        """
        Compute a depth-encoded 2D image from the skeleton volume.
        For each (y, x), compute the mean z location of skeleton voxels (where self.volume > 0)
        and then take the logarithm (with +1 offset) to amplify depth differences.
        Returns a 2D array of log-scaled depth values.
        """
        Z, Y, X = self.volume.shape
        mask = (self.volume > 0)
        count = np.sum(mask, axis=0)  # shape (Y, X)
        z_indices = np.arange(Z).reshape(Z, 1, 1)
        sum_depth = np.sum(mask * z_indices, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_depth = np.divide(sum_depth, count, out=np.zeros_like(sum_depth, dtype=np.float32), where=(count != 0))
        depth_img = np.log(mean_depth + 1)
        return depth_img

    def capture_depth_encoded(self):
        """
        Generate the depth-encoded image of the skeleton.
        This method computes a 2D depth map from the 3D skeleton volume using a logarithmic scale,
        then maps the result through the 'plasma' colormap.
        If an ROI was selected, the image is cropped accordingly.
        The final image is saved as "<output_file>_depth_encoded.png".
        """
        depth_img = self.depth_encoded_skeleton()  # 2D array, shape (Y, X)

        if self.roi_polygon is not None:
            poly = np.array(self.roi_polygon)  # Assumed (y, x) coordinates
            mask = np.zeros(depth_img.shape, dtype=np.uint8)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=depth_img.shape)
            mask[rr, cc] = 1
            rows, cols = np.nonzero(mask)
            if rows.size > 0 and cols.size > 0:
                min_row = int(rows.min())
                max_row = int(rows.max())
                min_col = int(cols.min())
                max_col = int(cols.max())
                depth_img = depth_img[min_row:max_row + 1, min_col:max_col + 1]

        norm_obj = plt.Normalize(vmin=depth_img.min(), vmax=depth_img.max())
        plasma = plt.get_cmap("plasma")
        colored_depth = plasma(norm_obj(depth_img))

        depth_file = os.path.splitext(self.output_file)[0] + "_depth_encoded.png"
        fig, ax = plt.subplots(figsize=(8,6), facecolor='black')
        ax.imshow(colored_depth)
        ax.set_title("Depth Encoded Skeleton (Log Scale)", color="white", fontsize=14)
        ax.axis('off')
        fig.savefig(depth_file, bbox_inches='tight', facecolor='black')
        plt.close(fig)
        print(f"Depth encoded image saved to {depth_file}")
        return depth_file

    def capture_modalities_with_preset_view(self, branch_file, original_file, output_file=None):
        """
        Capture screenshots for skeleton, branch, and original volumes using the pre-set camera view and ROI.
        Apply the same ROI to all screenshots and combine them side-by-side.
        
        Parameters:
        -----------
        branch_file : str
            Path to the branch volume file
        original_file : str
            Path to the original volume file
        output_file : str, optional
            Path to save the output composite image. If None, uses self.output_file
            
        Returns:
        --------
        str
            Path to the saved composite image
        """
        # Update output file if provided
        if output_file:
            self.output_file = output_file
            
        # Load volumes and capture screenshots for all four modalities
        img_skel, img_branch, img_original, img_depth = self._capture_all_screenshots(branch_file, original_file)
        
        # Apply ROI filtering and cropping
        cropped_images, _ = self._apply_roi_to_images(img_skel, img_branch, img_original, img_depth)
        cropped_skel, cropped_branch, cropped_original, cropped_depth = cropped_images
        
        # Create temporary composite image (will be used by TopologicalViewpointAnalyzer and then deleted)
        composite_file = os.path.splitext(self.output_file)[0] + "_temp_composite.png"
        
        # Combine images horizontally
        composite = np.hstack([cropped_original, cropped_branch, cropped_skel, cropped_depth])
        
        # Save directly without creating a matplotlib figure (faster and cleaner)
        imageio.imwrite(composite_file, composite)
        
        return composite_file

def main():
    # Hardcoded base directory (adjust as needed).
    base_dir = r"C:\Users\gavst\Box\Box-Gdrive\Calico\scripts\2025-02-13_yeast_mitographs\event1_2024-10-22_13-14-25_\crop1_snout\crop1_nellie_out\nellie_necessities"
    skeleton_file = os.path.join(base_dir, "crop1.ome-im_skel.ome.tif")
    branch_file = os.path.join(base_dir, "crop1.ome-im_branch_label_reassigned.ome.tif")
    original_file = os.path.join(base_dir, "crop1.ome.ome.tif")
    output_file = os.path.join(base_dir, "final_view.png")
    # For a T,Z,Y,X volume (using the first time point), use projection_axis=0 (change if needed)
    selector = ViewpointSelector(skeleton_file, output_file, projection_axis=0)
    composite_file, depth_file = selector.run_workflow(branch_file, original_file)
    print("Workflow completed. Composite image saved as:", composite_file)
    print("Depth encoded image saved as:", depth_file)
    return composite_file, depth_file

if __name__ == "__main__":
    # Optional second argument: set to True ("true", "1", "yes") to display the final cropped image.
    display_output = True
    if len(sys.argv) > 2:
         display_output = sys.argv[2].lower() in ['true', '1', 'yes']
    cropped_file, depth_file = main()
    if display_output:
         try:
             img = imageio.imread(cropped_file)
             plt.figure(figsize=(8, 6))
             plt.imshow(img)
             plt.title("Cropped Final 3D View")
             plt.axis('off')
             plt.show()
         except Exception as e:
             print("Error displaying the cropped final image:", e) 