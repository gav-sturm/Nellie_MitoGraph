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
        # Capture screenshot for skeleton (using raw image display).
        img_skel = self.capture_screenshot(self.volume, layer_type='image')

        # Load branch and original volumes.
        branch = imread(branch_file)
        if branch.ndim == 4:
            branch = branch[0]
        original = imread(original_file)
        if original.ndim == 4:
            original = original[0]

        # Capture branch using labels so that its pre-assigned colors (from napari labels) are preserved.
        img_branch = self.capture_screenshot(branch, layer_type='labels')
        # Capture raw (original) image.
        img_original = self.capture_screenshot(original, layer_type='image')

        # Apply ROI as a filter: pixels outside the ROI will be set to zero, then get the BBOX
        if self.roi_polygon is None:
            print("No ROI selected; using full image for all modalities.")
            cropped_skel = img_skel
            cropped_branch = img_branch
            cropped_original = img_original
        else:
            poly = np.array(self.roi_polygon)  # Assumed (y, x) coordinates.
            # Create a binary mask from the polygon using the skeleton screenshot size.
            mask = np.zeros(img_skel.shape[:2], dtype=np.uint8)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=img_skel.shape[:2])
            mask[rr, cc] = 1

            # Resize branch and original screenshots to match the skeleton screenshot size.
            target_shape = img_skel.shape
            if img_branch.shape != target_shape:
                img_branch = resize(img_branch, target_shape, preserve_range=True, anti_aliasing=False).astype(img_branch.dtype)
            if img_original.shape != target_shape:
                img_original = resize(img_original, target_shape, preserve_range=True, anti_aliasing=False).astype(img_original.dtype)

            # Apply the mask to each modality (for RGB images, expand mask along the color axis).
            if img_skel.ndim == 3:
                filtered_skel = img_skel * mask[:, :, None]
                filtered_branch = img_branch * mask[:, :, None]
                filtered_original = img_original * mask[:, :, None]
            else:
                filtered_skel = img_skel * mask
                filtered_branch = img_branch * mask
                filtered_original = img_original * mask

            # Compute the bounding box from the mask (nonzero entries).
            rows, cols = np.nonzero(mask)
            min_row = int(rows.min())
            max_row = int(rows.max())
            min_col = int(cols.min())
            max_col = int(cols.max())

            cropped_skel = filtered_skel[min_row:max_row + 1, min_col:max_col + 1]
            cropped_branch = filtered_branch[min_row:max_row + 1, min_col:max_col + 1]
            cropped_original = filtered_original[min_row:max_row + 1, min_col:max_col + 1]

        # Compute the depth-encoded image from the skeleton volume.
        depth_img = self.depth_encoded_skeleton()  # 2D array, shape (Y, X)
        if self.roi_polygon is not None:
            poly = np.array(self.roi_polygon)  # Assumed (y, x) coordinates.
            mask_depth = np.zeros(depth_img.shape, dtype=np.uint8)
            rr, cc = sk_polygon(poly[:, 0], poly[:, 1], shape=depth_img.shape)
            mask_depth[rr, cc] = 1
            rows_d, cols_d = np.nonzero(mask_depth)
            if rows_d.size > 0 and cols_d.size > 0:
                min_row_d = int(rows_d.min())
                max_row_d = int(rows_d.max())
                min_col_d = int(cols_d.min())
                max_col_d = int(cols_d.max())
                depth_img = depth_img[min_row_d:max_row_d+1, min_col_d:max_col_d+1]

        # Map the depth image using the plasma colormap.
        norm_obj_depth = plt.Normalize(vmin=depth_img.min(), vmax=depth_img.max())
        plasma = plt.get_cmap("plasma")
        colored_depth = plasma(norm_obj_depth(depth_img))
        # Optionally, one might convert the colored depth to a suitable uint8 image if needed.
        cropped_depth = colored_depth

        # Combine the cropped images with the order: raw (original), branch labels, skeleton, then depth-encoded.
        composite = np.hstack([cropped_original, cropped_branch, cropped_skel, cropped_depth])

        composite_file = os.path.splitext(self.output_file)[0] + "_composite.png"
        # Create a figure with a black background.
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='black')
        ax.imshow(composite)
        ax.set_title("Composite 3D View", color="white", fontsize=14)
        ax.axis('off')
        # Save the composite figure with a black background.
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