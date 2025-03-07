"""
ROISelector Module

This module provides the ROISelector class for selecting and processing regions of interest.
"""

import threading
import numpy as np
import napari
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize

class ROISelector:
    """
    Handles selection and processing of Regions of Interest (ROIs).
    
    This class is responsible for allowing the user to draw an ROI on a 2D image,
    and for applying this ROI to images for cropping and filtering.
    """
    
    def __init__(self):
        """Initialize the ROISelector."""
        self.roi_polygon = None
        self.original_image_shape = None
        self.roi_mask = None
    def select_roi(self, image):
        """
        Display an image in a 2D viewer and let the user draw an ROI polygon.
        
        Parameters:
        -----------
        image : ndarray
            Image to display for ROI selection
            
        Returns:
        --------
        ndarray or None
            ROI polygon coordinates, or None if no ROI was drawn
        """
        if image is None:
            print("No image available for ROI selection.")
            return None
        
        # Store the original image shape for scaling the ROI later
        self.original_image_shape = image.shape[:2]
        
        # Handle the input image - remove alpha channel if present
        if image.ndim == 3 and image.shape[2] == 4:
            # Remove alpha channel
            mip = image[..., :3]
        else:
            mip = image
        
        print(f"DEBUG: MIP shape: {mip.shape}")
        
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(mip, name="Image")
        shapes_layer = viewer.add_shapes(name="ROI", shape_type="polygon")
        
        print("Draw an ROI on the image.")
        print("Double-click to complete the polygon - viewer will close automatically.")
        
        # Create a callback function to detect when a polygon is completed
        def on_data_change(event):
            # Check if we have at least one complete polygon
            if len(shapes_layer.data) > 0 and len(shapes_layer.data[0]) >= 3:
                # Check if the last point is close to the first point (polygon completed)
                # This happens when user double-clicks to finish the polygon
                if shapes_layer.mode == 'pan_zoom':  # Mode changes to pan_zoom after completion
                    print("ROI polygon completed. Closing viewer...")
                    # Store the polygon before closing
                    self.roi_polygon = shapes_layer.data[0]
                    # print(f"DEBUG: ROI polygon shape: {self.roi_polygon.shape}")
                    
                    # Create a 2D mask from the polygon
                    h, w = mip.shape[:2]
                    roi_mask = np.zeros((h, w), dtype=bool)
                    rr, cc = sk_polygon(self.roi_polygon[:, 0], self.roi_polygon[:, 1], shape=(h, w))
                    roi_mask[rr, cc] = True
                    self.roi_mask = roi_mask
                    
                    # print(f"DEBUG: ROI mask shape: {self.roi_mask.shape}, non-zero: {np.count_nonzero(self.roi_mask)}")
                    # Reduced delay from 0.5 to 0.2 seconds
                    threading.Timer(0.2, viewer.close).start()
        
        # Connect the callback to the data change event
        shapes_layer.events.data.connect(on_data_change)
        shapes_layer.events.mode.connect(on_data_change)
        viewer.bind_key('r', on_data_change)
        
        # Run the viewer
        napari.run()
        
        # In case the user manually closed the viewer without completing a polygon
        if not self.roi_polygon and len(shapes_layer.data) > 0:
            self.roi_polygon = shapes_layer.data[0]
            # print(f"DEBUG: ROI polygon shape (after close): {self.roi_polygon.shape}")
            
            # Create a 2D mask from the polygon
            h, w = mip.shape[:2]
            roi_mask = np.zeros((h, w), dtype=bool)
            rr, cc = sk_polygon(self.roi_polygon[:, 0], self.roi_polygon[:, 1], shape=(h, w))
            roi_mask[rr, cc] = True
            self.roi_mask = roi_mask
            
            # print(f"DEBUG: ROI mask shape (after close): {self.roi_mask.shape}, non-zero: {np.count_nonzero(self.roi_mask)}")
            print("ROI captured.")
        elif not self.roi_polygon:
            print("No ROI drawn; using full image as ROI.")
            # Create a full-image ROI mask
            h, w = mip.shape[:2]
            self.roi_mask = np.ones((h, w), dtype=bool)
            # print(f"DEBUG: Created full-image ROI mask: {self.roi_mask.shape}")
        
        return self.roi_polygon
    
    def apply_roi_to_image(self, img, roi_polygon=None):
        """
        Apply an ROI polygon to an image and return the filtered image.
        
        Parameters:
        -----------
        img : ndarray
            Image to filter
        roi_polygon : ndarray, optional
            ROI polygon coordinates (default: self.roi_polygon)
            
        Returns:
        --------
        tuple
            (filtered_img, roi_bbox) where roi_bbox is (min_row, max_row, min_col, max_col)
        """
        if roi_polygon is None:
            roi_polygon = self.roi_polygon
        
        if roi_polygon is None or self.original_image_shape is None:
            # No ROI defined, return the original image with a full-image bounding box
            h, w = img.shape[:2]
            bbox = (0, h-1, 0, w-1)
            return img, bbox
        
        # Scale the ROI polygon if the current image dimensions differ from the original
        if img.shape[:2] != self.original_image_shape:
            # Calculate scaling factors
            scale_y = img.shape[0] / self.original_image_shape[0]
            scale_x = img.shape[1] / self.original_image_shape[1]
            
            # Apply scaling to the polygon
            scaled_polygon = roi_polygon.copy()
            scaled_polygon[:, 0] = scaled_polygon[:, 0] * scale_y
            scaled_polygon[:, 1] = scaled_polygon[:, 1] * scale_x
        else:
            scaled_polygon = roi_polygon
        
        # Convert the polygon to a mask
        rr, cc = sk_polygon(scaled_polygon[:, 0], scaled_polygon[:, 1], shape=img.shape[:2])
        mask = np.zeros(img.shape[:2], dtype=bool)
        mask[rr, cc] = True
        
        # Get the bounding box of the ROI
        rows, cols = np.nonzero(mask)
        if len(rows) > 0 and len(cols) > 0:
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            bbox = (min_row, max_row, min_col, max_col)
        else:
            h, w = img.shape[:2]
            bbox = (0, h-1, 0, w-1)
        
        # Apply mask to image
        filtered = img.copy()
        
        # Handle RGB images differently from grayscale
        if img.ndim == 3 and img.shape[2] in [3, 4]:  # RGB or RGBA
            # For RGB/RGBA images, we need to apply the mask to each channel
            for c in range(img.shape[2]):
                channel = filtered[:, :, c]
                # Set pixels outside the mask to black (0)
                channel[~mask] = 0
        else:
            # For grayscale images, simply apply the mask
            filtered[~mask] = 0

        # crop down to the bounding box
        filtered = self.crop_to_bbox(filtered, bbox)
        
        return filtered, bbox
    
    def crop_to_bbox(self, img, bbox):
        """
        Crop an image to a bounding box.
        
        Parameters:
        -----------
        img : ndarray
            Image to crop
        bbox : tuple
            Bounding box (min_row, max_row, min_col, max_col)
            
        Returns:
        --------
        ndarray
            Cropped image
        """
        min_row, max_row, min_col, max_col = bbox
        return img[min_row:max_row+1, min_col:max_col+1]
    
    def ensure_consistent_dimensions(self, images):
        """
        Ensure all images have the same dimensions.
        
        Parameters:
        -----------
        images : list
            List of images to make consistent
            
        Returns:
        --------
        list
            List of images with consistent dimensions
        """
        if not images:
            return []
        
        # Find the max dimensions, handling potential invalid/corrupted images
        valid_images = [img for img in images if img is not None and img.size > 0 and not np.isnan(img).any()]
        
        if not valid_images:
            print("Warning: No valid images found for ensuring consistent dimensions.")
            return images
        
        max_h = max(img.shape[0] for img in valid_images)
        max_w = max(img.shape[1] for img in valid_images)
        
        # Validate max dimensions to avoid NaN errors
        if np.isnan(max_h) or np.isnan(max_w) or max_h <= 0 or max_w <= 0:
            print(f"Warning: Invalid dimensions detected (h={max_h}, w={max_w}). Using fallback values.")
            max_h = max(100, max(img.shape[0] for img in valid_images if not np.isnan(img.shape[0])))
            max_w = max(100, max(img.shape[1] for img in valid_images if not np.isnan(img.shape[1])))
        
        # Resize all images to the same dimensions
        resized_images = []
        for img in images:
            # Skip invalid images
            if img is None or img.size == 0 or np.isnan(img).any():
                print("Warning: Skipping invalid image during resize operation.")
                # Provide a placeholder instead
                placeholder = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                placeholder[:, :, 0] = 255  # Make it red to indicate error
                resized_images.append(placeholder)
                continue
                
            # Handle valid images
            if img.shape[:2] != (max_h, max_w):
                try:
                    resized = resize(img, (max_h, max_w), 
                                   preserve_range=True, anti_aliasing=True).astype(img.dtype)
                    resized_images.append(resized)
                except Exception as e:
                    print(f"Error resizing image: {e}")
                    # Provide a placeholder instead
                    placeholder = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                    placeholder[:, :, 0] = 255  # Make it red to indicate error
                    resized_images.append(placeholder)
            else:
                resized_images.append(img)
        
        return resized_images
    
    def apply_roi_to_volume(self, volume_data, viewpoint_selector=None):
        """
        Apply the ROI to a 3D volume.
        
        Parameters:
        -----------
        volume_data : ndarray
            3D volume data
        viewpoint_selector : ViewpointSelector, optional
            ViewpointSelector instance with camera parameters
            
        Returns:
        --------
        ndarray
            Filtered volume
        """
        if self.roi_polygon is None or volume_data is None:
            print("No ROI or volume data available.")
            return volume_data
        
        # Get the volume shape
        volume_shape = volume_data.shape
        
        # Get the camera parameters from the viewpoint selector
        if viewpoint_selector is not None and hasattr(viewpoint_selector, 'angles') and hasattr(viewpoint_selector, 'center'):
            angles = viewpoint_selector.angles
            center = viewpoint_selector.center
            print(f"DEBUG: angles tuple contains: {angles}")
            print(f"DEBUG: center is: {center}")
        else:
            print("No viewpoint selector provided or missing camera parameters.")
            # Default to a simple z-projection approach
            return self._apply_simple_z_projection_mask(volume_data)
        
        # Create a 3D mask using the ROI and camera parameters
        # We'll use a simpler approach that works with the current napari API
        return self._apply_simple_z_projection_mask(volume_data)

    def _apply_simple_z_projection_mask(self, volume_data):
        """
        Apply a simple z-projection mask to the volume.
        This method extends the 2D ROI mask along the z-axis.
        
        Parameters:
        -----------
        volume_data : ndarray
            3D volume data
            
        Returns:
        --------
        ndarray
            Filtered volume
        """
        if self.roi_mask is None:
            print("No ROI mask available.")
            return volume_data
        
        # Get the volume shape
        z_dim, y_dim, x_dim = volume_data.shape
        
        # We need to resize the 2D ROI mask to match the y,x dimensions of the volume
        from skimage.transform import resize
        
        # Resize the ROI mask to match the volume's y,x dimensions
        resized_mask = resize(self.roi_mask.astype(float), (y_dim, x_dim), 
                             order=0, preserve_range=True, anti_aliasing=False).astype(bool)
        
        print(f"DEBUG: Original ROI mask shape: {self.roi_mask.shape}")
        print(f"DEBUG: Resized ROI mask shape: {resized_mask.shape}")
        print(f"DEBUG: Volume shape: {volume_data.shape}")
        
        # Broadcast the 2D mask to 3D
        mask_3d = np.broadcast_to(resized_mask, volume_data.shape)
        
        # Apply the mask to the volume
        filtered_volume = np.where(mask_3d, volume_data, 0)
        
        return filtered_volume

    def resize_image(self, image, target_shape):
        """
        Resize a 2D image to match the target dimensions.
        
        Parameters:
        -----------
        image : ndarray
            Image to resize (2D or 3D with channels)
        target_shape : tuple
            Target shape - can be (z, y, x), (y, x), or (y, x, channels)
            Only the height and width values are used
            
        Returns:
        --------
        ndarray
            Resized image
        """
        if image is None:
            # Create a placeholder image with the target dimensions
            if len(target_shape) == 3 and target_shape[2] <= 4:  # (h, w, channels)
                return np.zeros(target_shape, dtype=np.uint8)
            elif len(target_shape) == 3:  # (z, y, x)
                return np.zeros((target_shape[1], target_shape[2], 3), dtype=np.uint8)
            else:  # (h, w)
                return np.zeros((*target_shape, 3), dtype=np.uint8)
        
        # Convert image to numpy array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Extract target height and width based on the shape format
        if len(target_shape) == 3:
            if target_shape[2] <= 4:  # (h, w, channels)
                target_h, target_w = target_shape[0], target_shape[1]
            else:  # (z, y, x)
                target_h, target_w = target_shape[1], target_shape[2]
        else:  # (h, w)
            target_h, target_w = target_shape
        
        # Print debug information
        print(f"Resizing image from {image.shape} to ({target_h}, {target_w})")
        
        # Use skimage resize with anti-aliasing for better quality
        from skimage.transform import resize
        resized = resize(image, (target_h, target_w), 
                       preserve_range=True, anti_aliasing=True)
        
        # Convert back to original dtype
        return resized.astype(image.dtype)