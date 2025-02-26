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
        
        viewer = napari.Viewer(ndisplay=2)
        viewer.add_image(image, name="Image")
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
                    # Reduced delay from 0.5 to 0.2 seconds
                    threading.Timer(0.2, viewer.close).start()
        
        # Connect the callback to the data change event
        shapes_layer.events.data.connect(on_data_change)
        shapes_layer.events.mode.connect(on_data_change)
        
        # Run the viewer
        napari.run()
        
        # In case the user manually closed the viewer without completing a polygon
        if not self.roi_polygon and len(shapes_layer.data) > 0:
            self.roi_polygon = shapes_layer.data[0]
            print("ROI captured.")
        elif not self.roi_polygon:
            print("No ROI drawn; using full image as ROI.")
        
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