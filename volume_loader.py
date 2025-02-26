"""
VolumeLoader Module

This module provides the VolumeLoader class for loading and preprocessing 3D volumetric data.
"""

import os
import numpy as np
from tifffile import imread

class VolumeLoader:
    """
    Handles loading and preprocessing of different volume types.
    
    This class is responsible for loading skeleton, branch, original, and other volume types,
    and provides methods for accessing and preprocessing these volumes.
    """
    
    def __init__(self, volume_file, intensity_percentile=50):
        """
        Initialize the VolumeLoader with a path to a volume file.
        
        Parameters:
        -----------
        volume_file : str
            Path to the volume file
        intensity_percentile : int, optional
            Percentile cutoff for intensity thresholding (0-100, default: 50)
        """
        self.volume_file = volume_file
        self.intensity_percentile = intensity_percentile
        
        # Load the full volume data
        self.full_volume = imread(volume_file)
        
        # Check if we're working with 4D data
        self.is_timeseries = self.full_volume.ndim == 4
        self.num_timepoints = self.full_volume.shape[0] if self.is_timeseries else 1
        
        # For viewpoint selection, use only the first timepoint
        if self.is_timeseries:
            self.volume = self.full_volume[0].copy()  # First timepoint for UI interactions
        else:
            self.volume = self.full_volume  # Use the whole volume if it's 3D
    
    def load_volume_for_timepoint(self, timepoint=0):
        """
        Load the volume data for a specific timepoint.
        
        Parameters:
        -----------
        timepoint : int, optional
            Timepoint to load (default: 0)
            
        Returns:
        --------
        ndarray
            Volume data for the specified timepoint
        """
        if self.is_timeseries:
            if timepoint < 0 or timepoint >= self.num_timepoints:
                raise ValueError(f"Timepoint {timepoint} is out of range (0-{self.num_timepoints-1})")
            return self.full_volume[timepoint].copy()
        return self.volume.copy()
    
    def preprocess_volume(self, volume, apply_threshold=True):
        """
        Preprocess a volume by applying thresholding.
        
        Parameters:
        -----------
        volume : ndarray
            Volume data to preprocess
        apply_threshold : bool, optional
            Whether to apply intensity thresholding (default: True)
            
        Returns:
        --------
        ndarray
            Preprocessed volume
        """
        processed_volume = volume.copy()
        
        if apply_threshold:
            # Calculate threshold value based on user-specified percentile of non-zero values
            non_zero_values = processed_volume[processed_volume > 0]
            if len(non_zero_values) > 0:
                threshold = np.percentile(non_zero_values, self.intensity_percentile)
                # Set values below threshold to zero
                processed_volume[processed_volume < threshold] = 0
        
        return processed_volume
    
    @staticmethod
    def load_multiple_volumes(files_dict):
        """
        Load multiple volumes from a dictionary of file paths.
        
        Parameters:
        -----------
        files_dict : dict
            Dictionary mapping volume names to file paths
            
        Returns:
        --------
        dict
            Dictionary mapping volume names to VolumeLoader instances
        """
        volumes = {}
        for name, path in files_dict.items():
            if path and os.path.exists(path):
                volumes[name] = VolumeLoader(path)
        return volumes
