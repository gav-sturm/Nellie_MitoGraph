"""
NodeTracker Module

This module provides classes for tracking nodes across frames in 3D volumes.
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class NodeTracker:
    """
    Tracks nodes across frames using a Kalman filter.
    
    This class is responsible for tracking a single node across frames, keeping its history,
    and predicting its position in future frames.
    """
    
    def __init__(self, initial_pos, node_id, frame_idx):
        """
        Initialize a NodeTracker.
        
        Parameters:
        -----------
        initial_pos : tuple
            Initial position of the node (z, y, x)
        node_id : int
            ID of the node
        frame_idx : int
            Frame index where the node was first seen
        """
        self.id = node_id
        self.kalman_filter = KalmanFilter(dim_x=4, dim_z=3)
        # Initialize state transition matrix
        self.kalman_filter.F = np.array([[1, 0, 0, 1],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 1],
                                        [0, 0, 0, 1]])
        # Initialize measurement matrix
        self.kalman_filter.H = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0]])
        # Initial state
        self.kalman_filter.x = np.array([initial_pos[0], initial_pos[1], initial_pos[2], 0])
        self.last_seen = frame_idx
        self.history = [initial_pos]


class NodeTrackerManager:
    """
    Manages multiple NodeTrackers and handles node tracking across frames.
    
    This class is responsible for creating and updating NodeTrackers,
    matching nodes between frames, and assigning consistent IDs.
    """
    
    def __init__(self, max_unseen_frames=3):
        """
        Initialize the NodeTrackerManager.
        
        Parameters:
        -----------
        max_unseen_frames : int, optional
            Number of frames a node can be unseen before it's removed (default: 3)
        """
        self.trackers = []
        self.next_id = 1
        self.max_unseen_frames = max_unseen_frames
    
    def track_nodes(self, current_points, frame_idx):
        """
        Match nodes between frames with Hungarian algorithm.
        
        Parameters:
        -----------
        current_points : ndarray
            Array of current node positions
        frame_idx : int
            Current frame index
            
        Returns:
        --------
        list
            List of node IDs
        """
        # Remove stale trackers first
        self.trackers = [t for t in self.trackers 
                        if (frame_idx - t.last_seen) <= self.max_unseen_frames]
        
        if not self.trackers:  # First frame case
            node_ids = list(range(self.next_id, self.next_id + len(current_points)))
            for point, nid in zip(current_points, node_ids):
                tracker = NodeTracker(tuple(point), nid, frame_idx)
                self.trackers.append(tracker)
            self.next_id += len(current_points)
            return node_ids

        # Create cost matrix (trackers vs current points)
        cost_matrix = np.zeros((len(self.trackers), len(current_points)))
        for i, tracker in enumerate(self.trackers):
            predicted_pos = tracker.kalman_filter.x[:3]
            for j, point in enumerate(current_points):
                cost_matrix[i,j] = np.linalg.norm(predicted_pos - point)
        
        # Apply Hungarian algorithm with threshold
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r,c in zip(row_ind, col_ind):
            if cost_matrix[r,c] < 20:  # Max allowed movement between frames
                matches.append((r,c))
        
        # Update matched trackers
        node_ids = [-1]*len(current_points)
        for r,c in matches:
            tracker = self.trackers[r]
            tracker.kalman_filter.predict()
            tracker.kalman_filter.update(current_points[c])
            tracker.last_seen = frame_idx
            tracker.history.append(current_points[c])
            node_ids[c] = tracker.id
        
        # Handle unmatched current points (new nodes)
        unmatched_current = set(range(len(current_points))) - set(c for _,c in matches)
        for c in unmatched_current:
            new_id = self.next_id
            tracker = NodeTracker(current_points[c], new_id, frame_idx)
            self.trackers.append(tracker)
            node_ids[c] = new_id
            self.next_id += 1
        
        return node_ids