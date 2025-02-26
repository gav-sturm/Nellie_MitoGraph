"""
TimepointManager Module

This module provides the TimepointManager class for selecting and managing timepoints
in timeseries data.
"""

class TimepointManager:
    """
    Manages selection and parsing of timepoints for timeseries data.
    
    This class is responsible for allowing the user to select timepoints from a timeseries,
    and for parsing timepoint strings into lists of timepoint indices.
    """
    
    def __init__(self, num_timepoints):
        """
        Initialize the TimepointManager.
        
        Parameters:
        -----------
        num_timepoints : int
            Total number of timepoints in the timeseries
        """
        self.num_timepoints = num_timepoints
        self.selected_timepoints = None
        
    @property
    def is_timeseries(self):
        """Check if this is a timeseries (more than one timepoint)."""
        return self.num_timepoints > 1
    
    def select_timepoints(self):
        """
        Let the user select which timepoints to process from the timeseries.
        
        Returns:
        --------
        list
            List of selected timepoint indices
        """
        if self.num_timepoints <= 1:
            print("Not a timeseries or only one timepoint available.")
            return [0]
        
        print(f"\nTimeseries contains {self.num_timepoints} timepoints (0-{self.num_timepoints-1}).")
        print("Enter timepoints to process in one of these formats:")
        print("  - Individual timepoints separated by commas: 0,5,10,15")
        print("  - Range of timepoints: 0-20")
        print("  - Combination: 0,5,10-20,25,30-40")
        print("  - 'all' to process all timepoints")
        
        while True:
            selection = input("Enter timepoints to process: ").strip().lower()
            
            if selection == 'all':
                return list(range(self.num_timepoints))
            
            try:
                # Parse the input string to extract timepoints
                selected_timepoints = []
                for part in selection.split(','):
                    if '-' in part:
                        # Handle range
                        start, end = map(int, part.split('-'))
                        if start < 0 or end >= self.num_timepoints:
                            raise ValueError(f"Range {start}-{end} outside valid timepoints (0-{self.num_timepoints-1})")
                        selected_timepoints.extend(range(start, end + 1))
                    else:
                        # Handle individual timepoint
                        t = int(part)
                        if t < 0 or t >= self.num_timepoints:
                            raise ValueError(f"Timepoint {t} outside valid range (0-{self.num_timepoints-1})")
                        selected_timepoints.append(t)
                
                # Remove duplicates and sort
                selected_timepoints = sorted(set(selected_timepoints))
                
                if not selected_timepoints:
                    print("No valid timepoints selected. Please try again.")
                    continue
                
                print(f"Selected {len(selected_timepoints)} timepoints: {selected_timepoints}")
                self.selected_timepoints = selected_timepoints
                return selected_timepoints
                
            except ValueError as e:
                print(f"Error: {e}. Please try again.")
    
    def parse_timepoints_string(self, timepoints_str):
        """
        Parse a timepoints string from command line args.
        
        Parameters:
        -----------
        timepoints_str : str
            Timepoints string (e.g. "0,5,10-20,30" or "all")
            
        Returns:
        --------
        list
            List of timepoint indices
        """
        if timepoints_str.lower() == 'all':
            return list(range(self.num_timepoints))
        
        selected_timepoints = []
        try:
            for part in timepoints_str.split(','):
                if '-' in part:
                    # Handle range
                    start, end = map(int, part.split('-'))
                    if start < 0 or end >= self.num_timepoints:
                        print(f"Warning: Range {start}-{end} outside valid timepoints (0-{self.num_timepoints-1})")
                        # Clip to valid range
                        start = max(0, start)
                        end = min(self.num_timepoints-1, end)
                    selected_timepoints.extend(range(start, end + 1))
                else:
                    # Handle individual timepoint
                    t = int(part)
                    if t < 0 or t >= self.num_timepoints:
                        print(f"Warning: Timepoint {t} outside valid range (0-{self.num_timepoints-1})")
                        continue
                    selected_timepoints.append(t)
            
            # Remove duplicates and sort
            selected_timepoints = sorted(set(selected_timepoints))
            
            if not selected_timepoints:
                print("No valid timepoints selected. Using all timepoints.")
                return list(range(self.num_timepoints))
            
            print(f"Selected {len(selected_timepoints)} timepoints: {selected_timepoints}")
            self.selected_timepoints = selected_timepoints
            return selected_timepoints
        except ValueError as e:
            print(f"Error parsing timepoints string: {e}. Using all timepoints.")
            return list(range(self.num_timepoints))