# Nellie MitoGraph Visualization Tool

A modular system for selecting viewpoints and creating visualizations of 3D mitochondrial volumes with multiple modalities.


https://github.com/user-attachments/assets/28fd4899-5e17-47fd-bd35-ad6a88b33216



## Overview

The Nellie MitoGraph Visualization Tool is an extension of Nellie automated organell tracking specically desinged to generate topological graphs of the mitochondrial network, this pipeline can handle:

1. Loading and visualizing 3D/4D volumetric data of mitochondrial networks
2. Interactive selection of viewpoints for visualization
3. ROI selection and processing
4. Capturing screenshots of different data modalities
5. Creating composite visualizations with multiple data modalities
6. Processing timeseries data and creating GIF animations
7. Tracking nodes across frames in timeseries data

## Modules

The system is organized into the following modules:

- **volume_loader.py**: Handles loading and preprocessing of different volume types
- **viewpoint_selector.py**: Manages interactive viewpoint selection
- **roi_selector.py**: Handles selection and processing of Regions of Interest (ROIs)
- **screenshot_manager.py**: Manages capturing and processing screenshots
- **timepoint_manager.py**: Handles selection and parsing of timepoints for timeseries data
- **composite_image_creator.py**: Creates composite images from multiple screenshots
- **gif_generator.py**: Creates GIF animations from frame images
- **node_tracker.py**: Tracks nodes across frames in timeseries data
- **Nellie_MitoGraph_run.py**: Main entry point that coordinates all components

## Requirements

- Python 3.7+
- Required packages are listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
python Nellie_MitoGraph_run.py \
    --base-dir /path/to/data \
    --skeleton-file skeleton.ome.tif \
    --branch-file branch_label.ome.tif \
    --original-file original.ome.tif \
    --node-edge-file pixel_class.ome.tif \
    --obj-label-file obj_label.ome.tif \
    --output-file output.png \
    --display \
    --timepoints "0,5,10-20,30" \
    --intensity-percentile 50
```

### Python API

```python
from Nellie_MitoGraph_run import ModularViewpointSelector

# Initialize
selector = ModularViewpointSelector(
    skeleton_file="/path/to/skeleton.ome.tif",
    output_file="/path/to/output.png",
    intensity_percentile=50
)

# Run workflow
output_file = selector.run_workflow(
    branch_file="/path/to/branch_label.ome.tif",
    original_file="/path/to/original.ome.tif",
    node_edge_file="/path/to/pixel_class.ome.tif",
    obj_label_file="/path/to/obj_label.ome.tif",
    timepoints="0,5,10-20,30"
)

print(f"Output saved to: {output_file}")
```

## Interactive Workflow

1. **Viewpoint Selection**: A 3D viewer will open showing the skeleton volume. Adjust the view to your liking and press 'v' to capture the view.
2. **ROI Selection**: A 2D viewer will open showing the captured screenshot. Draw an ROI polygon to define the region of interest.
3. **Timepoint Selection**: If working with a timeseries, you'll be prompted to select which timepoints to process.
4. **Processing**: The system will process the selected timepoints, capturing screenshots of all modalities and creating composite images.
5. **Output**: For a single timepoint, a composite image will be created. For a timeseries, a GIF animation will be created.

## Examples

### Single Timepoint

For a single timepoint, the system will create a composite image with all modalities arranged in a grid layout, including:
- Raw Intensity Volume
- Branch Labels
- Object Labels
- Depth-Encoded View
- Node-Edge Network View
- Projected Topological Graph
- Cocentric Topological Graph

### Timeseries

For a timeseries, the system will create a GIF animation showing the composite views for each selected timepoint, with a timestamp indicating the current timepoint.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
3. **Timepoint Selection**: If working with a timeseries, you'll be prompted to select which timepoints to process.
4. **Processing**: The system will process the selected timepoints, capturing screenshots of all modalities and creating composite images.
5. **Output**: For a single timepoint, a composite image will be created. For a timeseries, a GIF animation will be created.

## Examples

### Single Timepoint

For a single timepoint, the system will create a composite image with all modalities arranged in a grid layout, including:
- Raw Intensity Volume
- Branch Labels
- Object Labels
- Depth-Encoded View
- Node-Edge Network View
- Projected Topological Graph
- Cocentric Topological Graph

  
### Timeseries

For a timeseries, the system will create a GIF animation showing the composite views for each selected timepoint, with a timestamp indicating the current timepoint.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
