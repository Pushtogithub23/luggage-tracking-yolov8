# YOLOv8 Luggage Tracking in Metro Video Footage

This project demonstrates how to use **YOLOv8** to track specific types of luggage (backpack, handbag, suitcase) in a video using OpenCV. The code tracks these objects in a video of people moving in a metro station, displays their locations, and counts on the video frames.

![tracking_luggage_with_YOLOv8x](https://github.com/user-attachments/assets/bd76cc2e-0248-4feb-851e-d916ec088ce9)


## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Customization](#customization)

## Overview

The code performs object detection and tracking using **YOLOv8**. It detects the following classes of objects:
- Backpack
- Handbag
- Suitcase

The bounding boxes and labels are displayed in real time along with the object count on the video frames. The processed video is saved to an output file.


## Requirements

Before running the code, ensure you have the following installed:

- Python 3.7 or higher
- [OpenCV](https://opencv.org/) (for video processing and visualization)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolov8.md) (for object detection and tracking)
- Additional utilities:
  - Custom utility functions like `draw_rounded_rect` and `draw_text_with_bg` can be accessed from `utils.py`.


## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Pushtogithub23/luggage-tracking-yolov8.git
   cd luggage-tracking-yolov8
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should include:
   ```txt
   opencv-python
   ultralytics
   torch
   ```

3. Download YOLOv8 pre-trained weights:

   - You can download the weights from the [Ultralytics YOLOv8 Model Zoo](https://github.com/ultralytics/ultralytics). Make sure to place the weights file (e.g., `yolov8x.pt`) in the root directory or update the path in the code.


## Usage

1. Prepare the input video. Place your video in the `DATA/INPUTS/` folder or specify the correct path in the code.
   
2. Run the tracking script:

   ```bash
   python main.py
   ```

   This will:
   - Load the YOLOv8 model.
   - Process the video, detect, and track luggage in real-time.
   - Display the tracking results with bounding boxes and save the output video to `DATA/OUTPUTS/`.

3. During execution, the video will be displayed with bounding boxes, and pressing the `p` key will pause or stop the processing.


## File Structure

```plaintext
├── DATA
│   ├── INPUTS
│   │   └── people_in_metro.mp4         # Input video file
│   ├── OUTPUTS
│   │   └── tracking_luggage_with_YOLOv8x.mp4  # Output video file
├── utils.py                             # Utility functions for drawing
├── main.py                              # Main script to run object detection and tracking
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation

```

## Customization

You can modify the following aspects of the code to suit your requirements:

### 1. **Target Classes**
   You can customize the classes YOLOv8 detects and tracks by modifying the `target_classes` dictionary. This dictionary maps class names to specific colours for the bounding boxes.

   ```python
   target_classes = {
       'backpack': (0, 255, 0),
       'handbag': (0, 0, 255),
       'suitcase': (255, 0, 0)
   }
   ```

### 2. **Confidence Threshold**
   The confidence threshold for displaying bounding boxes can be adjusted in the `draw_detections` function. Currently, it only shows bounding boxes for detections with a confidence score greater than `0.6`.

   ```python
   if class_name in target_classes and score > 0.6:
   ```

### 3. **Input/Output Paths**
   The paths for input and output videos are defined as follows:

   ```python
   video_path = "DATA/INPUTS/people_in_metro.mp4"
   output_path = "DATA/OUTPUTS/tracking_luggage_with_YOLOv8x.mp4"
   ```

   You can change these paths to point to other videos or output locations.


## Examples

Here’s how the bounding boxes and tracked objects are displayed in the video:

- The detected objects (backpack, handbag, suitcase) will have bounding boxes drawn in their respective colours (green, red, and blue).
- Each object is labelled with its class name and a unique ID.
- The total count of each object type is displayed in the top-left corner of the video.

![tracking_luggage_with_YOLOv8x](https://github.com/user-attachments/assets/bd76cc2e-0248-4feb-851e-d916ec088ce9)

---

