Purpose:
--------
This script captures video frames from an external camera (Camera 2), detects words in the frame using the EAST
text detector, crops each word into individual letters, and saves each letter as a 32x32 image suitable for
feeding into a GRU model for character recognition.

Key Features:
-------------
1. Uses OpenCV’s DNN module to load and run the EAST text detector.
2. Dynamically detects word bounding boxes in real-time when the user presses SPACE.
3. Adds padding and scaling to each word box to avoid clipping letters at edges.
4. Crops each word into individual letters using contour detection, adding letter-level padding for clarity.
5. Resizes each letter crop to 32x32 for uniform input size for downstream GRU recognition.
6. Sorts words top-to-bottom and left-to-right, and letters left-to-right, preserving natural reading order.
7. Provides live visual feedback by drawing word boxes on the camera feed.
8. Saves letters sequentially into a folder for later processing.

How It Works:
--------------
1. **Camera Input**
   - Captures frames from an external camera (Camera 2) using OpenCV VideoCapture.
   - Each frame is copied for visualization and processing.

2. **EAST Text Detection**
   - The frame is resized to the input dimensions of the EAST model (e.g., 320x320).
   - A blob is created from the frame and fed into the EAST network.
   - The EAST network outputs two things:
     a. Scores: confidence for each pixel belonging to text
     b. Geometry: bounding box coordinates and rotation angles

3. **Decode Predictions**
   - The `decode_predictions()` function converts the raw EAST outputs into bounding boxes with (startX, startY, endX, endY) coordinates.
   - Boxes with confidence below a threshold are ignored.
   - Non-Maximum Suppression (NMS) is applied to remove overlapping boxes.

4. **Word Box Adjustments**
   - Each detected box is scaled to the original frame size.
   - Additional padding (`PAD_X`, `PAD_Y`) and scaling (`SCALE`) are applied so that letters are not clipped at the edges.
   - Boxes are stored in `last_boxes` for visual feedback.

5. **Sorting Boxes**
   - The `sort_boxes()` function ensures words are read in natural order:
     - Top-to-bottom: words on higher rows are processed first
     - Left-to-right: words in the same row are sorted by x-coordinate

6. **Letter Cropping**
   - Each word image is converted to grayscale and thresholded (binary inverse).
   - Contours are found, representing individual letters.
   - Small noise contours are ignored.
   - Letter-level padding is applied (`letter_pad`) to slightly "zoom out" and capture the full letter.
   - Letters are sorted left-to-right within each word.
   - Each letter is resized to 32x32 pixels and saved sequentially.

7. **User Interaction**
   - Press SPACE: perform text detection and save letter crops from the current frame.
   - Press Q: quit the program.
   - Previous word boxes are drawn for live visual feedback.

Why It Works:
--------------
- EAST is designed for real-time scene text detection. It predicts word regions accurately even if text is rotated or of varying sizes.
- Padding + scaling ensures letters at the edges of words are not clipped, which improves downstream letter recognition.
- Sorting preserves reading order, which is important for reconstructing words correctly when feeding letters to a GRU.
- Resizing all letters to 32x32 ensures consistent input dimensions for the GRU model, simplifying training and inference.

Use Case:
---------
- This pipeline can be used in a real-time text recognition system, e.g., reading signs, handwritten notes, or images.
- The cropped letters are now ready to be used as input for a GRU-based character recognition model.
- The system is modular: you can replace EAST with any other text detector, or integrate the letter crops directly into memory for FPGA/GRU pipelines instead of saving to disk.
