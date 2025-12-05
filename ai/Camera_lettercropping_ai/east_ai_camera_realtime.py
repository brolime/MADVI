import cv2
import os
import numpy as np
import shutil
import time

# ---------------- CONFIG ----------------
CAMERA_INDEX = 2      # Using external camera 2
INPUT_W, INPUT_H = 320, 320  # EAST input size (must be multiple of 32)
PAD_X, PAD_Y = 3, 3   # Padding around word boxes
SCALE = 1.05          # Scale factor for box enlargement
LETTER_CROP_FOLDER = "letter_crops"
os.makedirs(LETTER_CROP_FOLDER, exist_ok=True)
DETECTION_INTERVAL = 0.5  # Detect every 1.0 second
# ----------------------------------------

letter_count = 0
last_capture_time = 0  # Timer for auto detection

# Load EAST model
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# Initialize camera 2
cap = cv2.VideoCapture(CAMERA_INDEX)
last_boxes = []

# --- Cleanup function ---
def cleanup_letter_images():
    """Deletes all letter crop images and removes the folder."""
    if os.path.exists(LETTER_CROP_FOLDER):
        shutil.rmtree(LETTER_CROP_FOLDER)
        print(f"Cleaned up folder: {LETTER_CROP_FOLDER}")

# --- EAST decoding ---
def decode_predictions(scores, geometry, conf_threshold=0.7):
    detections, confidences = [], []
    (numRows, numCols) = scores.shape[2:4]
    for y in range(numRows):
        scoresData = scores[0,0,y]
        x0 = geometry[0,0,y]
        x1 = geometry[0,1,y]
        x2 = geometry[0,2,y]
        x3 = geometry[0,3,y]
        anglesData = geometry[0,4,y]
        for x in range(numCols):
            if scoresData[x] < conf_threshold:
                continue
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0[x]+x2[x]
            w = x1[x]+x3[x]
            endX = int(x*4.0 + (cos*x1[x]) + (sin*x2[x]))
            endY = int(y*4.0 - (sin*x1[x]) + (cos*x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            detections.append((startX,startY,endX,endY))
            confidences.append(float(scoresData[x]))
    return detections, confidences

# --- Sorting boxes top-to-bottom, left-to-right ---
def sort_boxes(boxes, row_tol=15):
    boxes = sorted(boxes, key=lambda b: b[1])
    rows, current_row, last_y = [], [], -row_tol*2
    for box in boxes:
        x, y, w, h = box
        if abs(y - last_y) > row_tol:
            if current_row:
                rows.append(current_row)
            current_row = [box]
            last_y = y
        else:
            current_row.append(box)
    if current_row:
        rows.append(current_row)

    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend(sorted(row, key=lambda b: b[0]))
    return sorted_boxes

# --- Crop letters from word ---
def crop_letters(word_img, letter_pad=3):
    gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letters = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 2 and h > 2:
            x_new = max(0, x - letter_pad)
            y_new = max(0, y - letter_pad)
            w_new = min(word_img.shape[1] - x_new, w + 2*letter_pad)
            h_new = min(word_img.shape[0] - y_new, h + 2*letter_pad)
            letter_crop = gray[y_new:y_new+h_new, x_new:x_new+w_new]
            letters.append((x_new, letter_crop))

    letters = sorted(letters, key=lambda l: l[0])
    return [l[1] for l in letters]

# --- Main loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        (H, W) = frame.shape[:2]

        # Draw previous boxes
        for (x, y, w, h) in last_boxes:
            cv2.rectangle(orig, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Text Detection (Auto-every-second, Q=quit)", orig)
        key = cv2.waitKey(1) & 0xFF

        # Auto detect every second
        if time.time() - last_capture_time >= DETECTION_INTERVAL:
            last_capture_time = time.time()

            # Reset letter count for this capture
            letter_count = 0

            frame_resized = cv2.resize(frame, (INPUT_W, INPUT_H))
            blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (INPUT_W, INPUT_H),
                                        (123.68,116.78,103.94), swapRB=True, crop=False)
            net.setInput(blob)
            scores, geometry = net.forward(layerNames)

            detections, confidences = decode_predictions(scores, geometry)
            rects = [[sx, sy, ex-sx, ey-sy] for (sx, sy, ex, ey) in detections]
            indices = cv2.dnn.NMSBoxes(rects, confidences, 0.5, 0.4)

            last_boxes = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = rects[i]
                    x = int(x * (W / INPUT_W))
                    y = int(y * (H / INPUT_H))
                    w = int(w * (W / INPUT_W))
                    h = int(h * (H / INPUT_H))
                    x_new = max(0, int(x - PAD_X))
                    y_new = max(0, int(y - PAD_Y))
                    w_new = min(W - x_new, int(w * SCALE + 2*PAD_X))
                    h_new = min(H - y_new, int(h * SCALE + 2*PAD_Y))
                    last_boxes.append((x_new, y_new, w_new, h_new))

            sorted_words = sort_boxes(last_boxes, row_tol=15)
            word_count = 0
            for (x, y, w, h) in sorted_words:
                word_crop = frame[y:y+h, x:x+w]
                letters = crop_letters(word_crop)
                for letter_img in letters:
                    letter_resized = cv2.resize(letter_img, (32, 32))
                    letter_filename = f"{LETTER_CROP_FOLDER}/word_{word_count}_letter_{letter_count}.png"
                    cv2.imwrite(letter_filename, letter_resized)
                    letter_count += 1
                word_count += 1

            print(f"Detected {len(sorted_words)} words, saved {letter_count} letters.")

        if key == ord('q'):
            break


except KeyboardInterrupt:
    print("Interrupted by user!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    cleanup_letter_images()
    print("Exiting...")
