import cv2
import os
import numpy as np
import shutil
import time  # added by EH 12-23-25

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0      # Using external camera 2
INPUT_W, INPUT_H = 480, 480  # EAST input size (must be multiple of 32)
PAD_X, PAD_Y = 3, 3   # Padding around word boxes
SCALE = 1.05          # Scale factor for box enlargement
LETTER_CROP_FOLDER = "letter_crops"
os.makedirs(LETTER_CROP_FOLDER, exist_ok=True)
# ----------------------------------------

letter_count = 0

# (Optional) list available camera indices
for i in range(5):
    cap_test = cv2.VideoCapture(i)  # added by EH 12-23-25
    if cap_test.isOpened():
        print(f"Camera index {i} is available")
        cap_test.release()

# Load EAST model
read_nn_start_time = time.perf_counter()
net = cv2.dnn.readNet("frozen_east_text_detection.pb")
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
read_nn_stop_time = time.perf_counter()


camera_init_start_time = time.perf_counter()
# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)

# Set desired resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get current resolution
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Current camera resolution: {width} x {height}")
camera_init_stop_time = time.perf_counter()
last_boxes = []

# --- Cleanup function ---
def cleanup_letter_images():
    """Deletes all letter crop images and removes the folder."""
    if os.path.exists(LETTER_CROP_FOLDER):
        shutil.rmtree(LETTER_CROP_FOLDER)
        print(f"Cleaned up folder: {LETTER_CROP_FOLDER}")

# --- EAST decoding ---
def decode_predictions(scores, geometry, conf_threshold=0.5):
    detections, confidences = [], []
    (numRows, numCols) = scores.shape[2:4]
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        for x in range(numCols):
            if scoresData[x] < conf_threshold:
                continue
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            endX = int(x * 4.0 + (cos * x1[x]) + (sin * x2[x]))
            endY = int(y * 4.0 - (sin * x1[x]) + (cos * x2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            detections.append((startX, startY, endX, endY))
            confidences.append(float(scoresData[x]))
    return detections, confidences

# --- Sorting boxes top-to-bottom, left-to-right ---
def sort_boxes(boxes, row_tol=15):
    boxes = sorted(boxes, key=lambda b: b[1])
    rows, current_row, last_y = [], [], -row_tol * 2
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
            w_new = min(word_img.shape[1] - x_new, w + 2 * letter_pad)
            h_new = min(word_img.shape[0] - y_new, h + 2 * letter_pad)
            letter_crop = gray[y_new:y_new + h_new, x_new:x_new + w_new]
            letters.append((x_new, letter_crop))

    letters = sorted(letters, key=lambda l: l[0])
    return [l[1] for l in letters]

# ---------------- Headless debug config ----------------
DEBUG_DIR = "debug"
os.makedirs(DEBUG_DIR, exist_ok=True)  # added by EH 12-23-25
SAVE_HEATMAP_ONCE = True               # added by EH 12-23-25
SAVE_BOXES_ONCE = True                 # added by EH 12-23-25
heatmap_saved = False                  # added by EH 12-23-25
boxes_saved = False                    # added by EH 12-23-25
# ------------------------------------------------------

# --- Main loop ---
try:
    start_time = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (H, W) = frame.shape[:2]

        # Headless testing: always run detection (space pressed)
        key = ord(' ')  # Simulate pressing the spacebar for testing purposes

        if 1 == 1:
            # --- Enhance contrast for EAST input ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray_enhanced = clahe.apply(gray)

            # Convert back to BGR for EAST input
            frame_enhanced = cv2.cvtColor(gray_enhanced, cv2.COLOR_GRAY2BGR)

            resize_start_time = time.perf_counter()
            # Resize for EAST input
            frame_resized = cv2.resize(frame_enhanced, (INPUT_W, INPUT_H))
            blob = cv2.dnn.blobFromImage(
                frame_resized, 1.0, (INPUT_W, INPUT_H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False
            )
            net.setInput(blob)
            resize_stop_time = time.perf_counter()
            
            forward_start_time = time.perf_counter()
            scores, geometry = net.forward(layerNames)
            forward_stop_time = time.perf_counter()


            # --- Decode EAST predictions (safe rects) ---
            decode_start_time = time.perf_counter()
            detections, confidences = decode_predictions(scores, geometry)
            decode_stop_time = time.perf_counter()
            
            rects = []
            conf2 = []
            for (sx, sy, ex, ey), c in zip(detections, confidences):
                x = min(sx, ex)
                y = min(sy, ey)
                w = abs(ex - sx)
                h = abs(ey - sy)
                if w < 2 or h < 2:
                    continue
                rects.append([x, y, w, h])
                conf2.append(c)

            indices = cv2.dnn.NMSBoxes(rects, conf2, 0.5, 0.4)

            last_boxes = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = rects[i]

                    # Scale back up to original frame size
                    x = int(x * (W / INPUT_W))
                    y = int(y * (H / INPUT_H))
                    w = int(w * (W / INPUT_W))
                    h = int(h * (H / INPUT_H))

                    # Pad/scale
                    x_new = max(0, int(x - PAD_X))
                    y_new = max(0, int(y - PAD_Y))
                    w_new = min(W - x_new, int(w * SCALE + 2 * PAD_X))
                    h_new = min(H - y_new, int(h * SCALE + 2 * PAD_Y))

                    last_boxes.append((x_new, y_new, w_new, h_new))

            # --- Save EAST score heatmap (headless) ---
            scores_norm = cv2.normalize(scores[0, 0, :, :], None, 0, 255, cv2.NORM_MINMAX)
            scores_norm = scores_norm.astype(np.uint8)
            scores_color = cv2.applyColorMap(scores_norm, cv2.COLORMAP_JET)

            if (not SAVE_HEATMAP_ONCE) or (SAVE_HEATMAP_ONCE and not heatmap_saved):  # added by EH 12-23-25
                cv2.imwrite(os.path.join(DEBUG_DIR, "east_heatmap.png"), scores_color)  # added by EH 12-23-25
                heatmap_saved = True  # added by EH 12-23-25

            # --- Save detected boxes overlay (headless) ---
            if (not SAVE_BOXES_ONCE) or (SAVE_BOXES_ONCE and not boxes_saved):  # added by EH 12-23-25
                dbg = frame.copy()  # added by EH 12-23-25
                for (x, y, w, h) in last_boxes:  # added by EH 12-23-25
                    cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 2)  # added by EH 12-23-25
                cv2.imwrite(os.path.join(DEBUG_DIR, "detected_boxes.png"), dbg)  # added by EH 12-23-25
                boxes_saved = True  # added by EH 12-23-25

            # --- Crop and save words and letters ---
            sorted_words = sort_boxes(last_boxes, row_tol=15)
            word_count = 0
            letters_saved_this_frame = 0  # added by EH 12-23-25

            for (x, y, w, h) in sorted_words:
                word_crop = frame[y:y + h, x:x + w]  # Use original frame for clarity
                word_filename = f"{LETTER_CROP_FOLDER}/word_{word_count}.png"

                if word_crop.size == 0:
                    print("Skipping empty crop", x, y, w, h)
                    continue

                if not cv2.imwrite(word_filename, word_crop):  # added by EH 12-23-25
                    print("WARN: failed to write", word_filename)  # added by EH 12-23-25

                letters = crop_letters(word_crop)

                for letter_img in letters:
                    letter_resized = cv2.resize(letter_img, (32, 32), interpolation=cv2.INTER_CUBIC)
                    letter_filename = f"{LETTER_CROP_FOLDER}/word_{word_count}_letter_{letter_count}.png"

                    if cv2.imwrite(letter_filename, letter_resized):
                        letter_count += 1
                        letters_saved_this_frame += 1

                word_count += 1

            print(
                f"Detected {len(sorted_words)} words, "
                f"saved {letters_saved_this_frame} letters this frame "
                f"(total {letter_count})."
            )
            end_time = time.perf_counter()
            total_read_time = read_nn_stop_time - read_nn_start_time
            total_camera_init_time = camera_init_stop_time - camera_init_start_time
            total_resize_time = resize_stop_time - resize_start_time
            total_forward_time = forward_stop_time - forward_start_time
            total_decode_time = decode_stop_time - decode_start_time
            total_time = end_time - start_time

            print(f"Script execution total_read_time: {total_read_time:.4f} seconds")
            print(f"Script execution total_camera_init_time: {total_camera_init_time:.4f} seconds")
            print(f"Script execution total_resize_time: {total_resize_time:.4f} seconds")
            print(f"Script execution total_forward_time: {total_forward_time:.4f} seconds")
            print(f"Script execution total_decode_time: {total_decode_time:.4f} seconds")
            print(f"Script execution total_time: {total_time:.4f} seconds")

            break  # added by EH 12-23-25 this makes the script run only once and then exit

        elif key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Exiting...")
