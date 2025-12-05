import cv2
import numpy as np
import time
import os

# ---------------- CONFIG ----------------
CAMERA_INDEX = 2        # Camera index
SCAN_INTERVAL = 1.0     # seconds between scans
DISPLAY_WIDTH = 400     # width for display
UPSCALE_FACTOR = 2      # upscaling for small text
SAVE_FOLDER = "scanned_pages"
os.makedirs(SAVE_FOLDER, exist_ok=True)
# ----------------------------------------

last_scan_time = 0
display_clean = None  # placeholder for processed frame
scan_count = 0

# --- Perspective transform helpers ---
def order_points(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# --- Camera init ---
cap = cv2.VideoCapture(CAMERA_INDEX)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig = frame.copy()
        frame_display = cv2.resize(orig, (DISPLAY_WIDTH, int(orig.shape[0]*DISPLAY_WIDTH/orig.shape[1])))

        # --- Process at intervals ---
        if time.time() - last_scan_time >= SCAN_INTERVAL:
            last_scan_time = time.time()
            scan_count += 1

            # Resize for contour detection
            ratio = orig.shape[0] / 500.0
            small = cv2.resize(orig, (int(orig.shape[1]/ratio), 500))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (3,3), 0)
            edged = cv2.Canny(gray, 50, 150)

            # Find largest 4-point contour
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            pageCnt = None
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)
                if len(approx) == 4:
                    pageCnt = approx
                    break

            if pageCnt is not None:
                pts = pageCnt.reshape(4,2)  # original scale
                warped = four_point_transform(orig, pts)

                # Convert to gray
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                # Upscale for better small text
                warped_up = cv2.resize(warped_gray, (0,0), fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)

                # Adaptive threshold
                thresh = cv2.adaptiveThreshold(
                    warped_up, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )

                # Minimal noise removal
                kernel = np.ones((1,1), np.uint8)
                clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # --- Save high-res PNG ---
                filename = os.path.join(SAVE_FOLDER, f"scan_{scan_count:03d}.png")
                cv2.imwrite(filename, clean)
                print(f"[INFO] Saved: {filename}")

                # Resize for display only
                h_display = frame_display.shape[0]
                display_clean = cv2.resize(clean, (int(clean.shape[1]*h_display/clean.shape[0]), h_display))
            else:
                display_clean = frame_display

        # --- Show combined feed ---
        if display_clean is not None:
            # Match height of camera feed
            h, w = frame_display.shape[:2]
            display_resized = cv2.resize(display_clean, (int(display_clean.shape[1]*h/display_clean.shape[0]), h))
            combined = np.hstack((frame_display, cv2.cvtColor(display_resized, cv2.COLOR_GRAY2BGR)))
        else:
            combined = frame_display

        cv2.imshow("Live Scan Preview (Left: Camera, Right: Processed)", combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user!")

finally:
    cap.release()
    cv2.destroyAllWindows()
