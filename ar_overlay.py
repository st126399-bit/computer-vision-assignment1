import cv2
import numpy as np
import pywavefront

# -------------------------
# 1. Load calibration data
# -------------------------
data = np.load('calibration.npz')
mtx, dist = data['mtx'], data['dist']

# -------------------------
# 2. Load 3D model (.obj)
# -------------------------
# You can use any low-poly .obj model, e.g., "teapot.obj"
scene = pywavefront.Wavefront('model.obj', collect_faces=True)

# Extract vertices
vertices = np.array(scene.vertices)
print(f"Loaded 3D model with {len(vertices)} vertices")

# -------------------------
# 3. Start video capture
# -------------------------
cap = cv2.VideoCapture(0)
chessboard_size = (9, 6)
square_size = 0.025  # meters

# Prepare object points for the chessboard
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # 4. Detect chessboard corners
    # -------------------------
    ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret_cb:
        cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_cb)

        # -------------------------
        # 5. Estimate camera pose
        # -------------------------
        ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)
        if ret_pnp:
            # -------------------------
            # 6. Project 3D model points to 2D
            # -------------------------
            imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)

            # -------------------------
            # 7. Draw overlay
            # -------------------------
            for p in imgpts:
                cv2.circle(frame, tuple(p), 1, (0, 0, 255), -1)

            cv2.putText(frame, "3D model overlay active", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "Show chessboard to align model", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('AR Overlay', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
