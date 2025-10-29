import cv2
import numpy as np
import pywavefront

# -------------------------
# 1. Load calibration data
# -------------------------
data = np.load('calibration.npz')
mtx, dist = data['mtx'], data['dist']

# -------------------------
# 2. Load 3D model
# -------------------------
scene = pywavefront.Wavefront('model.obj', collect_faces=True)
vertices = np.array(scene.vertices)
print(f"Loaded 3D model with {len(vertices)} vertices")

# -------------------------
# 3. Setup ArUco detector
# -------------------------
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
marker_length = 0.05  # meters (5 cm)

# -------------------------
# 4. Start camera
# -------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose of marker ID 0
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 0:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], marker_length, mtx, dist
                )

                # Draw 3D axis
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.03)

                # Project model points onto the image
                imgpts, _ = cv2.projectPoints(vertices, rvec, tvec, mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1, 2)

                for p in imgpts:
                    cv2.circle(frame, tuple(p), 1, (0, 0, 255), -1)

                cv2.putText(frame, "3D model overlay active", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Show A4 ArUco marker (ID=0)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('AR Overlay', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# Choose the correct dictionary (the one that works for you)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate marker with ID = 0 (you can change ID if needed)
marker_id = 0
marker_size = 700  # in pixels (for A4 printing quality)

marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size)
cv2.imwrite('A4_ArUco_6x6_ID0.png', marker_image)

print("✅ Marker saved as A4_ArUco_6x6_ID0.png")

