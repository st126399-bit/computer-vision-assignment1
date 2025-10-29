# camera_calibration.py
import cv2
import numpy as np
import glob

def calibrate_camera(chessboard_size=(9,6), square_size=0.025, save_file='calibration.npz'):
    """
    Calibrate webcam using chessboard images and save calibration.npz
    - chessboard_size: number of internal corners (nx, ny)
    - square_size: real-world size of a square (e.g. 0.025 = 2.5 cm)
    """
    # prepare object points like (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp = objp * square_size  # scale by real size

    # arrays to store points
    objpoints = []  # 3D points in real world
    imgpoints = []  # 2D points in image plane

    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture frame when chessboard is visible. ESC to finish.")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret_cb:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret_cb)
            cv2.putText(frame, "Chessboard detected - press SPACE to save", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "Show full chessboard to camera", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow('Calibration', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32 and ret_cb:  # SPACE to save
            objpoints.append(objp)
            imgpoints.append(corners)
            count += 1
            print(f"Captured {count} frames")

    cap.release()
    cv2.destroyAllWindows()

    # Calibrate the camera using all collected points
    if len(objpoints) < 10:
        print("Not enough frames for calibration (need ~10+). Try again.")
        return

    print("Calibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("Calibration successful! RMS error:", ret)
    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist.ravel())

    # Save to file
    np.savez(save_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"Saved calibration to {save_file}")

if __name__ == "__main__":
    calibrate_camera(chessboard_size=(9,6), square_size=0.025)
