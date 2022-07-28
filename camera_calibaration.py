"""
This code assumes that images used for calibration are of the same arUco marker board provided with code
"""

import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str,
                default="DICT_ARUCO_ORIGINAL",
                help="type of ArUCo tag to generate")
args = vars(ap.parse_args())

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# root directory of repo for relative path specification.
root = Path(__file__).parent.absolute()

# Set this flsg True for calibrating camera and False for validating results real time
calibrate_camera = False
create_board = True
# Set bord type between aruco and charuco

board_type = 'Charuco'

# Set path to the images
# calib_imgs_path = root.joinpath("aruco_data")

# For validating results, show aruco board to camera.
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])

# Provide length of the marker's side
markerLength = 5  # Here, measurement unit is centimetre.
# Provide separation between markers
markerSeparation = 0.5   # Here, measurement unit is centimetre.
global board
'''uncomment following block to draw and show the board'''
if create_board == True:
    if board_type == 'aruco':
        # create arUco board
        board = cv2.aruco.GridBoard_create(
            3, 4, markerLength, markerSeparation, aruco_dict)
        img = board.draw((2480, 3508))
        cv2.imshow("aruco", img)
        cv2.imwrite(args["output"], img)

    if board_type == 'Charuco':
        aruco_dict.bytesList = aruco_dict.bytesList[30:, :, :]
        # board = cv2. aruco.CharucoBoard_create(3, 4, markerLength, markerSeparation, aruco_dict)
        board = cv2.aruco.CharucoBoard_create(3, 4, 0.7, 0.5, aruco_dict)
        imboard = board.draw((2480, 3508))
        cv2.imwrite("tags/chessboard1.png", imboard)

elif calibrate_camera == True:
    arucoParams = cv2.aruco.DetectorParameters_create()
    img_list = []
    calib_fnms = cv2.calib_imgs_path.glob('*.jpg')
    print('Using ...', end='')
    for idx, fn in enumerate(calib_fnms):
        print(idx, '', end='')
        img = cv2.imread(str(root.joinpath(fn)))
        img_list.append(img)
        h, w, c = img.shape
    print('Calibration images')

    counter, corners_list, id_list = [], [], []
    first = True
    for im in tqdm(img_list):
        img_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            img_gray, aruco_dict, parameters=arucoParams)
        if first == True:
            corners_list = corners
            id_list = ids
            first = False
        else:
            corners_list = np.vstack((corners_list, corners))
            id_list = np.vstack((id_list, ids))
        counter.append(len(ids))
    print('Found {} unique markers'.format(np.unique(ids)))

    counter = np.array(counter)
    print("Calibrating camera .... Please wait...")
    # mat = np.zeros((3,3), float)
    ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraAruco(
        corners_list, id_list, counter, board, img_gray.shape, None, None)

    print("Camera matrix is \n", mtx,
          "\n And is stored in calibration.yaml file along with distortion coefficients : \n", dist)
    data = {'camera_matrix': np.asarray(
        mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
    with open("calibration.yaml", "w") as f:
        yaml.dump(data, f)

else:
    arucoParams = cv2.aruco.DetectorParameters_create()
    camera = cv2.VideoCapture(0)
    ret, img = camera.read()

    with open('calibration.yaml') as f:
        loadeddict = yaml.load(f)
    mtx = loadeddict.get('camera_matrix')
    dist = loadeddict.get('dist_coeff')
    mtx = np.array(mtx)
    dist = np.array(dist)

    ret, img = camera.read()
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h,  w = img_gray.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist, (w, h), 1, (w, h))

    pose_r, pose_t = [], []
    while True:
        ret, img = camera.read()
        img_aruco = img
        im_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h,  w = im_gray.shape[:2]
        dst = cv2.undistort(im_gray, mtx, dist, None, newcameramtx)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            dst, aruco_dict, parameters=arucoParams)
        # cv2.imshow("original", img_gray)
        if corners == None:
            print("pass")
        else:

            ret, rvec, tvec = cv2.aruco.estimatePoseBoard(
                corners, ids, board, newcameramtx, dist)  # For a board
            print("Rotation ", rvec, "Translation", tvec)
            if ret != 0:
                img_aruco = cv2.aruco.drawDetectedMarkers(
                    img, corners, ids, (0, 255, 0))
                # axis length 100 can be changed according to your requirement
                img_aruco = cv2.aruco.drawAxis(
                    img_aruco, newcameramtx, dist, rvec, tvec, 10)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        cv2.imshow("World co-ordinate frame axes", img_aruco)

cv2.destroyAllWindows()
