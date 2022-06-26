from cmath import nan
import cv2 as cv
import glob
import numpy as np

class chessboard_calib:
    def __init__(self,row,column,chessboard_size):
        self.row=row
        self.column=column
        self.chessboard_size=chessboard_size
    def calibration(self):
        pattern_size = (self.row, self.column) 
        chessboard_scale_factor = self.chessboard_size # size of a square in mm
        image_size= None


        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2)
        objp=objp*chessboard_scale_factor
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        images = sorted(glob.glob('gui.chess/*.png'))
        count=0
        for img in images:
            img = cv.imread(img)
            size_of_image = (img.shape[1], img.shape[0])
            image_size=size_of_image
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points (after refining them)
            if ret == True:

                objpoints.append(objp)
                print("success: ")


                corners = cv.cornerSubPix(gray, corners, (10,10), (-1,-1), criteria)
                imgpoints.append(corners)

                

                # Draw and display the corners
                cv.drawChessboardCorners(img, pattern_size, corners, ret)
                cv.imshow('img', img)
                
                cv.waitKey()
                heightL, widthL, channelsL = img.shape
            else:
                print("Can not find!!! ")
        cv.destroyAllWindows()
                

        #objp=[objp]*len(imgpoints)
        retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
        newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
        print(newCameraMatrixL)
        return retL, cameraMatrixL, distL, rvecsL, tvecsL