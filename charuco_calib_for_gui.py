import numpy as np
import cv2.aruco as aruco
import cv2 as cv
import glob
class charuco_calib:
    def __init__(self,row,column,squareLength,markerLength,dic,dic_var):
        self.row=row
        self.column=column
        self.squareLength=squareLength
        self.markerLength=markerLength
        self.dic=dic
        self.dic_var=dic_var
    def calibration(self):
        key=getattr(aruco,f'DICT_{self.dic}X{self.dic}_{self.dic_var}')
        arucoDIC=aruco.Dictionary_get(key)
        squaresX=self.row
        squaresY=self.column
        squareLength=self.squareLength
        markerLength=self.markerLength
        board=aruco.CharucoBoard_create(squaresX, squaresY,squareLength,markerLength,arucoDIC)#Arucoboard parameter

        #-----------------------------------
        # whit this function 
        # Input=Images, that saved in part1
        # output= Corners and Ids and size of Image
        def read_charuco(frames):
            print("start to read Aruco")
            all_corners=[]
            all_ids=[]
            for frame in frames:
                print("=> Processing image {0}".format(frame))
                gray = cv.imread(frame)
                gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
                corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, arucoDIC)
                if len(corners)>0:
                    #Interpolate position of ChArUco board corners
                    ret, c_corners, c_ids = aruco.interpolateCornersCharuco(corners, ids, gray, board)
                    
                    if ret>0:
                        #ret is the number if detected corners
                        print("succesful")
                        all_corners.append(c_corners)
                        all_ids.append(c_ids)
                else:
                    print("Failed")
            imsize = gray.shape
            return all_corners, all_ids, imsize

        def calibrate_camera(allCorners,allIds,imsize):
            """
            #Calibrates the camera using the dected corners.
            """
            print("CAMERA CALIBRATION")
            allCorners = [x for x in allCorners if len(x) >= 4]
            allIds = [x for x in allIds if len(x) >= 4]
            ret, camera_matrix, dist_coeff, rvec, tvec = aruco.calibrateCameraCharuco(
                allCorners, allIds, board, imsize, None, None
            )
            return ret, camera_matrix, dist_coeff, rvec, tvec

        images= sorted(glob.glob('gui.charuco/*.png'))

        print("start to read images")
        allCorners,allIds,imsize=read_charuco(images)
        print("LENGTH OF CORNERS:",len(allCorners))
        ret, camera_matrix, dist_coeff, rvec, tvec=calibrate_camera(allCorners,allIds,imsize)
        return ret, camera_matrix, dist_coeff, rvec, tvec

