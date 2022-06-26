
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2  as cv
from ids_peak_ipl import ids_peak_ipl
import numpy as np
from PyQt5.QtGui import QImage



from live_video import camera
from chessboard_calib_for_gui import chessboard_calib
from charuco_calib_for_gui import charuco_calib

charuco=True
exposure=218000
frame=30

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1048, 748)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 110, 206, 364))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_items_calib = QtWidgets.QLabel(self.tab)
        self.label_items_calib.setObjectName("label_items_calib")
        self.verticalLayout.addWidget(self.label_items_calib)
        self.comboBox = QtWidgets.QComboBox(self.tab)
        self.comboBox.setObjectName("comboBox")
        self.verticalLayout.addWidget(self.comboBox)
        self.setting_first = QtWidgets.QCheckBox(self.tab)
        self.setting_first.setObjectName("setting_first")
        self.verticalLayout.addWidget(self.setting_first)
        self.label_ex = QtWidgets.QLabel(self.tab)
        self.label_ex.setObjectName("label_ex")
        self.verticalLayout.addWidget(self.label_ex)
        self.scroll_ex = QtWidgets.QScrollBar(self.tab)
        self.scroll_ex.setOrientation(QtCore.Qt.Horizontal)
        self.scroll_ex.setObjectName("scroll_ex")
        self.verticalLayout.addWidget(self.scroll_ex)
        self.value_ex = QtWidgets.QLabel(self.tab)
        self.value_ex.setObjectName("value_ex")
        self.verticalLayout.addWidget(self.value_ex)
        self.label_frame = QtWidgets.QLabel(self.tab)
        self.label_frame.setObjectName("label_frame")
        self.verticalLayout.addWidget(self.label_frame)
        self.scroll_frame = QtWidgets.QScrollBar(self.tab)
        self.scroll_frame.setOrientation(QtCore.Qt.Horizontal)
        self.scroll_frame.setObjectName("scroll_frame")
        self.verticalLayout.addWidget(self.scroll_frame)
        self.value_frame = QtWidgets.QLabel(self.tab)
        self.value_frame.setObjectName("value_frame")
        self.verticalLayout.addWidget(self.value_frame)
        self.start_live = QtWidgets.QPushButton(self.tab)
        self.start_live.setObjectName("start_live")
        self.verticalLayout.addWidget(self.start_live)
        self.save_image = QtWidgets.QPushButton(self.tab)
        self.save_image.setObjectName("save_image")
        self.verticalLayout.addWidget(self.save_image)
        self.calib_process = QtWidgets.QPushButton(self.tab)
        self.calib_process.setObjectName("calib_process")
        self.verticalLayout.addWidget(self.calib_process)
        self.exit_button = QtWidgets.QPushButton(self.tab)
        self.exit_button.setObjectName("exit_button")
        self.verticalLayout.addWidget(self.exit_button)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_5.setGeometry(QtCore.QRect(40, 60, 121, 41))
        self.pushButton_5.setObjectName("pushButton_5")
        self.tabWidget.addTab(self.tab_2, "")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 10, 121, 91))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap("../hskempten.png"))
        self.label_4.setScaledContents(True)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(590, 10, 121, 91))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap("../ids.jpg"))
        self.label_5.setScaledContents(True)
        self.label_5.setObjectName("label_5")
        self.show_mtx = QtWidgets.QLabel(self.centralwidget)
        self.show_mtx.setGeometry(QtCore.QRect(501, 140, 371, 171))
        self.show_mtx.setObjectName("show_mtx")
        self.img = QtWidgets.QLabel(self.centralwidget)
        self.img.setGeometry(QtCore.QRect(490, 350, 531, 351))
        self.img.setObjectName("img")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(220, 100, 211, 413))
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_items_calib_2 = QtWidgets.QLabel(self.widget)
        self.label_items_calib_2.setObjectName("label_items_calib_2")
        self.verticalLayout_2.addWidget(self.label_items_calib_2)
        self.comboBox_2 = QtWidgets.QComboBox(self.widget)
        self.comboBox_2.setObjectName("comboBox_2")
        self.verticalLayout_2.addWidget(self.comboBox_2)
        self.rows = QtWidgets.QLabel(self.widget)
        self.rows.setObjectName("rows")
        self.verticalLayout_2.addWidget(self.rows)
        self.lineEdit_raw = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_raw.setObjectName("lineEdit_raw")
        self.verticalLayout_2.addWidget(self.lineEdit_raw)
        self.columns = QtWidgets.QLabel(self.widget)
        self.columns.setObjectName("columns")
        self.verticalLayout_2.addWidget(self.columns)
        self.lineEdit_column = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_column.setObjectName("lineEdit_column")
        self.verticalLayout_2.addWidget(self.lineEdit_column)
        self.chess_size = QtWidgets.QLabel(self.widget)
        self.chess_size.setObjectName("chess_size")
        self.verticalLayout_2.addWidget(self.chess_size)
        self.lineEdit_chess = QtWidgets.QLineEdit(self.widget)
        self.lineEdit_chess.setObjectName("lineEdit_chess")
        self.verticalLayout_2.addWidget(self.lineEdit_chess)
        self.squarel = QtWidgets.QLabel(self.widget)
        self.squarel.setObjectName("squarel")
        self.verticalLayout_2.addWidget(self.squarel)
        self.lineEdite_squarel = QtWidgets.QLineEdit(self.widget)
        self.lineEdite_squarel.setObjectName("lineEdite_squarel")
        self.verticalLayout_2.addWidget(self.lineEdite_squarel)
        self.marker = QtWidgets.QLabel(self.widget)
        self.marker.setObjectName("marker")
        self.verticalLayout_2.addWidget(self.marker)
        self.lineEdite_marker = QtWidgets.QLineEdit(self.widget)
        self.lineEdite_marker.setObjectName("lineEdite_marker")
        self.verticalLayout_2.addWidget(self.lineEdite_marker)
        self.dic = QtWidgets.QLabel(self.widget)
        self.dic.setObjectName("dic")
        self.verticalLayout_2.addWidget(self.dic)
        self.lineEdite_dic = QtWidgets.QLineEdit(self.widget)
        self.lineEdite_dic.setObjectName("lineEdite_dic")
        self.verticalLayout_2.addWidget(self.lineEdite_dic)
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.lineEdite_num = QtWidgets.QLineEdit(self.widget)
        self.lineEdite_num.setObjectName("lineEdite_num")
        self.verticalLayout_2.addWidget(self.lineEdite_num)
        self.splitter = QtWidgets.QSplitter(self.widget)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.setObjectName("splitter")
        self.start_calibration = QtWidgets.QPushButton(self.splitter)
        self.start_calibration.setObjectName("start_calibration")
        self.save_matrix = QtWidgets.QPushButton(self.splitter)
        self.save_matrix.setObjectName("save_matrix")
        self.verticalLayout_2.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1048, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionmin = QtWidgets.QAction(MainWindow)
        self.actionmin.setObjectName("actionmin")
        self.actionnormal = QtWidgets.QAction(MainWindow)
        self.actionnormal.setObjectName("actionnormal")
        self.actionmax = QtWidgets.QAction(MainWindow)
        self.actionmax.setObjectName("actionmax")
        self.menuFile.addAction(self.actionExit)
        self.menuView.addAction(self.actionnormal)
        self.menuView.addAction(self.actionmax)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        #######
        self.num_charuco=0
        self.num_chess=0
        ####################
        # unenables all bottums 
        self.label_ex.setEnabled(False)
        self.value_ex.setEnabled(False)
        self.scroll_ex.setEnabled(False)

        self.label_frame.setEnabled(False)
        self.value_frame.setEnabled(False)
        self.scroll_frame.setEnabled(False)

        #self.start_live.setEnabled(False)
        self.save_image.setEnabled(False)
        #self.calib_process.setEnabled(False)
        self.exit_button.setEnabled(False)

        self.label_items_calib_2.setEnabled(False)
        self.comboBox_2.setEnabled(False)
        self.rows.setEnabled(False)
        self.lineEdit_raw.setEnabled(False)
        self.columns.setEnabled(False)
        self.lineEdit_column.setEnabled(False)
        self.chess_size.setEnabled(False)
        self.lineEdit_chess.setEnabled(False)

        self.squarel.setEnabled(False)
        self.lineEdite_squarel.setEnabled(False)
        self.marker.setEnabled(False)
        self.lineEdite_marker.setEnabled(False)


        self.dic.setEnabled(False)
        self.lineEdite_dic.setEnabled(False)
        self.lineEdite_num.setEnabled(False)

        self.start_calibration.setEnabled(False)
        self.save_matrix.setEnabled(False)
        

        #########################
        # set items for combiBox
        
        self.comboBox.addItem("Charucoboard")
        self.comboBox.addItem("Chessboard")

        ##################################
        # active Combibox when it clicked
        #this box connected to combi_active() function
        self.comboBox.activated.connect(self.combi_active)

        ###################################
        # actice setting box
        #this box connected to setting_active() function
        # active scrolls buttons to get value  
        self.setting_first.toggled.connect(self.setting_active)
        ###################################
        #connection start_live button to start() functtion
        self.start_live.clicked.connect(self.start)
        #####################################
        #connection  exit_button to exit() functtion
        self.exit_button.clicked.connect(self.exit)
        ######################################
        # connection  calib_process_button to calibration_items() functtion
        # in calibration_items() functtion must enable all items 
        self.calib_process.clicked.connect(self.calibration_items)
        #######################################
        self.comboBox_2.addItem("Charucoboard")
        self.comboBox_2.addItem("Chessboard")
        # active Combibox when it clicked
        #this box connected to combi_2_active() function
        self.comboBox_2.activated.connect(self.combi_2_active)
        ########################################
        #connection  start_calibration_button to calibration_process() functtion
        self.start_calibration.clicked.connect(self.calibration_process)
        #########################################
        # to save camera calibration matrix
        self.save_matrix.clicked.connect(self.save_mtx)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.actionExit.triggered.connect(MainWindow.close)
        self.actionnormal.triggered.connect(MainWindow.showNormal)
        self.actionmax.triggered.connect(MainWindow.showMaximized)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_items_calib.setText(_translate("MainWindow", "Please  select items to calibration"))
        self.setting_first.setText(_translate("MainWindow", "setting"))
        self.label_ex.setText(_translate("MainWindow", "Exposure time [ms]"))
        self.value_ex.setText(_translate("MainWindow", "value"))
        self.label_frame.setText(_translate("MainWindow", "Frame rate [fps]"))
        self.value_frame.setText(_translate("MainWindow", "value"))
        self.start_live.setText(_translate("MainWindow", "Start "))
        self.save_image.setText(_translate("MainWindow", "save Images"))
        self.calib_process.setText(_translate("MainWindow", "camera calibration matrix"))
        self.exit_button.setText(_translate("MainWindow", "exit"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "camera calibration"))
        self.pushButton_5.setText(_translate("MainWindow", "start"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "pose estimation"))
        self.show_mtx.setText(_translate("MainWindow", "TextLabel"))
        self.img.setText(_translate("MainWindow", "TextLabel"))
        self.label_items_calib_2.setText(_translate("MainWindow", "Please  select items to calibration"))
        self.rows.setText(_translate("MainWindow", "rows"))
        self.columns.setText(_translate("MainWindow", "columns"))
        self.chess_size.setText(_translate("MainWindow", "chessboard_size [mm]"))
        self.squarel.setText(_translate("MainWindow", "squareLength[m]"))
        self.marker.setText(_translate("MainWindow", "markerLength[m]"))
        self.dic.setText(_translate("MainWindow", "aruco.Dictionary"))
        self.label.setText(_translate("MainWindow", "X"))
        self.start_calibration.setText(_translate("MainWindow", "start calibration"))
        self.save_matrix.setText(_translate("MainWindow", "save  matrix"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionmin.setText(_translate("MainWindow", "min"))
        self.actionnormal.setText(_translate("MainWindow", "normal"))
        self.actionmax.setText(_translate("MainWindow", "max"))
    
     ##################################
        # active Combibox when it clicked
        #this function connected to combibox
        # charuco is by default true. if chessboard clicked it will be false. 
    def combi_active(self):
            self.label_items_calib.setText(f'{self.comboBox.currentText()}')
            global charuco
            s=self.comboBox.currentText()
            sf="Chessboard"
            if s==sf:
                charuco=False
            else:
                charuco=True
    ##################################################
    ## actice setting box
    # active scrolls buttons
    # set value for frame and exposure time with exposure_change and frame_change function 
    def setting_active(self):
        if self.setting_first.isChecked()==True:
                # active scrolls buttons
                self.label_ex.setEnabled(True)
                self.value_ex.setEnabled(True)
                self.scroll_ex.setEnabled(True)
                #set default value on lable
                self.value_ex.setText(str(exposure))
                self.scroll_ex.setValue(exposure)
                self.scroll_ex.setRange(0,1000000)
                #update exposure value - exposure_change function
                self.scroll_ex.valueChanged.connect(self.exposure_change)

        
                # active scrolls buttons
                self.label_frame.setEnabled(True)
                self.value_frame.setEnabled(True)
                self.scroll_frame.setEnabled(True)
                #set default value on lable
                self.value_frame.setText(str(frame))
                self.scroll_frame.setValue(frame)
                self.scroll_frame.setRange(0,100)
                #update frame value - frame_change function
                self.scroll_frame.valueChanged.connect(self.frame_change)
                


                
        if self.setting_first.isChecked()==False:
                self.label_ex.setEnabled(False)
                self.value_ex.setEnabled(False)
                self.scroll_ex.setEnabled(False)

                self.label_frame.setEnabled(False)
                self.value_frame.setEnabled(False)
                self.scroll_frame.setEnabled(False)
    ###########################################
    ###########################################
    def exposure_change(self):
        ## update exposure time 
        # default is exposure=218000 
        global exposure
        exposure = self.scroll_ex.value()
        # setting text to the label
        self.value_ex.setText(str(exposure))

    ##########################################
    def frame_change(self):
        # update frame rate
        # default is  frame=30
        global frame
        frame = self.scroll_frame.value()
        # setting text to the label
        self.value_frame.setText(str(frame))
    ###########################################
    ###########################################
    def setPhoto(self,image):

        # set opencv frame as QImage
        #with this function, video can play on img(label)
        self.image=image
        self.image=cv.resize(image,(460,300),interpolation = cv.INTER_AREA)
        self.image=cv.cvtColor(self.image,cv.COLOR_GRAY2RGB)
        image = QImage(self.image, self.image.shape[1],self.image.shape[0],self.image.strides[0],QImage.Format_RGB888)
        self.img.setPixmap(QtGui.QPixmap.fromImage(image))

    def start(self):

        self.started=True
        # enabl save and exit
        self.save_image.setEnabled(True)
        self.exit_button.setEnabled(True)
        # connect to save function
        ########################
        #####       #####       #
        #####       #####       #
        #          WARNING
        # CREATE TWO FOLDERS
        #gui.charuco for save images with charucoboard
        #gui.chess for save images with chessboard
        #####       #####       #
        #####       #####       #
        self.save_image.clicked.connect(self.save)

        c=camera(exposure,frame)# give as start parmeter
        buffer,m_dataStream,m_device,m_node_map_remote_device,target_fps,excosure=c.get_frame()
        while True:

            buffer = m_dataStream.WaitForFinishedBuffer(5000)
            image = ids_peak_ipl.Image_CreateFromSizeAndBuffer(
            buffer.PixelFormat(),
            buffer.BasePtr(),
            buffer.Size(),
            buffer.Width(),
            buffer.Height())
            converted_ipl_image = image.ConvertTo(ids_peak_ipl.PixelFormatName_Mono8)
            opencv_image = converted_ipl_image.get_numpy_3D()
            m_dataStream.QueueBuffer(buffer)
            #h=int(opencv_image.shape[0]/2)
            #w=int(opencv_image.shape[1]/2)
            #dim=(w,h)
            #opencv_image=cv.resize(opencv_image,dim,interpolation = cv.INTER_AREA)
            self.opencv_image=opencv_image
            ###### set as QImage
            self.setPhoto(opencv_image)
            k=cv.waitKey(5)
            #cv.imshow("Live Video",opencv_image)
    
            if self.started==False:
                break
        #cv.destroyAllWindows()
    ##########################################
    def exit(self):
        self.started=False
        self.save_image.setEnabled(False)
        self.exit_button.setEnabled(False)
    #########################################
    def save(self):
        if charuco==True:
            self.num_charuco+=1
            cv.imwrite('gui.charuco/image' + str(self.num_charuco) + '.png',self.opencv_image)
        else:
            self.num_chess+=1
            cv.imwrite('gui.chess/image' + str(self.num_chess) + '.png',self.opencv_image)
        
    ############################################
    ############################################
    def calibration_items(self):
        self.label_items_calib_2.setEnabled(True)
        self.comboBox_2.setEnabled(True)

        
    def combi_2_active(self):
        self.label_items_calib_2.setText(f'{self.comboBox_2.currentText()}')
        self.calib_with_charu=True
        s=self.comboBox_2.currentText()
        sf="Chessboard"
        if s==sf:
                self.calib_with_charu=False
        else:
                self.calib_with_charu=True
        
        if self.calib_with_charu==False:
            self.rows.setEnabled(True)
            self.lineEdit_raw.setEnabled(True)
            self.lineEdit_raw.setValidator(QtGui.QIntValidator())
            self.lineEdit_raw.setMaxLength(2)
            self.columns.setEnabled(True)
            self.lineEdit_column.setEnabled(True)
            self.lineEdit_column.setValidator(QtGui.QIntValidator())
            self.lineEdit_column.setMaxLength(2)
            self.chess_size.setEnabled(True)
            self.lineEdit_chess.setEnabled(True)
            self.lineEdit_chess.setValidator(QtGui.QIntValidator())
            self.lineEdit_chess.setMaxLength(4)
            self.start_calibration.setEnabled(True)
            self.squarel.setEnabled(False)
            self.lineEdite_squarel.setEnabled(False)
            self.marker.setEnabled(False)
            self.lineEdite_marker.setEnabled(False)
            self.dic.setEnabled(False)
            self.lineEdite_dic.setEnabled(False)
            self.lineEdite_num.setEnabled(False)
        if self.calib_with_charu==True:
            self.chess_size.setEnabled(False)
            self.lineEdit_chess.setEnabled(False)
            self.rows.setEnabled(True)
            self.lineEdit_raw.setEnabled(True)
            self.lineEdit_raw.setValidator(QtGui.QIntValidator())
            self.lineEdit_raw.setMaxLength(2)
            self.columns.setEnabled(True)
            self.lineEdit_column.setEnabled(True)
            self.lineEdit_column.setValidator(QtGui.QIntValidator())
            self.lineEdit_column.setMaxLength(2)
            self.squarel.setEnabled(True)
            self.lineEdite_squarel.setEnabled(True)
            self.lineEdite_squarel.setValidator(QtGui.QDoubleValidator())
            self.lineEdite_squarel.setMaxLength(6)
            
            self.marker.setEnabled(True)
            self.lineEdite_marker.setEnabled(True)
            self.lineEdite_marker.setValidator(QtGui.QDoubleValidator())
            self.lineEdite_marker.setMaxLength(6)


            self.dic.setEnabled(True)
            self.lineEdite_dic.setEnabled(True)
            self.lineEdite_dic.setValidator(QtGui.QIntValidator())
            self.lineEdite_dic.setMaxLength(1)
            self.lineEdite_num.setEnabled(True)
            self.lineEdite_num.setValidator(QtGui.QIntValidator())
            self.lineEdite_num.setMaxLength(3)
            self.start_calibration.setEnabled(True)
    def calibration_process(self):
           
        self.num_rows=int(self.lineEdit_raw.text())
        
        self.num_columns=int(self.lineEdit_column.text())
        
        if self.calib_with_charu==False:
            self.num_chess_size=int(self.lineEdit_chess.text())
            w=chessboard_calib(self.num_rows,self.num_columns,self.num_chess_size)
            _, self.mtx_chess, self.dis_chess, self.rvec_chess, self.tvec_chess=w.calibration()
            self.show_mtx.setText(str(self.mtx_chess)) 
        if self.calib_with_charu==True:
            self.num_squarel_size=float(self.lineEdite_squarel.text())
            self.num_marker_size=float(self.lineEdite_marker.text())
            self.num_dic_size=int(self.lineEdite_dic.text())
            self.num_dic_var=int(self.lineEdite_num.text())
            q=charuco_calib(self.num_rows,self.num_columns,self.num_squarel_size,self.num_marker_size
                ,self.num_dic_size,self.num_dic_var)
            _, self.mtx_charu, self.dis_charu, self.rvec_charu, self.tvec_charu=q.calibration()
            self.show_mtx.setText(str(self.mtx_charu)) 
            
        self.save_matrix.setEnabled(True)
    def save_mtx(self):
        if self.calib_with_charu==False:
            np.savez("chess_calib_params.npz", K=self.mtx_chess, D=self.dis_chess,R=self.rvec_chess, T=self.tvec_chess) 
        if self.calib_with_charu==True:
            np.savez("charuco_calib_params.npz", K=self.mtx_charu, D=self.dis_charu,R=self.rvec_charu, T=self.tvec_charu)

           


        


            



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
