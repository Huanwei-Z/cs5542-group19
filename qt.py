# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import (QLabel, QFrame, QLineEdit,   
                                    QPushButton, QHBoxLayout,  QVBoxLayout, QGroupBox, QTextEdit, QScrollArea, QCheckBox, 
                                     QGridLayout)
from PyQt5.QtCore import  QTimer, Qt
from PyQt5.QtGui import  QPixmap, QImage, QKeyEvent, QCursor
import cv2
import time
import os

#from keras.models import load_model  
#import train
class FrameBase(QFrame):
    def __init__(self, parent=None):
        super(FrameBase, self).__init__()
        self.parent = parent
        self.enterFunc = None
        self.m_flag = False

    def setEnterFunc(self, func):
        self.enterFunc = func

    def keyPressEvent(self, event):
        keyEvent = QKeyEvent(event)
        key = keyEvent.key()
        if key in (Qt.Key_Enter, Qt.Key_Return):
            if self.enterFunc:
                self.enterFunc()

    # event。
    """override mouse events to realize window dragging.。"""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # Get the position of the mouse relative to the window
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # Change the mouse icon

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # Change window position
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))


class Top(QFrame):
    def __init__(self, parent=None):
        super(Top, self).__init__()
        self.resize(400, 800)
        self.parent = parent
        self.setWindowTitle('SVM result')
        self.setWidgets()
        self.setLayouts()

    def setWidgets(self):
        """Create all the controls."""
        file = open('result/svmresult.txt','r')
        line=file.readlines()
        self.label = QTextEdit()
        self.label.setText(''.join(line))

    def setLayouts(self):
        """set the layout."""
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.setLayout(self.mainLayout)

class Top2(QFrame):
    def __init__(self, parent=None):
        super(Top2, self).__init__()
        self.resize(600, 500)
        self.parent = parent
        self.setWindowTitle('Training model')
        self.setWidgets()
        self.setLayouts()

    def setWidgets(self):
        """Create all the controls."""
        self.label = QLabel()
        self.label.setPixmap(QPixmap('reco.png'))

    def setLayouts(self):
        """set the layout."""
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.setLayout(self.mainLayout)

class Top3(QFrame):
    def __init__(self, parent=None):
        super(Top3, self).__init__()
        self.resize(400, 800)
        self.parent = parent
        self.setWindowTitle('Test result display')
        self.setWidgets()
        self.setLayouts()

    def setWidgets(self):
        """Create all the controls."""
        file = open('result/cnnresult.txt','r')
        line=file.readlines()
        self.label = QTextEdit()
        self.label.setText(''.join(line))

    def setLayouts(self):
        """set the layout."""
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.setLayout(self.mainLayout)


class Top4(QFrame):
    def __init__(self, parent=None):
        super(Top4, self).__init__()
        self.resize(1000, 600)
        self.parent = parent
        self.setWindowTitle('Attendance results')
        self.setWidgets()
        self.setLayouts()

    def setWidgets(self):
        """Create all the controls."""
        file = open('result/result.txt','r', encoding='utf-8')
        self.lst_label=[]
        self.lst_check=[]
        while True:
            line=file.readline()
            if not line:
                break
            lst=line.strip().split(':')
            label=QLabel()
            label.setMaximumSize(300, 200)
            label.setMinimumSize(300, 200)
            
            checkbox=QCheckBox()
            checkbox.setEnabled(False)
            label.setScaledContents(True)
            label.setPixmap(QPixmap('result/%s.jpg'%lst[0]))
            
            checkbox.setChecked(True if lst[1]=='1' else False)
            self.lst_label.append(label)
            self.lst_check.append(checkbox)

    def setLayouts(self):
        """set the layouts."""
        self.layout = QHBoxLayout()
        mainLayout = QVBoxLayout()
        lyt = QHBoxLayout()
        for i in range(len(self.lst_label)):
            if i%2==0 and i:
                mainLayout.addLayout(lyt)
                lyt = QHBoxLayout()
            lyt.addWidget(self.lst_label[i])
            lyt.addWidget(self.lst_check[i])
            
        mainLayout.addLayout(lyt)
        self.scroll = QScrollArea()
        self.scroll.setContentsMargins(0, 0, 0, 0)
        self.scroll.setLayout(mainLayout)
        self.layout.addWidget(self.scroll)
        self.setLayout(self.layout)


class WarnHeader(QFrame):
    def __init__(self, parent=None):
        super(WarnHeader, self).__init__()
        self.setObjectName('WarnHeader')
        self.parent = parent
        self.setWidgets()
        self.setLayouts()

    def setWidgets(self):
        """Create all the controls."""
        self.titleLabel = QLabel('Cow face recognition system', self)
        self.titleLabel.setObjectName("WarnLabel")
        self.closeButton = QPushButton('×', self)
        self.closeButton.setObjectName("closeButton")
        self.closeButton.setMinimumSize(70, 48)
        self.closeButton.clicked.connect(self.parent.close)

    def setLayouts(self):
        """set the layouts。"""
        self.mainLayout = QHBoxLayout()
        self.mainLayout.setSpacing(0)
        self.mainLayout.addWidget(self.titleLabel)
        self.mainLayout.addStretch(1)
        self.mainLayout.addWidget(self.closeButton)
        self.mainLayout.setContentsMargins(20, 0, 0, 5)
        self.setLayout(self.mainLayout)

class WarnBottom(QGroupBox):
    def __init__(self, parent=None):
        super(WarnBottom, self).__init__()
        self.parent = parent
        self.setObjectName("widget")
        self.setWindowTitle('Cow face recognition system')
        self.resize(1200, 900)
        
        self.btn=QPushButton("Turn on the camera")
        self.btn.setObjectName("save")
        self.btn.clicked.connect(self.openimage)
        self.btn.setMaximumSize(150, 36)
        self.btn.setMinimumSize(150, 36)
        
        self.btn2=QPushButton("Turn off the camera")
        self.btn2.setObjectName("save")
        self.btn2.clicked.connect(self.closeimage)
        self.btn2.setMaximumSize(150, 36)
        self.btn2.setMinimumSize(150, 36)
        
        self.btn3=QPushButton("Shoot")
        self.btn3.setObjectName("save_next")
        self.btn3.clicked.connect(self.train_data)
        self.btn3.setMaximumSize(100, 36)
        self.btn3.setMinimumSize(100, 36)
        
        self.btn4=QPushButton("Test result")
        self.btn4.setObjectName("save_next")
        self.btn4.clicked.connect(self.showtop3)
        self.btn4.setMaximumSize(100, 36)
        self.btn4.setMinimumSize(100, 36)
        
        self.btn5=QPushButton("Training curve ")
        self.btn5.setObjectName("save_next")
        self.btn5.clicked.connect(self.showtop2)
        self.btn5.setMaximumSize(100, 36)
        self.btn5.setMinimumSize(100, 36)
        
        self.btn6=QPushButton("SVM comparison")
        self.btn6.setObjectName("save_next")
        self.btn6.clicked.connect(self.showsvm)
        self.btn6.setMaximumSize(100, 36)
        self.btn6.setMinimumSize(100, 36)
        
        self.btn7=QPushButton("Attendance")
        self.btn7.setObjectName("save_next")
        self.btn7.clicked.connect(self.showtop4)
        self.btn7.setMaximumSize(100, 36)
        self.btn7.setMinimumSize(100, 36)

        self.label=QLabel()
        self.label.setMaximumSize(1200, 800)
        self.label.setMinimumSize(1200, 800)
        self.label.setScaledContents(True)

        left = QHBoxLayout()
        left.setSpacing(10)
        left.setContentsMargins(10, 10, 10, 10)
        left.addWidget(self.btn)
        
        left.addWidget(self.btn2)
        left.addWidget(self.btn3)
        left.addWidget(self.btn5)
        
        left.addWidget(self.btn4)
        left.addWidget(self.btn6)
        left.addWidget(self.btn7)
        left.addStretch(1)
        left.addWidget(self.btn3)
        
        bot = QHBoxLayout()
        bot.setContentsMargins(10, 10, 10, 10)
        bot.addStretch(1)
        bot.addWidget(self.label)
        bot.addStretch(1)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 0, 0)
       
        layout.addLayout(left)
        layout.addLayout(bot)
        layout.addStretch(1)
        
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.show_image)

        self.setStyleSheet('''QPushButton:focus{padding: -1;}
                    /* ????? */
                    
                    QGroupBox {
                        height: 48px;
                        width: 1440px;
                        border: none;
                        background: #FFFFFF;
                    }
                    
                    
                    QLabel {
                        font-size: 18px;
                        color: #333333;
                    }
                    
                    QLabel#label_statusv {
                        font-size: 18px;
                        color: #333333;
                    }

                    QHeaderView::section {
                        height: 42px;
                        padding: 4px;
                        border: 1px solid #E0DEDE;
                        border-top: 0px;
                        border-left: 0px;
                        border-right: 0px;
                        background: #FFFFFF;
                        color: #858D8D;
                        font-size: 14px;
                    }
                    
                    QTableWidget {
                        background: #FFFFFF;
                        border-radius: 1px;
                        font-size: 14px;
                        color: #283637;
                    }
                    
                    QTableWidget::item:selected {
                        background: #EDF4F4;
                        border-radius: 1px;
                        color: #283637;
                    }
                    
                    QCheckBox {
                        font-size: 14px;
                    }
                    
                    QCheckBox::indicator::unchecked {   
                        image: url(./resource/chkbox_unchk.png);
                    }
                    
                    QCheckBox::indicator::checked { 
                        image: url(./resource/chkbox_chk.png);
                    }
                    
                    QCheckBox::indicator::unchecked::hover {   
                        image: url(./resource/chkbox_unchk_hover.png);
                    }
                    
                    QCheckBox::indicator::unchecked::disabled {   
                        image: url(./resource/chkbox_unchk_disable.png);
                    }
                    
                    
                    
                    QTextEdit{
                        padding-left: 3px;
                        font-size: 18px;
                        background: #FFFFFF;
                        border: 1px solid #D5D5D5;
                        border-radius: 3px 3px 3px 0 3px;
                    }
                    
                    QLineEdit{
                        background: #FFFFFF;
                        border: 1px solid #D5D5D5;
                        border-radius: 3px;
                        padding-left:3px;
                        font-size: 18px;
                    }
                    
                    QLineEdit:disabled {
                        background: #E8E8E9;
                        border: 1px solid #D5D5D5;
                        border-radius: 3px;
                    }
                    
                    
                    QScrollBar{
                        width:10px;
                        background-color:#E1E4E3;
                        padding-left:2px;
                        padding-right:2px;
                        padding-top:0px;
                        padding-bottom:0px;
                    } 
                    
                    QScrollBar::handle{
                        background:#F8F9FB;
                        border-radius:3px;
                    } 
                    QScrollBar::add-page{
                      background-color:#E1E4E3;
                    }
                    QScrollBar::sub-page{
                        background-color:#E1E4E3;
                    }
                    QScrollBar::sub-line 
                    { 
                        background:#E1E4E3; 
                        border:none;
                    }
                    QScrollBar::add-line 
                    { 
                        background:#E1E4E3; 
                        border:none;
                    }
                    
                    QScrollBar:horizontal {
                        height:10px;
                        background-color:#E1E4E3;
                        padding-top:2px;
                        padding-bottom:2px;
                        padding-left:0px;
                        padding-right:0px;
                    }
                    QScrollBar::handle:horizontal {
                        background:#F8F9FB;
                        border-radius:3px;
                    }
                    
                    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                        background:#E1E4E3; 
                        border:none;
                    }
                    
                    QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal {
                        border: none;
                        width: 2px;
                        height: 2px;
                        background: white;
                    }
                    
                    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                        background-color:#E1E4E3;
                    }
                    QScrollArea{
                        border:none;
                    }
                    
                    QPushButton#save{
                        background: #FFFFFF;
                        border: 1px solid #01A7B2;
                        border-radius: 3px;
                        height: 32px;
                        width: 80px;
                        font-size: 18px;
                        color: #01A7B2;
                    }
                    
                    QPushButton#save::hover{
                        background: #DAFBFD;
                    }
                    
                    QPushButton#save::disabled{
                        background: #FFFFFF;
                        border: 1px solid #BCC5BD;
                        color: #BCC5BD;
                    }
                    
                    QPushButton#save_next {
                        background: #01A7B2;
                        border-radius: 3px;
                        font-size: 14px;
                        color: #FFFFFF;
                    }
                    
                    QPushButton#save_next::hover {
                        background: #00858E;
                    }
                    QPushButton#save_next::disabled{
                        background: #639496;
                        border: 1px solid #639496;
                    }
                    ''')

    def train_data(self):
        path = 'data/test/'  # file path
        for i in range(0, 100):  # Create a file and loop 99-0 times
            k = "%02d" % i  # Two digits, if not enough, add 0 to the previous one
            file_name = path + str(k)  # Name the file:  path + file label (take label as name)
            os.makedirs(file_name)  # Create the file
            mkpath = file_name + '/'  # Drill down the path
            #name = mkpath+"temp_%s.jpg" % (str(i))
            for n in range(0, 100):
                name = mkpath + "temp_%s.jpg" % (str(i))
                ret, readFrame = self.capture.read()
                cv2.imwrite(name, readFrame)
                n=n+1
                time.sleep(0.1)
        i = i + 1

    
 # def loadmode(self):
 #    self.model = load_model('model/classify.model')

    def showsvm(self):
        self.top=Top()
        self.top.show()
    
    def showtop2(self):
        self.top2=Top2()
        self.top2.show()
    
    def showtop3(self):
        self.top3=Top3()
        self.top3.show()
    def showtop4(self):
        self.top4=Top4()
        self.top4.show()
    def capFrame(self):
        #Get pictures from the video stream
        ret, readFrame = self.capture.read()
        return readFrame
    def openimage(self):
        self.capture = cv2.VideoCapture(0)
        self._timer.start(0)
    def convertFrame(self):
        ret, readFrame=self.capture.read()
        if(ret==True):
            self.currentFrame=cv2.cvtColor(readFrame,cv2.COLOR_BGR2RGB)
        else:
            return None
        try:
            height,width=self.currentFrame.shape[:2]
            img=QImage(self.currentFrame,
                              width,
                              height,
                              QImage.Format_RGB888)
            img=QPixmap.fromImage(img)
            return img
        except:
            return None
    def show_image(self):
        try:
            self.label.setPixmap(self.convertFrame())
        except TypeError:
            pass
    def closeimage(self):
        self._timer.stop()
        self.capture.release()
        self.destroyWindows()

class widget(FrameBase):
    def __init__(self,parent=None,message=''):
        super(widget, self).__init__()
        self.setObjectName('Warnbox')
        self.parent = parent
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.header = WarnHeader(self)
        self.bottom = WarnBottom(self)
        self.setStyleSheet('''
                QLabel {
                    font-size: 14px;
                    color: #333333;
                }
                
                QLabel#warn_line {
                    border: 1px solid #E0DEDE;
                    border-top: 0px;
                    border-left: 0px;
                    border-right: 0px;
                }
                
                QLabel#warn_img {
                    border-radius: 10px;
                    background-image: url(resource/warning.png);
                    background-repeat: no-repeat;
                    height: 20px;
                    width: 20px;
                    border: none;
                }
                QLabel#ask_img {
                    border-radius: 10px;
                    background-image: url(resource/ask.png);
                    background-repeat: no-repeat;
                    height: 20px;
                    width: 20px;
                    border: none;
                }
                
                QLabel#WarnLabel {
                    font-size: 18px;
                    color: #333333;
                }
                
                QLabel#warn_msg {
                    font-size: 16px;
                    color: #333333;
                }
                QDialog {
                    background: #FFFFFF;
                    border: 1px solid rgba(102, 102, 102, 61%);
                    border-radius: 8px;
                }
                
                QFrame#WarnHeader {
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                    background: #F3F4F7;
                }
                
                QFrame#WarnBottom {
                    background: #FFFFFF;
                    border-bottom-left-radius: 8px;
                    border-bottom-right-radius: 8px;
                }
                
                QPushButton#MessageNo {
                    background: #FFFFFF;
                    border: 1px solid #D5D5D5;
                    border-radius: 3px;
                    color: #666666;
                    font-size: 14px;
                }
                
                
                QPushButton#MessageNo::hover {
                    background: #F3F3F2;
                }
                
                
                QPushButton#MessageOk {
                    background: #01A7B2;
                    border-radius: 3px;
                    color: #FFFFFF;
                    font-size: 14px;
                }
                
                QPushButton#MessageOk::hover {
                    background: #00858E;
                }
                
                QPushButton#closeButton {
                    color: #828487;
                    border: none;
                    font-size: 30px;
                }
                
                QPushButton#closeButton::hover {
                    background: red;
                    color: white;
                    border: none;
                }
                ''')

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.header)
        self.mainLayout.addWidget(self.bottom)
        self.mainLayout.setStretchFactor(self.header, 48)
        self.mainLayout.setStretchFactor(self.bottom, 180)

        self.mainLayout.setSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.setLayout(self.mainLayout)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)
    mainwin = widget()
    mainwin.show()
    sys.exit(app.exec_())
