import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget,QLabel,QMessageBox,QFileDialog
from PyQt5 import QtCore, QtGui, uic, QtWidgets
from PyQt5.QtGui import QImage, QPixmap 
import cv2
from matplotlib import pyplot as plt
from matplotlib.backends.backend_template import FigureCanvas
from matplotlib.figure import Figure
import mahotas
import mahotas.demos

class MyWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('cv.ui', self)
        self.actionopen.triggered.connect(lambda: self.Open())
        self.histogram.clicked.connect(lambda: self.hist_equalization())
        self.choose_filter.activated.connect(lambda: self.Choose_Filter())
        
    def Open(self):
            fileName = QFileDialog.getOpenFileName(self, 'QFileDialog.getOpenFileName()', '',
                                                  'Images (*.png *.jpeg *.jpg *.bmp *.gif)')[0]

            image = QImage(fileName)
            self.normal_photo.setPixmap(QPixmap.fromImage(image))

            self.img = cv2.imread(fileName, 0)
            self.his_img = cv2.imread(fileName, 0)
            
            dft = cv2.dft(np.float32(self.img), flags=cv2.DFT_COMPLEX_OUTPUT)
            self.dft_shift = np.fft.fftshift(dft)
            plt.figure() 
            self.magnitude_spectrum = 20 * np.log(cv2.magnitude(self.dft_shift[:, :, 0], self.dft_shift[:, :, 1]))
            cv2.imwrite('fft_image.jpg',self.magnitude_spectrum)
            fft_img = QImage('fft_image.jpg')
            self.fft_image.setPixmap(QPixmap.fromImage(fft_img))
           
            # HPF_Filter
    def HP_Filter(self):
            rows, cols = self.img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            
            mask = np.ones((rows, cols, 2), np.uint8)
            r = 20
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 0
            
            # apply mask and inverse DFT
            fshift = self.dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])+1)
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            
            plt.figure()
            plt.imshow(fshift_mask_mag, cmap='gray')
            plt.axis('off')
            plt.savefig('FFT_HP_filter.jpg',bbox_inches='tight', pad_inches=0)
            pixmap = QPixmap('FFT_HP_filter.jpg').scaled(332, 300)
            self.fft_filter.setPixmap(pixmap)
            
            plt.figure()
            plt.imshow(img_back, cmap='gray')
            plt.axis('off')
            plt.savefig('HP_filter.jpg',bbox_inches='tight', pad_inches=0)
            pixmap = QPixmap('HP_filter.jpg').scaled(332, 300)
            self.spatial_filter.setPixmap(pixmap)

            # LPF_Filter
    def LP_Filter(self):        
            rows, cols = self.img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 50
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
            mask[mask_area] = 1
            
            # apply mask and inverse DFT
            fshift = self.dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])+1)
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
            
            plt.figure()
            plt.imshow(fshift_mask_mag, cmap='gray')
            plt.axis('off')
            plt.savefig('FFT_LP_filter.jpg',bbox_inches='tight', pad_inches=0)
            pixmap = QPixmap('FFT_LP_filter.jpg').scaled(332, 300)
            self.fft_filter.setPixmap(pixmap)
            
            plt.figure()
            plt.imshow(img_back, cmap='gray')
            plt.axis('off')
            plt.savefig('LP_filter.jpg',bbox_inches='tight', pad_inches=0)
            pixmap = QPixmap('LP_filter.jpg').scaled(332, 300)
            self.spatial_filter.setPixmap(pixmap)


    def Choose_Filter(self):
        if self.choose_filter.currentIndex() ==1:
            self.HP_Filter()
        if self.choose_filter.currentIndex() ==2:
            self.LP_Filter()
        if self.choose_filter.currentIndex() ==3:
            self.Median_Filter()
        if self.choose_filter.currentIndex() == 4:
            self.Laplacian_Filter()
    

    def Laplacian_Filter(self):
            new_img = mahotas.laplacian_2D(self.img)
            plt.figure()
            plt.imshow(new_img, cmap='gray')
            plt.axis('off')
            plt.savefig('laplacian_filter.jpg',bbox_inches='tight', pad_inches=0)
            pixmap = QPixmap('laplacian_filter.jpg').scaled(332, 300)
            self.spatial_filter.setPixmap(pixmap)


    def Median_Filter(self): 
            m, n = self.img.shape
            img_new1 = np.zeros([m, n])
    
            for i in range(1, m - 1):
                for j in range(1, n - 1):
                    temp = [self.img[i - 1, j - 1],
                            self.img[i - 1, j],
                            self.img[i - 1, j + 1],
                            self.img[i, j - 1],
                            self.img[i, j],
                            self.img[i, j + 1],
                            self.img[i + 1, j - 1],
                            self.img[i + 1, j],
                            self.img[i + 1, j + 1]]
    
                    temp = sorted(temp)
                    img_new1[i, j] = temp[4]
    
            img_new1 = img_new1.astype(np.uint8)
            cv2.imwrite('median_filter.jpg', img_new1)
            plt.figure()
            plt.imshow(img_new1, cmap='gray')
            plt.axis('off')
            plt.savefig('median_filter.jpg',bbox_inches='tight', pad_inches=0)
            pixmap = QPixmap('median_filter.jpg').scaled(332, 300)
            self.spatial_filter.setPixmap(pixmap)
            

    def hist_equalization(self):
        unique, count = np.unique(self.his_img, return_counts=True)
        plt.figure()
        plt.bar(unique, count)        
        plt.savefig('Histogram.jpg',bbox_inches='tight', pad_inches=0)
        pixmap = QPixmap('Histogram.jpg')
        self.histogram_label.setPixmap(pixmap)


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = MyWidget()
    application.show()
    app.exec_()

if __name__ == "__main__":
    main()     
