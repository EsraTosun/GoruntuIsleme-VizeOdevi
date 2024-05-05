from PyQt5.QtWidgets import QTabWidget,QAction, QFileDialog
from PyQt5.QtGui import  QColor


import cv2
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QLineEdit, QMessageBox
from math import cos, sin, radians

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import numpy as np
from PIL import Image

# Global değişken olarak file_path tanımlanıyor
file_path = None
class Odev1Page(QWidget):
    def __init__(self):
        super().__init__()
        # Histogramı tutacak değişkeni tanımla
        self.histogram_data = None
        self.image = None
        self.filePath = None

        layout = QVBoxLayout(self)

        # Resim göstermek için bir QLabel oluştur
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        # Resmi yüklemek için bir buton oluştur
        self.load_image_button = QPushButton("Resim Yükle")
        self.load_image_button.clicked.connect(self.load_image)

        # Histogram oluşturmak için bir buton oluştur
        self.load_histogram_button = QPushButton("Histogram oluştur")
        self.load_histogram_button.clicked.connect(self.load_histogram)

        # Layout'a widget'ları ekle
        layout.addWidget(self.image_label)
        layout.addWidget(self.load_image_button)
        layout.addWidget(self.load_histogram_button)

        # Histogram grafiği için bir QGraphicsView oluştur
        # self.histogram_view = pg.GraphicsView()
        # layout.addWidget(self.histogram_view)

    def load_image(self):
        global file_path  # global değişkene erişim sağlanıyor
        # Resim seçme işlemi için dosya iletişim kutusunu aç
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Resim dosyaları (*.jpg *.png)")
        file_dialog.setViewMode(QFileDialog.Detail)

        # Kullanıcı bir resim seçerse, resmi yükle
        if file_dialog.exec_():
            self.filePath = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(self.filePath)

            # Resmi boyutlandır
            pixmap = pixmap.scaledToWidth(1000)  # Genişliği 400 piksel olarak ayarla

            self.image_label.setPixmap(pixmap)

            # QPixmap'ı QImage'e dönüştür
            self.image = pixmap.toImage()

    def load_histogram(self):
        if self.image is not None:
            # QImage'i QPixmap'e dönüştür
            pixmap = QPixmap(self.image)

            # QPixmap'i QImage'e dönüştür
            qimage = pixmap.toImage()

            # QImage'i numpy dizisine dönüştür
            height, width = qimage.height(), qimage.width()
            ptr = qimage.bits()  # QByteArray al
            ptr.setsize(qimage.byteCount())  # QByteArray boyutunu ayarla
            arr = np.array(ptr).reshape(height, width, 4)  # 4 kanallı (RGBA) bir görüntü olduğunu varsayıyoruz

            # Görüntüyü BGR renk formatına dönüştür (OpenCV için gereklidir)
            bgr_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

            # Görüntüyü tek boyutlu bir numpy dizisine düzleştir
            flattened_image = bgr_image.ravel()

            # Histogramı oluştur
            plt.hist(flattened_image, bins=256, range=[0, 256])
            plt.show()


        else:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim yükleyin.")
