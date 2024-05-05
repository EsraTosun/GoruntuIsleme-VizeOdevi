from PyQt5.QtWidgets import QTabWidget,QAction, QFileDialog
import os

import pandas as pd

import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtGui import QPixmap, QImage

import numpy as np

from skimage import io

# Global değişken olarak file_path tanımlanıyor
file_path = None

class VizePage(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        # Ödev 2 sekmesi içinde başka bir QTabWidget oluştur
        self.inner_tab_widget = QTabWidget()

        # İç sekme widget'ını ödev 2 sayfasının ana layoutuna ekleyin
        layout.addWidget(self.inner_tab_widget)

        # İlk iç sekme oluştur
        self.create_inner_tabs()

    def create_inner_tabs(self):
        # İç sekme 1 oluştur
        inner_tab1 = InnerTab1()

        # İç sekme 2 oluştur
        inner_tab2 = InnerTab2()

        # İç sekme 3 oluştur
        inner_tab3 = InnerTab3()

        # İç sekme 4 oluştur
        inner_tab4 = InnerTab4()

        # İç sekme 5 oluştur
        inner_tab5 = InnerTab5()

        # İç sekme widget'ını içindeki sekme widget'ına ekle
        self.inner_tab_widget.addTab(inner_tab1, "Görüntü Güçlendirme")
        self.inner_tab_widget.addTab(inner_tab2, "Yol Tespiti")
        self.inner_tab_widget.addTab(inner_tab3, "Yüz Tespiti")
        self.inner_tab_widget.addTab(inner_tab4, "Deblurring")
        self.inner_tab_widget.addTab(inner_tab5, "Nesne Özelliği Çıkarma")


class InnerTab1(QWidget):
    def __init__(self):
        super().__init__()
        yolu = "C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/soru1.jpg"
        pencere = GoruntuGuclendirme(self,yolu)

class GoruntuGuclendirme(QWidget):
    def __init__(self, parent, yol):
        super().__init__(parent)
        self.setWindowTitle('Resim Güçlendirme')

        self.parent = parent
        self.yol = yol

        # Kullanıcı arayüzü bileşenlerini oluştur
        self.buton = QPushButton("Resmi Güçlendir")
        self.buton.clicked.connect(self.ResmiGuclendir)


        # Arayüzü düzenle
        layout = QVBoxLayout()
        layout.addWidget(self.buton)
        self.setLayout(layout)

    def ResmiGuclendir(self):

        image = io.imread(self.yol)

        # Görüntüyü 0-1 aralığına normalleştirme
        image = image / 255.0
        print(image)

        # Standart sigmoid fonksiyonu ile kontrast güçlendirme
        image_standard = self.s_curve_contrast(image, self.standard_sigmoid)

        # Yatay kaydırılmış sigmoid fonksiyonu ile kontrast güçlendirme
        image_shifted = self.s_curve_contrast(image, self.shifted_sigmoid)

        # Eğimli sigmoid fonksiyonu ile kontrast güçlendirme
        image_tilted = self.s_curve_contrast(image, self.tilted_sigmoid)


        # Sonuçları görselleştirme
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 4, 1)
        plt.imshow(image)
        plt.title('Orjinal Görüntü')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(image_standard)
        plt.title('Standart Sigmoid')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(image_shifted)
        plt.title('Yatay Kaydırılmış Sigmoid')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(image_tilted)
        plt.title('Eğimli Sigmoid')
        plt.axis('off')

        plt.show()

    def load_image(self):
        pixmap = QPixmap(self.image_path)
        self.label.setPixmap(pixmap)
        self.label.setPixmap(pixmap.scaledToWidth(400))  # Pixmap'i genişliği 400 piksele ölçekle
        self.label.adjustSize()

    def standard_sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def shifted_sigmoid(self, x, c=1):
        return 1 / (1 + np.exp(-c * (x - 0.5)))

    def tilted_sigmoid(self, x, a=1):
        return 1 / (1 + np.exp(-a * (x - 0.5)))

    def s_curve_contrast(self, image, sigmoid_func):
        # Görüntüyü tek boyutlu hale getirme ve sigmoid fonksiyonunu uygulama
        transformed_image = sigmoid_func(image.flatten())

        # Görüntüyü [0, 1] aralığına yeniden ölçeklendirme
        transformed_image = (transformed_image - transformed_image.min()) / (
                transformed_image.max() - transformed_image.min())

        # Yeniden şekillendirme ve 0-255 aralığında piksel değerlerine dönüştürme
        transformed_image = (transformed_image * 255).astype(np.uint8)
        transformed_image = transformed_image.reshape(image.shape)

        return transformed_image

class InnerTab2(QWidget):
    def __init__(self):
        super().__init__()
        yolu = "C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/soru2-1.jpg"
        pencere = YolBulma(self,yolu)

class YolBulma(QWidget):
    def __init__(self, parent, yol):
        super().__init__(parent)
        self.setWindowTitle('Yol Bul')

        self.parent = parent
        self.yol = yol

        # Kullanıcı arayüzü bileşenlerini oluştur
        self.buton = QPushButton("Yol Bul")
        self.buton.clicked.connect(self.yolBul)


        # Arayüzü düzenle
        layout = QVBoxLayout()
        layout.addWidget(self.buton)
        self.setLayout(layout)

    def preprocess_image(self,image):
        # Gaussian bulanıklığı uygula
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        # Morfolojik işlemler için kernel oluştur
        kernel = np.ones((5, 5), np.uint8)

        # Aşınma (erosion) işlemi uygula
        erosion = cv2.erode(blurred, kernel, iterations=1)

        # Erozyon sonrası genişletme (dilation) işlemi uygula
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        return dilation

    def yolBul(self):
        # Görüntüyü yükle
        image = cv2.imread(self.yol)

        # Sonucu göster
        cv2.imshow('Orjinal Goruntu', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Görüntüyü gri tonlamalıya dönüştür
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Görüntüyü ön işleme yap
        preprocessed_image = self.preprocess_image(gray)

        # Kenarları algıla (Canny Edge Detection)
        edges = cv2.Canny(preprocessed_image, 50, 150)

        # Hough Transform'u uygula
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

        # Orijinal görüntü üzerine çizgileri çiz
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Sonucu göster
        cv2.imshow('Islem Sonucu Olusan Goruntu', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class InnerTab3(QWidget):
    def __init__(self):
        super().__init__()
        yolu = "C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/soru2-2-2.jpg"
        pencere = YuzBulma(self,yolu)

class YuzBulma(QWidget):
    def __init__(self, parent, yol):
        super().__init__(parent)
        self.setWindowTitle('Yüz Bulma')

        self.parent = parent
        self.yol = yol

        # Kullanıcı arayüzü bileşenlerini oluştur
        self.buton = QPushButton("Yüz Bulma")
        self.buton.clicked.connect(self.yuzbulma)


        # Arayüzü düzenle
        layout = QVBoxLayout()
        layout.addWidget(self.buton)
        self.setLayout(layout)

    def yuzbulma(self):
        # Haar Cascade sınıflandırıcılarını yükle
        face_cascade = cv2.CascadeClassifier(
            'C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(
            'C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/haarcascade_eye.xml')

        # Resmi yükle
        image = cv2.imread(self.yol)

        # Görüntüyü göster
        cv2.imshow('Orjinal Goruntu', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Yüzleri tespit et
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Her yüz için gözleri tespit et
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]

            # Göz tespiti için minNeighbors parametresini ayarla
            eyes = eye_cascade.detectMultiScale(roi_gray, minNeighbors=25)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Görüntüyü göster
        cv2.imshow('Islem Sonucu Olusan Goruntu', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class InnerTab4(QWidget):
    def __init__(self):
        super().__init__()
        yolu = "C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/soru3.jpg"
        pencere = Deblurring(self,yolu)

class Deblurring(QWidget):
    def __init__(self, parent, yol):
        super().__init__(parent)
        self.setWindowTitle('Deblurring')

        self.parent = parent
        self.yol = yol

        # Kullanıcı arayüzü bileşenlerini oluştur
        self.buton = QPushButton("Deblurring")
        self.buton.clicked.connect(self.deblurblur)


        # Arayüzü düzenle
        layout = QVBoxLayout()
        layout.addWidget(self.buton)
        self.setLayout(layout)

    def deblurblur(self):
        # Görüntüyü yükle
        self.img = cv2.imread(self.yol)

        # Motion blur'u gider
        kernel_size = 10
        angle = 55  # Hareket açısı
        deblurred_image = self.deblur_motion_blur(self.img, kernel_size, angle)

        # Görüntüleri göster
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title('Önceki Görüntü')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(deblurred_image, cv2.COLOR_BGR2RGB))
        plt.title('Motion Blur Düzeltildi')
        plt.axis('off')

        plt.show()

    def deblur_motion_blur(self, image, kernel_size, angle):
        # Hareket bulanıklığı kernel'ini oluştur
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = cv2.getRotationMatrix2D((int((kernel_size - 1) / 2), int((kernel_size - 1) / 2)), angle, 1)
        kernel = cv2.warpAffine(kernel, kernel, (kernel_size, kernel_size))

        # Kernel'i normalize et
        kernel /= kernel_size

        # Motion blur'u kaldır
        deblurred_image = cv2.filter2D(image, -1, kernel)
        deblurred_image = cv2.convertScaleAbs(deblurred_image, alpha=1.9, beta=30)

        return deblurred_image

class InnerTab5(QWidget):
    def __init__(self):
        super().__init__()
        yolu = 'C:/Users/esrat/Dersler/BaharDonemi/DijitalGoruntuIsleme/Vize/Sorular/images/soru4.jpg'
        pencere = NesneOzelligi(self,yolu)

class NesneOzelligi(QWidget):
    def __init__(self, parent, yol):
        super().__init__(parent)
        self.setWindowTitle('Nesne Özelligi Çıkarma')

        self.parent = parent
        self.yol = yol

        # Kullanıcı arayüzü bileşenlerini oluştur
        self.buton = QPushButton("Nesne Özelligi Çıkarma")
        self.buton.clicked.connect(self.nesneOzelligiCikarma)


        # Arayüzü düzenle
        layout = QVBoxLayout()
        layout.addWidget(self.buton)
        self.setLayout(layout)

    def nesneOzelligiCikarma(self):
        image = cv2.imread(self.yol)

        # Yeşil renk aralığını tanımla (düşük ve yüksek sınırlar)
        lower_green = np.array([0, 100, 0], dtype="uint8")
        upper_green = np.array([50, 255, 50], dtype="uint8")

        # Renk uzayını değiştir (BGR'den HSV'ye)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Belirtilen renk aralığına göre maske oluştur
        mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Koyu yeşil bölgeleri bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Boş bir görüntü oluştur
        contour_image = np.zeros_like(image)

        # Konturları bu görüntünün üzerine çiz
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

        # Görüntüyü göster
        cv2.imshow('Islem Sonucu Olusan Goruntu', contour_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Excel tablosu için veri çıkar
        data = []
        k = 0
        # Konturları işle
        for contour in contours:

            moments = cv2.moments(contour)
            if moments["m00"] != 0:  # Sıfıra bölme hatasını önle
                center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
                x, y, w, h = cv2.boundingRect(contour)
                length = w
                width = h
                diagonal = np.sqrt(w ** 2 + h ** 2)

                # Energy ve Entropy hesapla
                mask_kontur = np.zeros_like(mask)
                cv2.drawContours(mask_kontur, [contour], -1, 255, -1)
                moments = cv2.moments(mask_kontur)
                hu_moments = cv2.HuMoments(moments).flatten()
                energy = np.sum(hu_moments[1:] ** 2)
                entropy = -np.sum(hu_moments * np.log(np.abs(hu_moments) + 1e-10))

                # Mean ve Median hesapla
                mean_val = np.mean(hsv_image[mask_kontur == 255])
                median_val = np.median(hsv_image[mask_kontur == 255])

                # Veriyi listeye ekle
                data.append(
                    {'No': k + 1, 'Center': center, 'Length': f"{length} px", 'Width': f"{width} px",
                     'Diagonal': f"{diagonal} px",
                     'Energy': energy, 'Entropy': entropy, 'Mean': mean_val, 'Median': median_val})
                k += 1

        # Pandas DataFrame oluştur
        df = pd.DataFrame(data, columns=['No', 'Center', 'Length', 'Width', 'Diagonal', 'Energy', 'Entropy', 'Mean',
                                         'Median'])

        # Yeni bir dosya adı oluştur
        output_file = 'output.xlsx'
        counter = 1
        while os.path.exists(output_file):
            output_file = f'output_{counter}.xlsx'
            counter += 1

        # DataFrame'i Excel dosyasına yaz
        df.to_excel(output_file, index=False)

        # Verileri yazdır
        print(df)
        print(f"Excel dosyası başarıyla oluşturuldu: {output_file}")







