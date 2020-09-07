import cv2
import numpy as np
import os

Kamera = cv2.VideoCapture(0)
kernel = np.ones((12,12),np.uint8)

def ResimFarkBul(Resim1,Resim2):
    Resim2 = cv2.resize(Resim2,(Resim1.shape[1],Resim1.shape[0]))
    Fark_Resim = cv2.absdiff(Resim1,Resim2)
    Fark_Sayı = cv2.countNonZero(Fark_Resim)
    return Fark_Sayı

def VeriYükle():
    Veri_İsimler = []
    Veri_Resimler = []
    Dosyalar = os.listdir("Veri/")
    for Dosya in Dosyalar:
        Veri_İsimler.append(Dosya.replace(".jpg",""))
        Veri_Resimler.append(cv2.imread("Veri/"+Dosya,0))
    return Veri_İsimler,Veri_Resimler



def Sınıflandır (Resim,Veri_İsimler,Veri_Resimler):
    Min_Index = 0
    Min_Deger = ResimFarkBul(Resim,Veri_Resimler[0])
    for t in range(len(Veri_İsimler)):
        Fark_Değer = ResimFarkBul(Resim,Veri_Resimler[t])
    if (Fark_Değer < Min_Deger):
        Min_Deger = Fark_Değer
        Min_Index = t
    return Veri_İsimler[Min_Index]

Veri_İsimler, Veri_Resimler = VeriYükle()
#Veri_Resim1 = cv2.imread("Veri/bir.jpg",0)

while True:
    ret, Kare = Kamera.read()
    Kesilmiş_Kare = Kare[0:250,0:250]
    Kesilmiş_Kare_Gri = cv2.cvtColor(Kesilmiş_Kare, cv2.COLOR_BGR2GRAY)
    Kesilmiş_Kare_HSV = cv2.cvtColor(Kesilmiş_Kare,cv2.COLOR_BGR2HSV)

    Alt_Değerler = np.array([0,30,60])
    Üst_Değerler = np.array([40,255,255])

    Renk_Filtresi_Sonucu = cv2.inRange(Kesilmiş_Kare_HSV,Alt_Değerler,Üst_Değerler)
    Renk_Filtresi_Sonucu = cv2.morphologyEx(Renk_Filtresi_Sonucu,cv2.MORPH_CLOSE,kernel)
    Renk_Filtresi_Sonucu = cv2.dilate(Renk_Filtresi_Sonucu, kernel, iterations=1)

    Sonuç = Kesilmiş_Kare.copy()

    cnts, hierarchy = cv2.findContours(Renk_Filtresi_Sonucu,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    Max_Genişlik = 0
    Max_Uzunluk = 0
    Max_Index = -1
    for t in range(len(cnts)):
        cnt = cnts[t]
        x,y,w,h = cv2.boundingRect(cnt)
        if(w>Max_Genişlik and h>Max_Uzunluk):
            Max_Uzunluk = h
            Max_Genişlik = w
            Max_Index = t

    if(len(cnts)>0):
        x,y,w,h = cv2.boundingRect(cnts[Max_Index])
        cv2.rectangle(Sonuç,(x,y),(x+w,y+h),(0,255,0),2)
        El_Resim = Renk_Filtresi_Sonucu[y:y+h,x:x+w]
        cv2.imshow("El Resim", El_Resim)
        print(Sınıflandır(El_Resim,Veri_İsimler,Veri_Resimler))

    cv2.imshow("Kare",Kare)
    cv2.imshow("Kesilmiş Kare",Kesilmiş_Kare)
    cv2.imshow("Renk Filtresi Sonucu", Renk_Filtresi_Sonucu)
    cv2.imshow("Sonuç",Sonuç)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

Kamera.release()
cv2.destroyWindows()