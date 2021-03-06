from tkinter import *
from tkinter import filedialog
import tkinter as tk
import os 
from PIL import Image, ImageTk
import numpy as np
import cv2



cap = cv2.VideoCapture(0)

def showimage():
    fln = filedialog.askopenfilename(initialdir=os.getcwd(),title="Select Image Files", filetypes=(("JPG File","*.jpg"), ("PNG File", "*.png"),("ALL Files", "*.*")))
    img = Image.open(fln)
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img
    
    
def showvideo():
    
   

# Playing video from file:
    cap = cv2.VideoCapture(0)

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')

    currentFrame = 0
    while(True):
        
    # Capture frame-by-frame
        ret, frame = cap.read()

    # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture

    

    
root = Tk()


frm = Frame(root)
frm.pack(side=BOTTOM, padx=15, pady=15) 


lbl= Label(root) 
lbl.pack()



btn1 = Button(frm, text="Browse Image", command=showimage)
btn1.pack(side=tk.LEFT)

btn2 = Button(frm, text="Open Cam", command=showvideo)
btn2.pack(side=tk.LEFT)

btn3 = Button(frm, text="Exit", command=lambda: exit())

btn3.pack(side=tk.LEFT, padx=10)
cap.release()
cv2.destroyAllWindows()

import numpy as np
import tensorflow as tf
from tensorflow import keras

image = keras.preprocessing.image
model = keras.models.load_model('CNNmodel.h5')
#path to any image to be predicted
path = 'test/Abax parallelepipedus (Piller & Mitterpacher, 1783)/' + 'd125s0001' + '.jpg'
img = image.load_img(name, target_size=(256, 256))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
#[x] can be an array of images 
images = np.vstack([x])
a=0
clas=["Abax parallelepipedus (Piller & Mitterpacher, 1783)","Acupalpus brunnipes (Sturm, 1825)","Acupalpus dorsalis (Fabricius, 1787)","Acupalpus dubius Schilsky, 1888","Acupalpus exiguus Dejean, 1829","Acupalpus flavicollis (Sturm, 1825)","Acupalpus meridianus (Linnaeus, 1760)","Aepus marinus (Stroem, 1783)","Aepus robinii (LaboulbŠne, 1849)","Agonum emarginatum (Gyllenhal, 1827)","Agonum ericeti (Panzer, 1809)","Agonum fuliginosum (Panzer, 1809)","Agonum gracile Sturm, 1824","Agonum marginatum (Linnaeus, 1758)","Agonum micans (Nicolai, 1822)","Agonum muelleri (Herbst, 1784)","Agonum nigrum Dejean, 1828","Agonum piceum (Linnaeus, 1758)","Agonum sexpunctatum (O.F.Mller, 1776)","Agonum thoreyi Dejean, 1828","Agonum versutum Sturm, 1824","Agonum viduum (Panzer, 1796)","Amara aenea (DeGeer, 1774)","Amara anthobia A. & G.B.Villa, 1833","Amara apricaria (Paykull, 1790)","Amara aulica (Panzer, 1796)","Amara bifrons (Gyllenhal, 1810)","Amara communis (Panzer, 1797)","Amara consularis (Duftschmid, 1812)","Amara convexior Stephens, 1828","Amara curta Dejean, 1828","Amara eurynota (Panzer, 1796)","Amara famelica Zimmermann, 1832","Amara familiaris (Duftschmid, 1812)","Amara fulva (O.F.Mller, 1776)","Amara infima (Duftschmid, 1812)","Amara lucida (Duftschmid, 1812)","Amara lunicollis Schiodte, 1837","Amara ovata Fabricius, 1792","Amara plebeja (Gyllenhal, 1810)","Amara similata (Gyllenhal, 1810)","Amara spreta Dejean, 1831","Amara tibialis Paykull, 1798","Anchomenus dorsalis (Pontoppidan, 1763)","Anisodactylus binotatus (Fabricius, 1787)","Anisodactylus nemorivagus (Duftschmid, 1812)","Anisodactylus poeciloides (Stephens, 1828)","Anthracus consputus (Duftschmid, 1812)","Asaphidion curtum (Heyden, 1870)","Asaphidion flavipes (Linnaeus, 1760)","Asaphidion pallipes (Duftschmid, 1812)","Asaphidion stierlini (Heyden, 1880)","Badister bullatus (Schrank, 1798)","Badister dilatatus Chaudoir, 1837","Badister sodalis (Duftschmid, 1812)","Badister unipustulatus Bonelli, 1813","Batenus livens (Gyllenhal, 1810)","Bembidion aeneum Germar, 1824","Bembidion articulatum (Panzer, 1796)","Bembidion assimile Gyllenhal, 1810","Bembidion atrocaeruleum (Stephens, 1828)","Bembidion biguttatum Motschulsky, 1850","Bembidion bipunctatum (Linnaeus, 1760)","Bembidion bruxellense Wesmael, 1835","Bembidion bualei Jacquelin du Val, 1852","Bembidion clarkii (Dawson, 1849)","Bembidion decorum (Panzer, 1799)","Bembidion deletum Audinet-Serville, 1821","Bembidion dentellum (Thunberg, 1787)","Bembidion doris (Panzer, 1796)","Bembidion ephippium (Marsham, 1802)","Bembidion femoratum Sturm, 1825","Bembidion fluviatile Dejean, 1831","Bembidion fumigatum (Duftschmid, 1812)","Bembidion geniculatum Heer, 1837","Bembidion gilvipes Sturm, 1825","Bembidion guttula (Fabricius, 1792)","Bembidion illigeri Netolitzky, 1914","Bembidion iricolor Bedel, 1879","Bembidion lampros (Herbst, 1784)","Bembidion laterale (Leach, 1819)","Bembidion litorale (G.A.Olivier, 1790)","Bembidion lunatum (Duftschmid, 1812)","Bembidion lunulatum (Geoffroy, 1785)","Bembidion mannerheimii C.R.Sahlberg, 1827","Bembidion maritimum (Motschulsky, 1850)","Bembidion minimum (Fabricius, 1792)","Bembidion monticola Sturm, 1825","Bembidion nigricorne Gyllenhal, 1827","Bembidion nigropiceum (Marsham, 1802)","Bembidion normannum Dejean, 1831","Bembidion obliquum (Linnaeus, 1767)","Bembidion obtusum Audinet-Serville, 1821","Bembidion pallidipenne (Illiger, 1802)","Bembidion prasinum (Duftschmid, 1812)","Bembidion properans (Stephens, 1828)","Bembidion punctulatum Drapiez, 1820","Bembidion quadrimaculatum (Linnaeus, 1760)","Bembidion quadripustulatum Audinet-Serville, 1821","Bembidion saxatile Gyllenhal, 1827","Bembidion schueppelii Dejean, 1831","Bembidion semipunctatum (Donovan, 1806)","Bembidion stephensii Crotch, 1866","Bembidion stomoides Dejean, 1831","Bembidion testaceum (Duftschmid, 1812)","Bembidion tetracolum Say, 1823","Bembidion tibiale (Duftschmid, 1812)","Bembidion varium (G.A.Olivier, 1795)","Blemus discus (Fabricius, 1792)","Blethisa multipunctata (Linnaeus, 1758)","Brachinus crepitans (Linnaeus, 1758)","Bradycellus caucasicus (Chaudoir, 1846)"," Bradycellus harpalinus (Audinet-Serville, 1821)","Bradycellus ruficollis (Stephens, 1828)","Bradycellus sharpi Joy, 1912","Bradycellus verbasci (Duftschmid, 1812)","Broscus cephalotes (Linnaeus, 1758)","Calathus ambiguus (Paykull, 1790)","Calathus cinctus Motschulsky, 1850","Calathus erratus (C.R.Sahlberg, 1827)","Calathus fuscipes (Goeze, 1777)","Calathus melanocephalus (Linnaeus, 1758)","Calathus micropterus (Duftschmid, 1812)","Calathus mollis (Marsham, 1802)","Calathus rotundicollis Dejean, 1828","Callistus lunatus (Fabricius, 1775)","Calodromius spilotus (Illiger, 1798)","Calosoma inquisitor (Linnaeus, 1758)","Carabus arvensis Herbst, 1784","Carabus clatratus Linnaeus, 1760","Carabus glabratus Paykull, 1790","Carabus granulatus Linnaeus, 1758","Carabus monilis Fabricius, 1792","Carabus nemoralis O.F.Mller, 1764","Carabus nitens Linnaeus, 1758","Carabus problematicus Herbst, 1786","Carabus violaceus Linnaeus, 1758","Chlaenius nigricornis (Fabricius, 1787)","Chlaenius vestitus (Paykull, 1790)","Cicindela campestris Linnaeus, 1758","Cicindela hybrida Linnaeus, 1758","Cicindela maritima Dejean, 1822","Cicindela sylvatica Linnaeus, 1758","Clivina collaris (Herbst, 1784)","Clivina fossor (Linnaeus, 1758)","Curtonotus convexiusculus (Marsham, 1802)","Cychrus caraboides (Linnaeus, 1758)","Cylindera germanica (Linnaeus, 1758)","Cymindis axillaris (Fabricius, 1794)","Cymindis vaporariorum (Linnaeus, 1758)","Demetrias atricapillus (Linnaeus, 1758)","Demetrias imperialis (Germar, 1824)","Demetrias monostigma Leach, 1819","Dicheirotrichus cognatus (Gyllenhal, 1827)","Dicheirotrichus gustavii Crotch, 1871","Dicheirotrichus obsoletus (Dejean, 1829)","Dicheirotrichus placidus (Gyllenhal, 1827)","Dromius agilis (Fabricius, 1787)","Dromius angustus Brull‚, 1834","Dromius meridionalis Dejean, 1825","Dromius quadrimaculatus (Linnaeus, 1758)","Drypta dentata (P.Rossi, 1790)","Dyschirius aeneus (Ahrens, 1830)","Dyschirius globosus (Herbst, 1784)","Dyschirius impunctipennis Dawson, 1854","Dyschirius nitidus (Dejean, 1825)","Dyschirius obscurus (Gyllenhal, 1827)","Dyschirius politus (Dejean, 1825)","Dyschirius salinus Schaum, 1843","Dyschirius thoracicus (Zetterstedt, 1840)","Dyschirius tristis Stephens, 1827","Elaphropus walkerianus (Sharp, 1913)","Elaphrus cupreus Duftschmid, 1812","Elaphrus lapponicus Gyllenhal, 1810","Elaphrus riparius (Linnaeus, 1758)","Elaphrus uliginosus Fabricius, 1792","Harpalus affinis Ballion, 1878","Harpalus anxius (Duftschmid, 1812)","Harpalus attenuatus Stephens, 1828","Harpalus dimidiatus (P.Rossi, 1790)","Harpalus laevipes Zetterstedt, 1828","Harpalus latus Linnaeus, 1758","Harpalus neglectus Audinet-Serville, 1821","Harpalus pumilus Sturm, 1818","Harpalus rubripes (Duftschmid, 1812)","Harpalus rufipalpis Sturm, 1818","Harpalus rufipes DeGeer, 1774","Harpalus serripes (Quensel, 1806)","Harpalus servus (Duftschmid, 1812)","Harpalus smaragdinus (Duftschmid, 1812)","Harpalus tardus (Panzer, 1796)","Harpalus tenebrosus Dejean, 1829","Laemostenus complanatus (Dejean, 1828)","Laemostenus terricola (Herbst, 1784)","Lebia chlorocephala (J.J.Hoffmann, 1803)","Leistus ferrugineus (Linnaeus, 1758)","Leistus fulvibarbis Dejean, 1826","Leistus spinibarbis (Fabricius, 1775)","Leistus terminatus (Panzer, 1793)","Licinus depressus (Paykull, 1790)","Licinus punctatulus (Fabricius, 1792)","Loricera pilicornis (Fabricius, 1775)","Masoreus wetterhallii (Gyllenhal, 1813)","Microlestes maurus (Sturm, 1827)","Miscodera arctica (Paykull, 1798)","Nebria brevicollis (Fabricius, 1792)","Nebria complanata (Linnaeus, 1767)","Nebria livida (Linnaeus, 1758)","Nebria rufescens (Stroem, 1768)","Nebria salina Fairmaire & LaboulbŠne, 1854","Notiophilus aquaticus (Linnaeus, 1758)","Notiophilus biguttatus (Fabricius, 1779)","Notiophilus germinyi Fauvel, 1863","Notiophilus palustris Duftschmid, 1812","Notiophilus quadripunctatus Dejean, 1826","Notiophilus rufipes Curtis, 1829","Notiophilus substriatus Waterhouse, 1833","Ocys harpaloides (Audinet-Serville, 1821)","Ocys quinquestriatus (Gyllenhal, 1810)","Odacantha melanura (Linnaeus, 1767)","Olisthopus rotundatus (Paykull, 1790)","Oodes helopioides (Fabricius, 1792)","Ophonus ardosiacus (Lutshnik, 1922)","Ophonus azureus (Fabricius, 1775)","Ophonus cordatus (Duftschmid, 1812)","Ophonus laticollis Mannerheim, 1825","Ophonus melletii (Heer, 1837)","Ophonus puncticeps Stephens, 1828","Ophonus rufibarbis (Fabricius, 1792)","Ophonus rupicola (Sturm, 1818)","Ophonus schaubergerianus (Puel, 1937)","Oxypselaphus obscurus (Herbst, 1784)","Panagaeus bipustulatus (Fabricius, 1775)","Panagaeus cruxmajor (Linnaeus, 1758)","Paradromius linearis (G.A.Olivier, 1795)","Paradromius longiceps (Dejean, 1826)","Paranchus albipes (Fabricius, 1796)","Patrobus assimilis Chaudoir, 1844","Patrobus atrorufus (Stroem, 1768)","Patrobus septentrionis Dejean, 1828","Pedius longicollis (Duftschmid, 1812)","Pelophila borealis (Paykull, 1790)","Perileptus areolatus (Creutzer, 1799)","Philorhizus melanocephalus (Dejean, 1825)","Philorhizus notatus (Stephens, 1827)","Platyderus depressus (Audinet-Serville, 1821)","Platynus assimilis (Paykull, 1790)","Poecilus cupreus (Linnaeus, 1758)","Poecilus kugelanni (Panzer, 1797)","Poecilus lepidus (Leske, 1785)","Poecilus versicolor (Sturm, 1824)","Pogonus chalceus (Marsham, 1802)","Pogonus littoralis (Duftschmid, 1812)","Pogonus luridipennis (Germar, 1823)","Polistichus connexus (Geoffroy, 1785)","Pterostichus adstrictus Eschscholtz, 1823","Pterostichus aethiops (Panzer, 1796)","Pterostichus anthracinus (Illiger, 1798)","Pterostichus cristatus (L.Dufour, 1820)","Pterostichus diligens (Sturm, 1824)","Pterostichus gracilis (Dejean, 1828)","Pterostichus macer (Marsham, 1802)","Pterostichus madidus (Fabricius, 1775)","Pterostichus melanarius (Illiger, 1798)","Pterostichus minor (Gyllenhal, 1827)","Pterostichus niger (Schaller, 1783)","Pterostichus nigrita (Paykull, 1790)","Pterostichus oblongopunctatus (Fabricius, 1787)","Pterostichus quadrifoveolatus Letzner, 1852","Pterostichus rhaeticus Heer, 1837","Pterostichus strenuus (Panzer, 1796)","Pterostichus vernalis (Panzer, 1796)","Sericoda quadripunctata (DeGeer, 1774)","Stenolophus mixtus (Herbst, 1784)","Stenolophus skrimshiranus Stephens, 1828","Stenolophus teutonus (Schrank, 1781)","Stomis pumicatus (Panzer, 1796)","Syntomus foveatus (Geoffroy, 1785)","Syntomus obscuroguttatus (Duftschmid, 1812)","Syntomus truncatellus (Linnaeus, 1760)","Synuchus vivalis (Illiger, 1798)","Tachys bistriatus (W.J.MacLeay, 1871)","Tachys micros (Fischer von Waldheim, 1828)","Tachys scutellaris Stephens, 1828","Trechoblemus micros (Herbst, 1784)","Trechus fulvus Dejean, 1831","Trechus obtusus Erichson, 1837","Trechus quadristriatus (Schrank, 1781)","Trechus rubens (Fabricius, 1792)","Trechus secalis (Paykull, 1790)","Zabrus tenebrioides (Goeze, 1777)"]
classes = model.predict(images, batch_size=64)
for i in range (0,289):
    if classes[0][i]<classes[0][i+1]:
        a=i+1
print(len(clas))
print(clas[a-1])
# Desired output. Charts with training and validation metrics. No crash :)




root.title("Image Browser")
root.geometry("400x350")
root.mainloop()

    
