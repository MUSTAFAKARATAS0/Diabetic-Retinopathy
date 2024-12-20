# Gerekli kütüphaneleri içe aktarın
import pandas as pd  # Veri işleme
import matplotlib.pyplot as plt  # Grafik çizimi
import seaborn as sns  # Gelişmiş veri görselleştirme
sns.set(style='darkgrid')  # Grafiklerin arka planını koyu ızgara stiline ayarlayın
import copy  # Nesne kopyalama
import os  # Dosya ve dizin işlemleri
import torch  # Derin öğrenme çerçevesi
from PIL import Image  # Görüntü işleme
from torch.utils.data import Dataset  # Veri seti sınıfları
import torchvision  # Görüntü işleme kütüphanesi
import torchvision.transforms as transforms  # Görüntü dönüşümleri
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Öğrenme oranı azaltıcı
import torch.nn as nn  # Sinir ağı modülleri
from torchvision.datasets import ImageFolder  # Klasör tabanlı veri yükleme
from torchsummary import summary  # Model özetini görüntülemek için
import torch.nn.functional as F  # Sinir ağı aktivasyonları ve kayıp fonksiyonları
from sklearn.metrics import classification_report  # Sınıflandırma raporu oluşturma
import itertools  # Kombinasyon ve permütasyon işlemleri
from tqdm import tqdm  # Eğitim ilerleme çubuğu
from torch import optim  # Optimizasyon algoritmaları
import warnings  # Uyarıları yönetme
import numpy as np  # Sayısal hesaplama kütüphanesi
warnings.filterwarnings('ignore')  # Uyarıları devre dışı bırak

# Görselleştirme ve HTML için IPython kütüphanesi
from IPython.core.display import display, HTML, Javascript  



# Veri ön işleme için görüntü dönüşümlerini tanımla
transform = transforms.Compose(
    [
        # Görüntüyü 255x255 piksel boyutuna yeniden ölçeklendir
        transforms.Resize((255, 255)),
        
        # Görüntüyü %50 olasılıkla yatay olarak çevir
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Görüntüyü %50 olasılıkla dikey olarak çevir
        transforms.RandomVerticalFlip(p=0.5),
        
        # Görüntüyü -30 ile +30 derece arasında rastgele döndür
        transforms.RandomRotation(30),
        
        # Görüntüyü PyTorch tensörüne dönüştür ve piksel değerlerini [0, 1] aralığına çek
        transforms.ToTensor(),
        
        # Görüntüyü belirli bir ortalama ve standart sapmaya göre normalize et (ImageNet için yaygın değerler)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


# Eğitim (train), doğrulama (validation) ve test verilerini tanımla

# Eğitim veri kümesi (train set):
# 'train' klasöründeki görüntüleri yükler ve tanımlanan dönüşümleri uygular
train_set = torchvision.datasets.ImageFolder(
    "C:\\Users\\karat\\Desktop\\retino\\train", 
    transform=transform
)
train_set.transform  # Eğitim veri kümesine dönüşümleri uygula

# Doğrulama veri kümesi (validation set):
# 'valid' klasöründeki görüntüleri yükler ve tanımlanan dönüşümleri uygular
val_set = torchvision.datasets.ImageFolder(
    "C:\\Users\\karat\\Desktop\\retino\\valid", 
    transform=transform
)
val_set.transform  # Doğrulama veri kümesine dönüşümleri uygula

# Test veri kümesi (test set):
# 'test' klasöründeki görüntüleri yükler ve tanımlanan dönüşümleri uygular
test_set = torchvision.datasets.ImageFolder(
    "C:\\Users\\karat\\Desktop\\retino\\test", 
    transform=transform
)
test_set.transform  # Test veri kümesine dönüşümleri uygula


# Eğitim veri setinden bazı görüntülerin görselleştirilmesi

# Görüntü sınıflarını etiketlere dönüştürmek için bir sözlük tanımla
CLA_label = {
        0: 'DR',       # DR: Diabetic Retinopathy (Diyabetik Retinopati)
        1: 'No_DR',    # No_DR: Diyabetik Retinopati yok
}

# Görsellerin yerleştirileceği bir figür tanımla (10x10 boyutunda)
figure = plt.figure(figsize=(10, 10))

# Görselleri 4x4'lük bir ızgarada göster
cols, rows = 4, 4

# Her bir görsel için döngü başlat
for i in range(1, cols * rows + 1):
    # Eğitim veri setinden rastgele bir örnek seç
    sample_idx = torch.randint(len(train_set), size=(1,)).item()
    img, label = train_set[sample_idx]  # Görüntü ve etiket al
    
    # Figüre bir alt grafik (subplot) ekle
    figure.add_subplot(rows, cols, i)
    
    # Görüntünün sınıf adını başlık olarak ekle
    plt.title(CLA_label[label])
    
    # Grafik eksenlerini gizle
    plt.axis("off")
    
    # Tensörü numpy dizisine çevir ve kanal sırasını (C, H, W -> H, W, C) değiştir
    img_np = img.numpy().transpose((1, 2, 0))
    
    # Piksel değerlerini [0, 1] aralığına kırp (normalize edilmiş görüntüleri doğru göstermek için)
    img_valid_range = np.clip(img_np, 0, 1)
    
    # Görüntüyü matplotlib ile göster
    plt.imshow(img_valid_range)

# Tüm görsellerin başlığı olarak genel bir başlık ekle
plt.suptitle('Retinopathy Images', y=0.95)

# Görselleri göster
plt.show()


# Eğitim, doğrulama ve test veri kümelerini yükleme ve batch işlemleri için DataLoader tanımlama

# Batch boyutunu belirle (her bir batch'te 64 görüntü olacak)
batch_size = 64

# Eğitim veri kümesi için DataLoader oluştur:
# 'train_set' veri kümesinden batch'ler alır, veriyi karıştırarak (shuffle=True) eğitimi hızlandırır
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

# Doğrulama veri kümesi için DataLoader oluştur:
# 'val_set' veri kümesinden batch'ler alır, veriyi karıştırarak doğrulama işlemini yapar
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

# Test veri kümesi için DataLoader oluştur:
# 'test_set' veri kümesinden batch'ler alır, veriyi karıştırarak test işlemini yapar
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

# Veri kümesinin boyutlarını yazdırma (Training ve Validation veri kümesi için)

# Eğitim ve doğrulama veri kümesinin boyutlarını yazdırmak için bir döngü başlat
for key, value in {'Training data': train_loader, "Validation data": val_loader}.items():
    # DataLoader'dan bir batch örneği al
    for X, y in value:
        print(f"{key}:")  # Veri kümesinin adını yazdır
        
        # Görüntülerin (X) boyutunu yazdır
        print(f"Shape of X : {X.shape}")
        
        # Etiketlerin (y) boyutunu ve veri tipini yazdır
        print(f"Shape of y: {y.shape} {y.dtype}\n")
        
        # Sadece ilk batch için yazdırma işlemi yap
        break


'''Bu fonksiyon, bir konvolüsyonel katmanın çıkış boyutunu hesaplamak için kullanışlı olabilir.
Girdi boyutları ve konvolüsyonel katmanın parametrelerine göre çıkış boyutunu belirler.'''

def findConv2dOutShape(hin, win, conv, pool=2):
    # Konvolüsyonel katmanın kernel boyutu, stride, padding ve dilasyon parametrelerini al
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation

    # Çıkış yüksekliğini (hout) ve genişliğini (wout) hesapla
    hout = np.floor((hin + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    wout = np.floor((win + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    # Eğer havuzlama (pooling) varsa, çıkış boyutlarını havuzlama faktörü ile bölelim
    if pool:
        hout /= pool
        wout /= pool

    # Hesaplanan çıkış boyutlarını tamsayı olarak döndür
    return int(hout), int(wout)


# Retinopati Modeli için Mimari Tanımlaması
class CNN_Retino(nn.Module):
    
    def __init__(self, params):
        # CNN_Retino sınıfını başlat, gerekli parametreleri al
        super(CNN_Retino, self).__init__()

        # Parametreleri al
        Cin, Hin, Win = params["shape_in"]      # Girdi kanal sayısı (Cin), yükseklik (Hin), genişlik (Win)
        init_f = params["initial_filters"]       # Başlangıç filtre sayısı
        num_fc1 = params["num_fc1"]              # İlk tam bağlantılı katman (fully connected) boyutu
        num_classes = params["num_classes"]      # Çıktı sınıf sayısı
        self.dropout_rate = params["dropout_rate"]  # Dropout oranı
        
        # CNN Katmanları
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)   # İlk konvolüsyonel katman
        h, w = findConv2dOutShape(Hin, Win, self.conv1)       # Çıkış boyutlarını hesapla
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)  # İkinci konvolüsyonel katman
        h, w = findConv2dOutShape(h, w, self.conv2)           # Çıkış boyutlarını hesapla
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3) # Üçüncü konvolüsyonel katman
        h, w = findConv2dOutShape(h, w, self.conv3)           # Çıkış boyutlarını hesapla
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3) # Dördüncü konvolüsyonel katman
        h, w = findConv2dOutShape(h, w, self.conv4)           # Çıkış boyutlarını hesapla
        
        # Yassılaştırılmış (flatten) boyutunu hesapla
        self.num_flatten = h * w * 8 * init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)        # İlk tam bağlantılı katman
        self.fc2 = nn.Linear(num_fc1, num_classes)             # Son tam bağlantılı katman

    def forward(self, X):
        # Konvolüsyonel katmanlarda ve max-pooling katmanlarında ileri geçişi tanımla
        X = F.relu(self.conv1(X))                              # İlk konvolüsyonel katman + ReLU aktivasyonu
        X = F.max_pool2d(X, 2, 2)                              # Max-pooling katmanı
        X = F.relu(self.conv2(X))                              # İkinci konvolüsyonel katman + ReLU aktivasyonu
        X = F.max_pool2d(X, 2, 2)                              # Max-pooling katmanı
        X = F.relu(self.conv3(X))                              # Üçüncü konvolüsyonel katman + ReLU aktivasyonu
        X = F.max_pool2d(X, 2, 2)                              # Max-pooling katmanı
        X = F.relu(self.conv4(X))                              # Dördüncü konvolüsyonel katman + ReLU aktivasyonu
        X = F.max_pool2d(X, 2, 2)                              # Max-pooling katmanı
        X = X.view(-1, self.num_flatten)                        # Yassılaştırma (flatten)
        X = F.relu(self.fc1(X))                                # İlk tam bağlantılı katman + ReLU aktivasyonu
        X = F.dropout(X, self.dropout_rate)                     # Dropout uygulama
        X = self.fc2(X)                                        # Son tam bağlantılı katman
        return F.log_softmax(X, dim=1)                          # Softmax ile çıktı ver

    
# Model için parametreleri tanımla
params_model = {
    "shape_in": (3, 255, 255),    # Girdi şekli (3 kanal, 255x255 boyutunda görseller)
    "initial_filters": 8,         # Başlangıçta 8 filtre kullan
    "num_fc1": 100,               # İlk tam bağlantılı katman için 100 nöron
    "dropout_rate": 0.15,         # Dropout oranı 0.15
    "num_classes": 2              # Çıktı sınıf sayısı: 2 (örneğin, Retinopati ve No_Retinopati)
}

# Modelin bir örneğini oluştur
Retino_model = CNN_Retino(params_model)

# Hesaplama donanımını (GPU/CPU) belirle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modeli belirlenen cihaza taşı
Retino_model = Retino_model.to(device)

# CNN_Retino modelinin özetini almak için 'summary' fonksiyonunu kullanma
summary(Retino_model, input_size=(3, 255, 255), device=device.type)


#NLLLoss, modelin tahminleri ile gerçek değerler arasındaki farkı ölçer.
loss_func = nn.NLLLoss(reduction="sum")

#Adam, modelin parametrelerini günceller.
opt = optim.Adam(Retino_model.parameters(), lr=1e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)



# Öğrenme oranını almak için fonksiyon
def get_lr(opt):
    # Optimizer'ın parametre gruplarını dolaş ve her bir gruptaki öğrenme oranını al
    for param_group in opt.param_groups:
        return param_group['lr']

# Her bir batch için kayıp (loss) değerini hesaplayan fonksiyon
def loss_batch(loss_func, output, target, opt=None):
    # Kayıp değerini hesapla
    loss = loss_func(output, target)
    
    # Çıktıdan sınıf tahminini al
    pred = output.argmax(dim=1, keepdim=True)
    
    # Performans metriğini hesapla (doğru tahmin sayısı)
    metric_b = pred.eq(target.view_as(pred)).sum().item()
    
    # Eğer optimizer verilmişse, geri yayılım ve optimizasyon adımlarını uygula
    if opt is not None:
        opt.zero_grad()      # Gradients sıfırla
        loss.backward()      # Geri yayılım
        opt.step()           # Optimizer adımını at
    
    return loss.item(), metric_b  # Kaybı ve metrik değerini döndür

# Bir epoch boyunca tüm veri kümesi için kayıp ve performans metriğini hesaplayan fonksiyon
def loss_epoch(model, loss_func, dataset_dl, opt=None):
    run_loss = 0.0   # Toplam kayıp değeri
    t_metric = 0.0   # Toplam performans metriği
    len_data = len(dataset_dl.dataset)  # Veri kümesinin uzunluğu

    # Veri kümesi üzerinde döngü
    for xb, yb in dataset_dl:
        # Batch'i cihaza taşı
        xb = xb.to(device)
        yb = yb.to(device)
        
        # Modelin çıktısını al
        output = model(xb)
        
        # Kayıp ve metrik hesapla
        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)
        
        # Toplam kayıp değerini güncelle
        run_loss += loss_b
        
        # Performans metriğini güncelle
        if metric_b is not None:
            t_metric += metric_b

    # Ortalama kayıp ve metrik hesapla
    loss = run_loss / float(len_data)
    metric = t_metric / float(len_data)
    
    return loss, metric  # Ortalama kayıp ve metrik değeri döndür





def train_val(model, params, verbose=False):
    # Parametreleri al
    epochs = params["epochs"]  # Eğitim süresi (epoch sayısı)
    loss_func = params["f_loss"]  # Kayıp fonksiyonu
    opt = params["optimiser"]  # Optimizer
    train_dl = params["train"]  # Eğitim veri yükleyici
    val_dl = params["val"]  # Doğrulama veri yükleyici
    lr_scheduler = params["lr_change"]  # Öğrenme oranı değişim planlayıcı
    weight_path = params["weight_path"]  # Ağırlıkların kaydedileceği dosya yolu
    
    # Eğitim ve doğrulama için kayıp değerlerinin geçmişi
    loss_history = {"train": [], "val": []}
    # Eğitim ve doğrulama için metrik değerlerinin geçmişi
    metric_history = {"train": [], "val": []}
    
    # En iyi performans gösteren modelin ağırlıklarını derin bir kopya olarak sakla
    best_model_wts = copy.deepcopy(model.state_dict())
    # Başlangıçta en iyi kaybı çok büyük bir değere ata
    best_loss = float('inf')

    # Modelin eğitim süreci (her epoch için eğitim ve doğrulama)
    for epoch in tqdm(range(epochs)):
        
        # Öğrenme oranını al
        current_lr = get_lr(opt)
        if verbose:
            print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))

        # Eğitim adımı
        model.train()  # Modeli eğitim moduna al
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)

        # Eğitim kaybı ve metriğini kaydet
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)

        # Modeli değerlendirme adımı
        model.eval()  # Modeli doğrulama moduna al
        with torch.no_grad():  # Değerlendirme sırasında geri yayılım yapma
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)

        # En iyi modeli sakla (daha düşük kayıp)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            # Modelin ağırlıklarını kaydet
            torch.save(model.state_dict(), weight_path)
            if verbose:
                print("Copied best model weights!")

        # Doğrulama kaybı ve metriğini kaydet
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        # Öğrenme oranı programı
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):  # Öğrenme oranı değişti mi?
            if verbose:
                print("Loading best model weights!")
            model.load_state_dict(best_model_wts)  # En iyi modelin ağırlıklarını yükle

        # Eğer verbose True ise, eğitim ve doğrulama kaybı ve doğruluğunu yazdır
        if verbose:
            print(f"train loss: {train_loss:.6f}, dev loss: {val_loss:.6f}, accuracy: {100*val_metric:.2f}")
            print("-"*10) 

    # Eğitim tamamlandığında, en iyi modelin ağırlıklarını yükle
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history



# Eğitim ve model değerlendirme için kullanılan parametrelerin tanımlanması

params_train = {
    "train": train_loader,  # Eğitim veri yükleyici
    "val": val_loader,  # Doğrulama veri yükleyici
    "epochs": 30,  # Eğitim süresi (epoch sayısı)
    "optimiser": optim.Adam(Retino_model.parameters(), lr=1e-4),  # Adam optimizasyonu ve öğrenme oranı
    "lr_change": ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=1),  # Öğrenme oranı planlayıcı (plateau tabanlı)
    "f_loss": nn.NLLLoss(reduction="sum"),  # Negatif log likelihood loss fonksiyonu
    "weight_path": "weights.pt",  # En iyi modelin ağırlıklarının kaydedileceği dosya yolu
}

# Modeli eğitme ve doğrulama
model, loss_hist_m, metric_hist_m = train_val(Retino_model, params_train)


# Epoch sayısını parametrelerden alıyoruz
epochs=params_train["epochs"]

# Grafik için figür ve eksenler oluşturuluyor
fig,ax = plt.subplots(1,2,figsize=(12,5))

# Eğitim kaybı (train loss) grafiği çiziliyor
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist_m["train"],ax=ax[0],label='loss_hist["train"]')
# Doğrulama kaybı (validation loss) grafiği çiziliyor
sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist_m["val"],ax=ax[0],label='loss_hist["val"]')

# Eğitim doğruluğu (train accuracy) grafiği çiziliyor
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist_m["train"],ax=ax[1],label='Acc_hist["train"]')
# Doğrulama doğruluğu (validation accuracy) grafiği çiziliyor
sns.lineplot(x=[*range(1,epochs+1)],y=metric_hist_m["val"],ax=ax[1],label='Acc_hist["val"]')





# Eğitim, doğrulama ve test veri setlerinde modelin tahmin sonuçları classification_report ile özetleniyor.
# Sınıflandırma raporunu almak için fonksiyon tanımlanıyor

def ture_and_pred_val(val_loader, model):
    i = 0
    y_true = []  # Gerçek etiketlerin saklanacağı liste
    y_pred = []  # Tahmin edilen etiketlerin saklanacağı liste
    
    # Verileri iterasyon ile dolaşıyoruz
    for images, labels in val_loader:
        images = images.to(device)  # Veriyi cihazımıza (GPU/CPU) taşıyoruz
        labels = labels.numpy()  # Etiketleri numpy dizisine çeviriyoruz (PyTorch tensor'ları yerine)
        
        outputs = model(images)  # Modelin tahmin çıktısı
        _, pred = torch.max(outputs.data, 1)  # Modelin en yüksek tahmin değerine sahip sınıfını seçiyoruz
        pred = pred.detach().cpu().numpy()  # Tahminleri numpy dizisine çeviriyoruz ve CPU'ya taşıyoruz
        
        # Gerçek etiketleri ve tahmin edilen etiketleri birleştiriyoruz
        y_true = np.append(y_true, labels)
        y_pred = np.append(y_pred, pred)
    
    # Sonuç olarak gerçek etiketler ve tahmin edilen etiketleri döndürüyoruz
    return y_true, y_pred



# Eğitim veri seti üzerinden Retinopathy sınıflandırma modelinin sınıflandırma raporunu alıyoruz

# Eğitim veri setindeki gerçek etiketler ve modelin tahminlerini elde ediyoruz
y_true, y_pred = ture_and_pred_val(train_loader, Retino_model)

# Gerçek etiketler ve tahmin edilen etiketler arasındaki sınıflandırma raporunu yazdırıyoruz
print(classification_report(y_true, y_pred), '\n\n')



# Doğrulama veri seti üzerinden Retinopathy sınıflandırma modelinin sınıflandırma raporunu alıyoruz

# Doğrulama veri setindeki gerçek etiketler ve modelin tahminlerini elde ediyoruz
y_true, y_pred = ture_and_pred_val(val_loader, Retino_model)

# Gerçek etiketler ve tahmin edilen etiketler arasındaki sınıflandırma raporunu yazdırıyoruz
print(classification_report(y_true, y_pred), '\n\n')



#model sonra da kullanılabilmek için kaydediliyor.
torch.save(Retino_model, "Retino_model.pt")     


# Önceden eğitilmiş modeli yüklüyoruz
model = torch.load("Retino_model.pt")

# Modeli GPU cihazına taşıyoruz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Test veri seti üzerinde tahmin yapmak için test loader'ı üzerinden geçiyoruz
with torch.no_grad():  # Gradients hesaplanmayacak, sadece tahmin yapılacak
    for images, _ in test_loader:  # Test veri setindeki her bir batch için

        images = images.to(device)  # Görselleri GPU'ya taşıyoruz
        output = model(images)  # Modeli test verisi ile çalıştırıyoruz
        probabilities = torch.softmax(output, dim=1)  # Çıktıyı sınıflara ait olasılıklara dönüştürüyoruz
        predicted_classes = torch.argmax(probabilities, dim=1)  # En yüksek olasılıkla tahmin edilen sınıfı buluyoruz

        # Tahmin edilen sınıfı yazdırıyoruz
        for predicted_class in predicted_classes:
            print("Predicted class:", predicted_class.item())

        ## Classification Report for Retinopathy Classification Model based on Test set
y_true, y_pred = ture_and_pred_val(test_loader, model)
print(classification_report(y_true, y_pred), '\n\n')
