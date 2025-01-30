import torch
import torchvision.models as models
from torch import nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. ResNet-50 modelini yükle ve özellik çıkarıcı oluştur
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.eval()
feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])

# 2. Görüntüleri işlemek için dönüşümler
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Ana dizin ve alt klasörleri tanımla
base_dir = r"C:\Users\karat\Desktop\retino"
sub_dirs = ["train", "valid", "test"]
classes = {"DR": 1, "No_DR": 0}  # DR: Hastalık var (1), No_DR: Yok (0)

# 4. Özellik çıkarma fonksiyonu
def extract_features(image_path):
    """Görüntüden özellikleri çıkarır."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = feature_extractor(input_tensor)
            return features.view(-1).cpu().numpy()
    except Exception as e:
        print(f"Hata: {image_path} işlenemedi! ({e})")
        return None

# 5. Verileri işle ve CSV olarak kaydet
for dataset in sub_dirs:
    dataset_path = os.path.join(base_dir, dataset)
    if not os.path.exists(dataset_path):
        print(f"UYARI: {dataset_path} bulunamadı, atlanıyor!")
        continue

    data = []
    for label, class_id in classes.items():
        label_path = os.path.join(dataset_path, label)
        if not os.path.exists(label_path):
            print(f"UYARI: {label_path} klasörü eksik, atlanıyor!")
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue  # Yalnızca görüntü dosyalarını işle
            features = extract_features(img_path)
            if features is not None:
                data.append([img_name, class_id] + list(features))

    # 6. CSV olarak kaydet
    df = pd.DataFrame(data)
    csv_filename = os.path.join(base_dir, f"{dataset}_features.csv")
    df.to_csv(csv_filename, index=False, header=False)
    print(f"{dataset} verisi işlendi ve {csv_filename} dosyasına kaydedildi!")

# 7. CSV dosyasını oku
train_csv_path = os.path.join(base_dir, "train_features.csv")
train_df = pd.read_csv(train_csv_path, header=None)

# 8. Sütun isimlerini ayarla
num_features = train_df.shape[1] - 2  # İlk iki sütun dosya adı ve etiket
train_df.columns = ["filename", "label"] + [f"feature_{i}" for i in range(num_features)]

# 9. Sınıf dağılımını göster
print("\n📌 Sınıf Dağılımı:")
print(train_df["label"].value_counts())

# 10. İlk birkaç özelliğin istatistiksel dağılımı
print("\n📊 İlk 10 özellik için özet istatistikler:")
print(train_df.iloc[:, 2:12].describe())

# 11. Özellikleri görselleştir (Feature 0 vs Feature 1)
plt.figure(figsize=(8, 6))
plt.scatter(train_df["feature_0"], train_df["feature_1"], 
            c=train_df["label"], cmap="coolwarm", alpha=0.5)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Feature 0 vs Feature 1")
plt.colorbar(label="DR (1) / No_DR (0)")
plt.show()

# 12. PCA ile 2D'ye indir ve görselleştir
features = train_df.iloc[:, 2:].values
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)
df_pca = pd.DataFrame(features_pca, columns=["PCA1", "PCA2"])
df_pca["label"] = train_df["label"]

plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PCA1"], df_pca["PCA2"], c=df_pca["label"], cmap="coolwarm", alpha=0.5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA ile Özellikleri 2D'ye İndirme")
plt.colorbar(label="DR (1) / No_DR (0)")
plt.show()

# 13. Makine Öğrenmesi Modeli Eğitme
X = train_df.iloc[:, 2:].values
y = train_df["label"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 14. Model Performans Değerlendirme
y_pred = model.predict(X_test)

# Doğruluk (Accuracy) hesapla
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Doğruluğu: {accuracy:.4f}")

# Precision, Recall ve F1-score hesapla
print("\n📊 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["No_DR", "DR"]))

# Confusion Matrix hesapla ve görselleştir
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n📌 Confusion Matrix:")
print(conf_matrix)

# Görselleştirme
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No_DR", "DR"], yticklabels=["No_DR", "DR"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix")
plt.show()
