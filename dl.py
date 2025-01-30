import torch
import torchvision.models as models
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1️⃣ **ResNet-50'yi Yükle ve Özellik Çıkarıcı Olarak Kullan**
resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
resnet_model.eval()
feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])  # Son FC katmanı çıkar

# 2️⃣ **Görüntü Ön İşleme**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3️⃣ **Özellik Çıkarma Fonksiyonu**
def extract_features(image_path):
    """Görüntüyü ResNet-50 ile işleyip özellik vektörüne dönüştürür."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # 4D tensör (1, 3, 224, 224)
        with torch.no_grad():
            features = feature_extractor(input_tensor)
            return features.view(-1).cpu().numpy()  # 2048 boyutlu vektör
    except Exception as e:
        print(f"Hata: {image_path} işlenemedi! ({e})")
        return None

# 4️⃣ **CSV'den Özellikleri Oku**
base_dir = r"C:/Users/karat/Desktop/retino"
train_csv_path = os.path.join(base_dir, "train_features.csv")

train_df = pd.read_csv(train_csv_path, header=None)
num_features = train_df.shape[1] - 2
train_df.columns = ["filename", "label"] + [f"feature_{i}" for i in range(num_features)]

# 5️⃣ **Veriyi Hazırla**
X = train_df.iloc[:, 2:].values  # 2048 boyutlu özellikler
y = train_df["label"].values     # Etiketler (0 veya 1)

# PyTorch tensörlerine çevir
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# PyTorch Dataset ve DataLoader
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 6️⃣ **Derin Öğrenme Modeli Tanımla**
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 sınıf (DR & No_DR)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # Son katmanda aktivasyon yok çünkü CrossEntropyLoss softmax içeriyor
        return x

# Modeli oluştur
model = DeepNN()

# Kayıp fonksiyonu ve optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7️⃣ **Modeli Eğit**
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 8️⃣ **Modeli Test Et**
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.numpy())
        all_labels.extend(batch_y.numpy())

# 9️⃣ **Sonuçları Değerlendir**
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n🎯 Derin Öğrenme Modelinin Doğruluğu: {accuracy:.4f}")

print("\n📊 Sınıflandırma Raporu:")
print(classification_report(all_labels, all_preds, target_names=["No_DR", "DR"]))

# Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No_DR", "DR"], yticklabels=["No_DR", "DR"])
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.title("Confusion Matrix - Derin Öğrenme Modeli")
plt.show()
