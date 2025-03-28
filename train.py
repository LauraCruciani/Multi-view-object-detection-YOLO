import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import glob
from ultralytics import YOLO

class YOLOv8MultiView(nn.Module):
    def __init__(self, model_path='yolov8n.pt'):
        super(YOLOv8MultiView, self).__init__()
        self.yolo = YOLO(model_path).model
        self.fusion_layer = nn.Conv2d(512, 512, kernel_size=1)  # Fusione feature
    
    def forward(self, images):
        features = [self.yolo.model[0](img) for img in images]  # Estrazione feature per ogni vista
        fused_feat = torch.mean(torch.stack(features), dim=0)  # Media delle feature maps
        fused_feat = self.fusion_layer(fused_feat)  # Fusione
        return self.yolo.head(fused_feat)  # Testa YOLO per la detection

class MultiViewDataset(Dataset):
    def __init__(self, img_root, label_root, transform=None):
        self.obj_folders = sorted(glob.glob(os.path.join(img_root, '*')))  # Cartelle degli oggetti
        self.label_root = label_root
        self.transform = transform
    
    def __len__(self):
        return len(self.obj_folders)
    
    def __getitem__(self, idx):
        obj_folder = self.obj_folders[idx]
        img_paths = sorted(glob.glob(os.path.join(obj_folder, '*.jpg')))  # Tutte le viste
        label_path = os.path.join(self.label_root, os.path.basename(obj_folder) + ".txt")  # Unica label

        images = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        images = torch.stack(images)  # Stack delle immagini
        labels = self.load_label(label_path)
        
        return images, torch.tensor(labels)

    def load_label(self, label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            cls, x, y, w, h = map(float, line.strip().split())
            labels.append([cls, x, y, w, h])
        return labels

class CustomYoloLoss(nn.Module):
    def __init__(self, base_loss):
        super(CustomYoloLoss, self).__init__()
        self.base_loss = base_loss
    
    def forward(self, preds, targets):
        loss = self.base_loss(preds, targets)
        confidence_scores = preds[..., 4]
        penalty = torch.sum(confidence_scores > 0.5) - 1  # Penalizza più predizioni
        penalty = torch.clamp(penalty, min=0)
        return loss + 0.1 * penalty
# Configurazione del dataset e DataLoader
train_dataset = MultiViewDataset('dataset/images/train', 'dataset/labels/train')
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Inizializzazione del modello, ottimizzatore e funzione di loss
model = YOLOv8MultiView()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = CustomYoloLoss(model.yolo.loss)

# Impostazione del dispositivo (GPU se disponibile)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Ciclo di allenamento
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]  # Batch di viste variabili
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')

