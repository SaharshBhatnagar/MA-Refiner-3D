import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import binvox_rw

from model import Pix2VoxWithMAR

class ShapeNetMultiViewDataset(Dataset):
    def __init__(self, rendering_dir, voxel_dir, num_views=5, transform=None):
        self.rendering_dir = rendering_dir
        self.voxel_dir = voxel_dir
        self.num_views = num_views
        self.transform = transform
        self.samples = self._load_metadata()

    def _load_metadata(self):
        samples = []
        categories = os.listdir(self.rendering_dir)
        for cat in categories:
            cat_render_path = os.path.join(self.rendering_dir, cat)
            cat_voxel_path = os.path.join(self.voxel_dir, cat)
            
            if not os.path.isdir(cat_render_path) or not os.path.isdir(cat_voxel_path):
                continue
                
            models = os.listdir(cat_render_path)
            for model_id in models:
                model_render_dir = os.path.join(cat_render_path, model_id, 'rendering')
                voxel_path = os.path.join(cat_voxel_path, model_id, 'model.binvox')
                
                if os.path.exists(model_render_dir) and os.path.exists(voxel_path):
                    samples.append({
                        'render_dir': model_render_dir,
                        'voxel_path': voxel_path
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image_paths = sorted(glob.glob(os.path.join(sample['render_dir'], '*.png')))
        selected_paths = np.random.choice(image_paths, self.num_views, replace=False)
        
        images = []
        for path in selected_paths:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        images_tensor = torch.stack(images) 
        
        with open(sample['voxel_path'], 'rb') as f:
            voxel_model = binvox_rw.read_as_3d_array(f)
            voxel_data = voxel_model.data.astype(np.float32)
            
        voxel_tensor = torch.tensor(voxel_data).unsqueeze(0) 
        
        return images_tensor, voxel_tensor

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hardware specific adjustments
    batch_size = 16 
    epochs = 50
    lr = 0.001
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ShapeNetMultiViewDataset(
        rendering_dir="data/ShapeNetRendering",
        voxel_dir="data/ShapeNetVox32",
        num_views=8, 
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    model = Pix2VoxWithMAR().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)
    criterion = nn.BCELoss()

    os.makedirs("weights", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_acc = 0.0
        running_fscore = 0.0
        
        for i, (images, voxels) in enumerate(dataloader):
            images = images.to(device)
            voxels = voxels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, voxels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            with torch.no_grad():
                predicted_voxels = (outputs > 0.3).float()
                
                correct = (predicted_voxels == voxels).float().sum()
                total = voxels.numel()
                acc = correct / total
                running_acc += acc.item()
                
                intersection = (predicted_voxels * voxels).sum()
                pred_sum = predicted_voxels.sum()
                target_sum = voxels.sum()
                
                union = pred_sum + target_sum - intersection
                iou = (intersection + 1e-6) / (union + 1e-6)
                running_iou += iou.item()
                
                f_score = (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)
                running_fscore += f_score.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}, Acc: {acc.item():.4f}, IoU: {iou.item():.4f}, F-Score: {f_score.item():.4f}")
        
        scheduler.step()
        avg_loss = running_loss / len(dataloader)
        avg_acc = running_acc / len(dataloader)
        avg_iou = running_iou / len(dataloader)
        avg_fscore = running_fscore / len(dataloader)
        
        print(f"--- Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}, Avg IoU: {avg_iou:.4f}, Avg F-Score: {avg_fscore:.4f} ---")
        
        torch.save(model.state_dict(), "weights/model_latest.pth")

if __name__ == "__main__":
    train()