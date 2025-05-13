import os
import nibabel as nib
import numpy as np
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.utils.losses import GeneralizedDiceLoss,  BCEDiceLoss
from torch.utils.data import Dataset, DataLoader
from src.utils.utils import custom_collate_BHSD
import torch.nn.functional as F
import torch
from src.configuration.config import (
    datadict, TrainingDir, batch_size, num_epochs, num_workers,
    pin_memory, LEARNING_RATE, IMAGE_HEIGHT, IMAGE_WIDTH,newDatadict
)
from src.Models.D_UNet import UNet3D, ResidualUNetSE3D
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class BHSD_3D(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, datadict=newDatadict):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.series = os.listdir(mask_dir)
        self.datadict = datadict
        reversed_dict = {v: k for k, v in datadict.items()}
        self.reversed_dict = reversed_dict

    def transform_volume(self, image_volume, mask_volume):
        transformed = self.transform(
                image=image_volume, 
                mask=mask_volume
            )
        images = transformed['image']
        masks = transformed['mask'].permute(2, 0, 1)
        masks = F.one_hot(masks.long(), num_classes=6)
        masks = masks.permute(3, 0, 1, 2)
        return images , masks.float()

    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        nii_segementation = nib.load(os.path.join(self.mask_dir, self.images[index]))
        nii_image = nib.load(os.path.join(self.image_dir, self.images[index]))
        
        # Get the image data as a NumPy array
        image_data = nii_image.get_fdata()
        segementation_data = nii_segementation.get_fdata()

        if self.transform is not None:
            transformed_image_volume, transformed_mask_volume = self.transform_volume(image_data, segementation_data)

        transformed_image_volume = transformed_image_volume.unsqueeze(0)
        # transformed_mask_volume = transformed_mask_volume.unsqueeze(0)
        return transformed_image_volume, transformed_mask_volume
    


# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, inputs, targets):
#         # Flatten the tensors
#         inputs = inputs.contiguous().view(-1)
#         targets = targets.contiguous().view(-1)
        
#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
#         return 1 - dice
    


def new_custom_collate_BHSD(batch):
    max_depth = 0
    for x,y in batch:
        max_depth = max(max_depth, x.shape[1])


    newImageVolume = []
    newMaskVolume = []
    for i in range(len(batch)):
        remmaining_slice = max_depth - batch[i][0].shape[1]
        # print(remmaining_slice)
        if remmaining_slice > 0:
            if remmaining_slice > batch[i][0].shape[1]:
                return None, None
            # empty_slice = torch.zeros((1,remmaining_slice,batch[i][0].shape[2], batch[i][0].shape[3]))
            # empty_slice_mask = torch.zeros((6,remmaining_slice,batch[i][0].shape[2], batch[i][0].shape[3]))

            empty_slice = batch[i][0][:,:remmaining_slice, :,:]
            empty_slice_mask = batch[i][1][:,:remmaining_slice, :,:]
            # print(empty_slice.shape,' ',empty_slice_mask.shape)
            
            newImageVolume.append(torch.cat((batch[i][0], empty_slice), dim=1))
            newMaskVolume.append(torch.cat((batch[i][1], empty_slice_mask), dim=1))
        else:
            newImageVolume.append(batch[i][0])
            newMaskVolume.append(batch[i][1])
    

    newImageVolume = torch.stack(newImageVolume, dim=0)
    newMaskVolume = torch.stack(newMaskVolume, dim=0)

    return newImageVolume, newMaskVolume
    
def new_custom_collate_BHSD_v2(batch):
    # Determine the max depth in the batch
    max_depth = max(x.shape[1] for x, _ in batch)

    newImageVolume = []
    newMaskVolume = []

    for x, y in batch:
        current_depth = x.shape[1]
        rem_slices = max_depth - current_depth

        # Repeat entire volume until we are close to or over the max depth
        while x.shape[1] < max_depth:
            x = torch.cat((x, x), dim=1)
            y = torch.cat((y, y), dim=1)

        # Trim if we exceeded max_depth
        x = x[:, :max_depth, :, :]
        y = y[:, :max_depth, :, :]

        newImageVolume.append(x)
        newMaskVolume.append(y)

    newImageVolume = torch.stack(newImageVolume, dim=0)
    newMaskVolume = torch.stack(newMaskVolume, dim=0)

    return newImageVolume, newMaskVolume


def main():

    IMAGE_HEIGHT = 128
    IMAGE_WIDTH = 128
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    masks_path = os.path.join(TrainingDir, 'ground truths')
    images_path = os.path.join(TrainingDir, 'images')

    data = BHSD_3D(images_path, masks_path, train_transform)


    train_loader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=new_custom_collate_BHSD_v2,
        )





    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResidualUNetSE3D(in_channels=1, out_channels=6).to(device)
    checkpoint = torch.load('model_checkpoint.pth')  # or 'model_checkpoint.pth.tar' 
    model.load_state_dict(checkpoint)

    # criterion = nn.BCEWithLogitsLoss()
    criterion = BCEDiceLoss()
    weights = torch.tensor([0.2, 0.5, 1.0, 1.0, 0.7, 0.5]).to(device)
    criterion.dice.weight = weights  # Set weight buffer manually


    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backpropagation with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item()}")

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")

        scheduler.step(avg_loss)  # 🔁 Update LR based on loss
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Save model checkpoint
        checkpoint_path = f"model_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()