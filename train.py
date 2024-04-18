from imports import *

def train(train_loader, val_loader):
    num_epochs = num_epochs
    base_lr =  base_lr
    model = smp.Unet(encoder_name='resnet34', in_channels=in_channels, classes=classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_iou = 0

    for i in trange(num_epochs):
        train_loss = []
        train_iou = []
        for data,labels in tqdm(train_loader):
            labels[labels == 255] = 1
            model.train()
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(data)
            labels = labels.float().unsqueeze(1)
            
            pred = pred.to(device)
            loss = criterion(pred,labels)
            loss.backward()
            optimizer.step()
            
            predicted_masks = (pred > 0.5).float()
            intersection = torch.logical_and(predicted_masks, labels).sum().item()
            union = torch.logical_or(predicted_masks, labels).sum().item()
            iou = intersection / union if union != 0 else 0
            
            train_loss.append(loss.item())
            train_iou.append(iou)
        print(f"Mean train loss at this stage is {(np.array(train_loss).mean())}, IoU = {(np.array(train_iou).mean())}")
        with torch.no_grad():
            val_loss = []
            val_iou = []
            for data,labels in tqdm(val_loader):
                labels[labels == 255] = 1
                model.eval()

                data, labels = data.to(device), labels.to(device)

                pred = model(data)
                labels = labels.float().unsqueeze(1)
                loss = criterion(pred, labels)
        
                predicted_masks = (pred > 0.5).float()
                intersection = torch.logical_and(predicted_masks, labels).sum().item()
                union = torch.logical_or(predicted_masks, labels).sum().item()
                iou = intersection / union if union != 0 else 0
                
                val_loss.append(loss.item())
                val_iou.append(iou)
            
        print(f"Mean validation loss at this stage is {(np.array(val_loss).mean())}, IoU = {(np.array(val_iou).mean())}")
        if np.array(val_iou).mean() > best_iou:
            best_iou = np.array(val_iou).mean()
            torch.save(model.state_dict(), "model.pt")
        scheduler.step()