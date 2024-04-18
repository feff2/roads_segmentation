from imports import *


def test(test_loader):
    model = None
    criterion = nn.BCEWithLogitsLoss()
    model.load_state_dict(torch.load('model/model.pt'))

    model.eval()
    test_loss = 0.0
    test_total_iou = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:

            targets[targets == 255] = 1

            inputs = inputs.to(device)
            targets = targets.to(device)
            targets = targets.float().unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
                
            predicted_masks = (outputs > 0.5).float()
            intersection = torch.logical_and(predicted_masks, targets).sum().item()
            union = torch.logical_or(predicted_masks, targets).sum().item()
            iou = intersection / union if union != 0 else 0
            test_total_iou += iou

    print(f'Test Loss: {test_loss / len(test_loader)}')
    print(f'Test IOU: {test_total_iou / len(test_loader)}')