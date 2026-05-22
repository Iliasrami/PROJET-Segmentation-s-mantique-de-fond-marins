import torch
import wandb
# Define a function to calculate accuracy
def accuracy_(outputs, labels):
    _, predicted = torch.max(outputs, 1)  # Get class predictions

    total = labels.nelement()
    correct = predicted.eq(labels).sum().item()
    return correct / total

# Define a function to train the model
def train(model,model_name, backbone_name, train_loader, val_loader, criterion, optimizer, num_epochs, batch_size, learning_rate, p, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val=[]
    if model_name == "UNet":
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_accuracy = 0.0

            for images, labels in train_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.int64) #.reshape(batch_size, 8,256, 256)
                # labels = labels.max(1)[1]  # Convert from one-hot back to class indices if necessary
                optimizer.zero_grad()
                
                outputs = model(images) #['out']

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                accuracy = accuracy_(outputs, labels)
                running_loss += loss.item()
                running_accuracy += accuracy
            if epoch % 2 == 0:
                # torch.save(model.state_dict(), f"model_{epoch}.pth")
                val_loss = 0.0
                val_accuracy = 0.0
                model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.int64) #.reshape(batch_size, 8,256, 256)
                        outputs = model(images) #['out']
                        # labels = torch.argmax(labels, dim=1)
                        loss = criterion(outputs, labels)
                        accuracy = accuracy_(outputs, labels)
                        val_loss += loss.item()
                        val_accuracy += accuracy
                wandb.log({"Validation Loss": val_loss/len(val_loader)})
                wandb.log({"Validation Accuracy": val_accuracy/len(val_loader)})
                val.append(val_accuracy/len(val_loader))
            # Print average loss for the epoch
            if val[-1] == max(val):
                torch.save(model.state_dict(),output_dir + f"model_best_unet_{backbone_name}_{batch_size}_{num_epochs}_{learning_rate}_{p}.pth")
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
            print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {running_accuracy/len(train_loader)}")
            wandb.log({"Training Loss": running_loss/len(train_loader)})
            wandb.log({"Training Accuracy": running_accuracy/len(train_loader)})
            torch.save(model.state_dict(), output_dir + f"checkpoint_unet_{backbone_name}_{batch_size}_{num_epochs}_{learning_rate}_{p}.pth")
    elif model_name == "DeepLabV3":
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            running_loss = 0.0
            running_accuracy = 0.0

            for images, labels in train_loader:
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.int64) #.reshape(batch_size, 8,256, 256)
                optimizer.zero_grad()
                
                outputs = model(images) ['out']

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                accuracy = accuracy_(outputs, labels)
                running_loss += loss.item()
                running_accuracy += accuracy
            if epoch % 2 == 0:
                # torch.save(model.state_dict(), f"model_{epoch}.pth")
                val_loss = 0.0
                val_accuracy = 0.0
                model.eval()
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.int64) #.reshape(batch_size, 8,256, 256)
                        outputs = model(images) ['out']
                        loss = criterion(outputs, labels)
                        accuracy = accuracy_(outputs, labels)
                        val_loss += loss.item()
                        val_accuracy += accuracy
                wandb.log({"Validation Loss": val_loss/len(val_loader)})
                wandb.log({"Validation Accuracy": val_accuracy/len(val_loader)})
                val.append(val_accuracy/len(val_loader))
            # Print average loss for the epoch
            if val[-1] == max(val):
                # torch.save(model.state_dict(), f"model_best_unet_densenet_{batch_size}_{num_epochs}_{learning_rate}_{p}.pth")
                torch.save(model.state_dict(), output_dir + f"model_best_deeplab_{batch_size}_{num_epochs}_{learning_rate}_{p}.pth")
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
            print(f"Epoch {epoch+1}/{num_epochs}, Accuracy: {running_accuracy/len(train_loader)}")
            wandb.log({"Training Loss": running_loss/len(train_loader)})
            wandb.log({"Training Accuracy": running_accuracy/len(train_loader)})
            torch.save(model.state_dict(), output_dir + f"checkpoint_deeplab_{batch_size}_{num_epochs}_{learning_rate}_{p}.pth")

def compute_mIoU(predicted, labels, num_classes):
    iou_list = []
    present_iou_list = []

    predicted = predicted.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = predicted == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()  # Intersection
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection  # Union

        if union > 0:
            iou = float(intersection) / float(max(union, 1))
            present_iou_list.append(iou)
            iou_list.append(iou)
        else:
            iou_list.append(float('nan'))

    mean_iou = float(sum(present_iou_list)) / float(len(present_iou_list))
    return mean_iou, iou_list


def test(model, model_name, test_loader, suffix=""):
    model.eval()  # Set model to evaluation mode
    total = 0
    correct = 0
    miou_total = 0
    num_batches = 0
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda').long()
            if model_name == "UNet":
                outputs = model(inputs)
            elif model_name == "DeepLabV3":
                outputs = model(inputs)['out']
            _, predicted = torch.max(outputs, 1)  # Get class predictions

            total += labels.nelement()
            correct += predicted.eq(labels).sum().item()
            miou, _ = compute_mIoU(predicted, labels, num_classes=8)
            miou_total += miou
            num_batches += 1

    pixel_accuracy = 100 * correct / total
    print(f'Pixel Accuracy: {pixel_accuracy:.2f}%')
    mean_iou = miou_total / num_batches*100
    print(f'Mean IoU: {mean_iou:.2f}%')
    wandb.log({"Test Pixel Accuracy using "+suffix : pixel_accuracy})
    wandb.log({"Test Mean IoU using "+suffix: mean_iou})

