
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


class MiniVGGNet(nn.Module):
    def __init__(self, inputShape, classes):
        super(MiniVGGNet, self).__init__()

        # Channel dimension
        chanDim = 1 if inputShape[0] == 1 else 3

        # First CONV => RELU => CONV => RELU => POOL layer set
        self.conv1a = nn.Conv2d(inputShape[0], 32, (3, 3), padding="same")
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, (3, 3), padding="same")
        self.bn1b = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)

        # Second CONV => RELU => CONV => RELU => POOL layer set
        self.conv2a = nn.Conv2d(32, 64, (3, 3), padding="same")
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, (3, 3), padding="same")
        self.bn2b = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)

        # First (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(64 * (inputShape[1] // 4) * (inputShape[2] // 4), 512)
        self.bn_fc1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.5)

        # Softmax classifier
        self.fc2 = nn.Linear(512, classes)

    def forward(self, x):
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)  # Flatten the layer
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def calculate_loss_and_accuracy(model, data_loader, criterion):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    loss /= len(data_loader)
    accuracy = 100. * correct / total
    return loss, accuracy


# 数据加载和变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Instantiate the model
input_shape = (1, 28, 28)  # The image dimensions for the Fashion MNIST dataset
num_classes = 10  # The number of classes in the Fashion MNIST dataset
model = MiniVGGNet(input_shape, num_classes).to(device)  # Move the model to the appropriate device (CPU or GPU)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss is commonly used for classification tasks
optimizer = Adam(model.parameters(), lr=0.001)  # Adam optimizer with a learning rate of 0.001


writer = SummaryWriter('Ass2/runs/fashion_mnist_experiment')

# Initialize lists to store metrics
train_accuracies = []  # List to store training accuracy per epoch
test_accuracies = []   # List to store test accuracy per epoch
train_losses = []      # List to store training loss per epoch
test_losses = []       # List to store test loss per epoch

# 训练循环
epochs = range(0,20)
for epoch in epochs:
    model.train()
    train_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch:
        for data, target in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}")

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    # 在训练集上计算损失和准确率
    train_loss, train_accuracy = calculate_loss_and_accuracy(model, train_loader, criterion)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(f'End of Epoch {epoch + 1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
    # 记录到 TensorBoard
    writer.add_scalar('Training Loss', train_loss, epoch)
    writer.add_scalar('Training Accuracy', train_accuracy, epoch)
    # 在测试集上计算损失和准确率
    test_loss, test_accuracy = calculate_loss_and_accuracy(model, test_loader, criterion)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    print(f'End of Epoch {epoch + 1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    # 记录到 TensorBoard
    writer.add_scalar('Test Loss', test_loss, epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, epoch)


# Create a dummy input tensor that matches the model's input dimensions
dummy_input = torch.zeros(1, *input_shape).to(device)

# Add the model graph to TensorBoard
writer.add_graph(model, dummy_input)

# Close the SummaryWriter
writer.close()

# Save the model to a file
# torch.save(model, 'Ass2/model/CNN2_model.pth')


# 最终测试集评估
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    with tqdm(test_loader, unit="batch") as ttest:
        for data, target in ttest:
            ttest.set_description("Test")
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            ttest.set_postfix(loss=test_loss / len(test_loader), accuracy=100. * correct / len(test_loader.dataset))

test_loss /= len(test_loader.dataset)
print(
    f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)')


# for name, param in model.named_parameters():
#     print(f"{name}: {param}")


# 绘制准确率对比图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies, 'bo-', label='Train Accuracy')
plt.plot(epochs, test_accuracies, 'ro-', label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.show()

# 绘制损失对比图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 1)
plt.legend()
plt.show()