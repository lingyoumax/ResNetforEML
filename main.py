import warnings
import wandb
import random
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from Model.Schema import resModel
from torch import nn, optim
from tqdm import tqdm

warnings.filterwarnings('ignore')

seed = 1024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device='cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 64

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root="dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder(root="dataset/valid", transform=test_transform)
test_dataset = datasets.ImageFolder(root="dataset/test", transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class_names = train_dataset.classes
class_count = [train_dataset.targets.count(i) for i in range(len(class_names))]
num_classes = len(class_names)

model=resModel(num_classes).to(device)

num_epoch = 20
lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

wandb.init(
    project="EML",
    config={
    "learning_rate": lr,
    "architecture": "PretrainedResNet18",
    "dataset": "70 Dog Breeds",
    "epochs": num_epoch,
    }
)

def train():
    model.train()
    total_loss=0.0
    correct = 0
    total=0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = 100 * correct / total
    return val_loss, val_accuracy

def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_accuracy = 100 * correct / total
    return test_accuracy

for epoch in tqdm(range(num_epoch)):
    train_loss, train_accuracy=train()
    val_loss, val_accuracy=validate()
    test_accuracy =test()
    wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy, "test_accuracy": test_accuracy})
    model_path = f"Weights/model_{epoch}.pth"
    torch.save(model.state_dict(), model_path)

wandb.finish()    