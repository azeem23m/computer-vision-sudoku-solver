import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import CNN
from torchvision import datasets, transforms

class Trainer:
  def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, epochs: int, criterion: nn.Module, device: torch.device):
    
    self.model = model
    self.optimizer = optimizer
    self.epochs = epochs
    self.criterion = criterion
    self.device = device
    self.model.to(self.device)


  def train(self, train_loader: DataLoader):
    
    self.model.train()
    for epoch in range(self.epochs):

      running_loss = 0.0
      for X, y in train_loader:

        X = X.to(self.device)
        y = y.to(self.device)

        self.optimizer.zero_grad()

        y_hat = self.model(X)
        loss = self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()

      print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(train_loader):.4f}')


  def evaluate(self, test_loader: DataLoader):

    self.model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
      for X, y in test_loader:

        X = X.to(self.device)
        y = y.to(self.device)

        outputs = self.model(X)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  model = CNN()

  transform = transforms.Compose([
    transforms.ToTensor(),
  ])

  train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

  train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

  trainer = Trainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
    epochs=20,
    criterion=nn.CrossEntropyLoss(),
    device=device
  )

  trainer.train(train_loader)

  accuracy = trainer.evaluate(train_loader)
  print(f'Test Accuracy: {accuracy:.2f}%') # 99.91%

  torch.save(model.state_dict(), 'models/model.pth')