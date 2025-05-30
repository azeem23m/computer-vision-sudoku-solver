import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from image_processing import prepare_image, preprocess_image, display_solution
import cv2
from solve import solve_sudoku, is_valid
from typing import Any
import os


class CNN(nn.Module):
    
  def __init__(self):

    super().__init__()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.fc1 = nn.Linear(5 * 5 * 64, 128)
    self.dropout = nn.Dropout(p=0.5)  
    self.fc2 = nn.Linear(128, 10)


  def forward(self, x: torch.Tensor) -> torch.Tensor:

    x = self.pool(
      F.relu(self.conv1(x))
    )
    x = self.pool(
      F.relu(self.conv2(x))
    )
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x
  
  def predict(self, image: np.ndarray) -> int:
    self.eval()

    image = preprocess_image(image)

    with torch.no_grad():

      image = cv2.resize(image, (28, 28))
      image = image.astype(np.float32) / 255.0

      # Add batch and channel dimensions: (1, 1, 28, 28)
      tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

      output = self(tensor)
      probabilities = F.softmax(output, dim=1)
      prob, pred = torch.max(probabilities, dim=1)

      return prob.item(), pred.item()


  def load_model(self, path: str):
    self.load_state_dict(torch.load(path, weights_only=True))

  def solve_sudoku(self, image_path) -> Any:

    image = cv2.imread(image_path)
    boxes, img_warp = prepare_image(image)
    grid = []
    for i, box in enumerate(boxes):

      prob, pred = self.predict(box)
      grid.append(pred if prob > 0.7 else 0)

    grid = np.array(grid).reshape((9, 9))

    if not is_valid(grid):
      return False

    solved_grid = grid.copy()
    solve_sudoku(solved_grid)
    overlay = np.array([solved_grid[i][j] if grid[i][j]==0 else 0 for i in range(9) for j in range(9)]).reshape((9, 9))
    solution = display_solution(image, img_warp, overlay)

    path = os.path.join('images', 'solution.png')
    cv2.imwrite(path, solution)

    return True


if __name__ == "__main__":
  model = CNN()
  model.load_model('models/model.pth')

  image = cv2.imread('images/1 (1).png')
  boxes, img_warp = prepare_image(image.copy())
  grid = []
  for i, box in enumerate(boxes):

    prob, pred = model.predict(box)
    grid.append(pred if prob > 0.7 else 0)
    print(f'Box {i+1}: Predicted: {pred}, Probability: {prob:.4f}')

  print(model.predict(boxes[44]))

  grid = np.array(grid).reshape((9, 9))
  
  print(grid)
  
  solved_grid = grid.copy()
  solve_sudoku(solved_grid)
  overlay = np.array([solved_grid[i][j] if grid[i][j]==0 else 0 for i in range(9) for j in range(9)]).reshape((9, 9))
  
  cv2.imshow("Sudoku Grid", display_solution(image, img_warp, overlay))
  cv2.waitKey()