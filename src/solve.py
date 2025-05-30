from collections import defaultdict
from typing import Tuple, List
import time

def is_valid(grid: List[List[int]]) -> bool:

  set_rows = defaultdict(set)
  set_cols = defaultdict(set)
  set_boxes = defaultdict(set)

  for i in range(9):
    for j in range(9):
      
      if grid[i][j] == 0:
        continue

      if grid[i][j] in set_rows[i] or grid[i][j] in set_cols[j] or grid[i][j] in set_boxes[(i // 3, j // 3)]:
        return False
      
      set_rows[i].add(grid[i][j])
      set_cols[j].add(grid[i][j])
      set_boxes[(i // 3, j // 3)].add(grid[i][j])
  
  return True



def solve_sudoku(grid: List[List[int]]) -> bool:

  for i in range(9):
    for j in range(9):
  
      if grid[i][j] == 0:
  
        for num in range(1, 10):
          grid[i][j] = num
          if is_valid(grid):
            if solve_sudoku(grid):
              return True
          grid[i][j] = 0
  
        return False
  return True


if __name__ == "__main__":
  

  puzzle = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
  ]

  start = time.time()

  if solve_sudoku(puzzle):
    print("Sudoku solved:")
    print(puzzle)
  
  print(time.time() - start)