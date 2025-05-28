from typing import List
from collections import defaultdict
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



def solve_sudoku(grid: List[List[int]]) -> List[List[int]] or None:
  
  if not is_valid(grid):
    return None 
  
  for i in range(9):
    for j in range(9):
  
      if grid[i][j] == 0:
  
        for num in range(1, 10):
          grid[i][j] = num
          if is_valid(grid):
            if solve_sudoku(grid):
              return grid
          grid[i][j] = 0
  
        return None
  return grid


# if __name__ == "__main__":
#   sudoku_puzzles = [

#       # Puzzle 1 - Valid partially filled board
#       [
#           [5, 3, 0, 0, 7, 0, 0, 0, 0],
#           [6, 0, 0, 1, 9, 5, 0, 0, 0],
#           [0, 9, 8, 0, 0, 0, 0, 6, 0],
#           [8, 0, 0, 0, 6, 0, 0, 0, 3],
#           [4, 0, 0, 8, 0, 3, 0, 0, 1],
#           [7, 0, 0, 0, 2, 0, 0, 0, 6],
#           [0, 6, 0, 0, 0, 0, 2, 8, 0],
#           [0, 0, 0, 4, 1, 9, 0, 0, 5],
#           [0, 0, 0, 0, 8, 0, 0, 7, 9],
#       ],

#       # Puzzle 2 - Invalid: duplicate '5' in first row
#       [
#           [5, 3, 5, 0, 7, 0, 0, 0, 0],
#           [6, 0, 0, 1, 9, 5, 0, 0, 0],
#           [0, 9, 8, 0, 0, 0, 0, 6, 0],
#           [8, 0, 0, 0, 6, 0, 0, 0, 3],
#           [4, 0, 0, 8, 0, 3, 0, 0, 1],
#           [7, 0, 0, 0, 2, 0, 0, 0, 6],
#           [0, 6, 0, 0, 0, 0, 2, 8, 0],
#           [0, 0, 0, 4, 1, 9, 0, 0, 5],
#           [0, 0, 0, 0, 8, 0, 0, 7, 9],
#       ],

#       # Puzzle 3 - Invalid: duplicate in 3x3 box
#       [
#           [5, 3, 0, 0, 7, 0, 0, 0, 0],
#           [6, 0, 0, 1, 9, 5, 0, 0, 0],
#           [5, 9, 8, 0, 0, 0, 0, 6, 0],  # <- duplicate 5 in top-left box
#           [8, 0, 0, 0, 6, 0, 0, 0, 3],
#           [4, 0, 0, 8, 0, 3, 0, 0, 1],
#           [7, 0, 0, 0, 2, 0, 0, 0, 6],
#           [0, 6, 0, 0, 0, 0, 2, 8, 0],
#           [0, 0, 0, 4, 1, 9, 0, 0, 5],
#           [0, 0, 0, 0, 8, 0, 0, 7, 9],
#       ],

#       # Puzzle 4 - Valid sparse board
#       [
#           [0, 0, 0, 0, 0, 0, 0, 1, 2],
#           [0, 0, 0, 0, 0, 0, 0, 3, 4],
#           [0, 0, 0, 0, 0, 0, 0, 5, 6],
#           [0, 0, 0, 0, 0, 0, 0, 7, 8],
#           [0, 0, 0, 0, 0, 0, 0, 9, 1],
#           [0, 0, 0, 0, 0, 0, 0, 2, 3],
#           [0, 0, 0, 0, 0, 0, 0, 4, 5],
#           [0, 0, 0, 0, 0, 0, 0, 6, 7],
#           [0, 0, 0, 0, 0, 0, 0, 8, 9],
#       ],

#       # Puzzle 5 - Invalid: duplicate in a column
#       [
#           [5, 3, 0, 0, 7, 0, 0, 0, 0],
#           [6, 0, 0, 1, 9, 5, 0, 0, 0],
#           [0, 9, 8, 0, 0, 0, 0, 6, 0],
#           [8, 0, 0, 0, 6, 0, 0, 0, 3],
#           [4, 0, 0, 8, 0, 3, 0, 0, 1],
#           [7, 0, 0, 0, 2, 0, 0, 0, 6],
#           [5, 6, 0, 0, 0, 0, 2, 8, 0],  # <- duplicate 5 in column 0
#           [0, 0, 0, 4, 1, 9, 0, 0, 5],
#           [0, 0, 0, 0, 8, 0, 0, 7, 9],
#       ]
#   ]
  
#   start = time.time()
#   solve_sudoku(sudoku_puzzles[0])
#   print(time.time() - start)