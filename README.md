# Computer Vision Sudoku Solver

This app takes an image of Sudoku puzzles via a FLask server. The image is processed using GaussianBlur and AdaptiveThreshold .Puzzle grid is extracted by finding the biggest contour, It's split into 81 squares. The number in each grid is predicted using a Convolutional Neural Network of 99.91% accuracy. Puzzle is solved using backtracking algorithm, and solution is overlayed on the original image.

# Demo
https://github.com/user-attachments/assets/6a75f715-6b09-4fb8-ba00-464c9930e4b8


# Installation & Usage

1. Go to src and create a virutal enviroment `cd src && python -m venv venv`
2. Activate it `source venv/bin/activate`
3. Install dependencies `pip install requirements.txt`
4. run app `python app.py`