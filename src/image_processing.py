import cv2
import numpy as np
from typing import List


IMAGE_DIM = 720
SINGLE_BOX_DIM = IMAGE_DIM // 9
MARGIN = 4


def preprocess_image(img: np.ndarray) -> np.ndarray:

  # img = cv2.resize(img, (IMAGE_DIM, IMAGE_DIM))
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_blurred = cv2.GaussianBlur(img_gray, (9, 9), 0)
  img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

  img_thresh = cv2.bitwise_not(img_thresh, img_thresh)

  return img_thresh


def find_grid_corners(image_thresh: np.ndarray) -> np.ndarray:
    
    contours, h = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    biggest = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return np.array(biggest)


def sort_points(points: np.ndarray) -> np.ndarray:
    
    points = points.reshape((4, 2))
    sorted_points = np.zeros((4, 1, 2), dtype=np.int32)

    add = points.sum(1)
    sorted_points[0] = points[np.argmin(add)]
    sorted_points[3] = points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    sorted_points[1] =points[np.argmin(diff)]
    sorted_points[2] = points[np.argmax(diff)]

    return sorted_points


def slice_image(img):

	boxes = []
	for i in range(9):
		left = i* SINGLE_BOX_DIM
		right = i*SINGLE_BOX_DIM + SINGLE_BOX_DIM
		for j in range(9):
			top = j*SINGLE_BOX_DIM
			bottom = j*SINGLE_BOX_DIM + SINGLE_BOX_DIM

			boxes.append(img[left+MARGIN: right-MARGIN, top+MARGIN: bottom-MARGIN])
	
	return boxes


def warp_image(image: np.ndarray, corners: List[np.ndarray]) -> np.ndarray:

  src = np.float32(corners)
  dst = np.float32([[0, 0],[IMAGE_DIM, 0], [0, IMAGE_DIM],[IMAGE_DIM, IMAGE_DIM]])

  matrix = cv2.getPerspectiveTransform(src, dst)
  img_warped = cv2.warpPerspective(image, matrix, (IMAGE_DIM, IMAGE_DIM))

  return img_warped


def get_corners(img_thresh: np.ndarray) -> List[np.ndarray]:

  corners = find_grid_corners(img_thresh)
  corners = sort_points(corners)

  return corners

def prepare_image(image: np.ndarray) -> np.ndarray:
  
  img_thresh = preprocess_image(image)

  corners = get_corners(img_thresh)

  img_warped = warp_image(image, corners)
  boxes = slice_image(img_warped)

  return boxes, img_warped


def display_solution(original: np.ndarray, image: np.ndarray, overlay: List[List[int]]):
  
  original = cv2.resize(original, (IMAGE_DIM, IMAGE_DIM))
  image = cv2.resize(image, (IMAGE_DIM, IMAGE_DIM))

  l, w = IMAGE_DIM // 9, IMAGE_DIM // 9

  for i in range(9):
    for j in range(9):
        if overlay[i][j] != 0:
            
            cv2.putText(
                image,
                str(overlay[i][j]),
                (int(j * w + w // 3), int(i * l + l // 1.5)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2,
                (0, 255, 0), 2, cv2.LINE_AA
            )
  
  corners = get_corners(preprocess_image(original))
  
  l, w = original.shape[:2]
  dst = np.float32(corners)
  src = np.float32([[0, 0],[IMAGE_DIM, 0], [0, IMAGE_DIM],[IMAGE_DIM, IMAGE_DIM]])
  
  matrix = cv2.getPerspectiveTransform(src, dst)

  imgInvWarpColored = cv2.warpPerspective(image, matrix, (IMAGE_DIM, IMAGE_DIM))
  inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, original, 0.5, 1)

  # cv2.imshow('Sudoku Grid', inv_perspective)
  # cv2.waitKey(0)

  return inv_perspective
          

if __name__ == "__main__":

  image = cv2.imread('images/image.png')
  prepare_image(image)
