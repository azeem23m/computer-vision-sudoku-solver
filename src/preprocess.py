import cv2
import numpy as np


IMAGE_DIM = 720

def preprocess_image(image_path: str) -> np.ndarray:

  img = cv2.imread(image_path)
  img = cv2.resize(img, (IMAGE_DIM, IMAGE_DIM))

  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 1)
  img_thresh = cv2.adaptiveThreshold(
      img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
      cv2.THRESH_BINARY_INV, 11, 2
  )

  return img_thresh, img



def find_biggest_contour(image_thresh: np.ndarray) -> np.ndarray:
    
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
    sorted_points[3] =points[np.argmax(add)]

    diff = np.diff(points, axis=1)
    sorted_points[1] =points[np.argmin(diff)]
    sorted_points[2] = points[np.argmax(diff)]

    return sorted_points


def slice_image(img):

	boxes = []
	for i in range(9):
		left = i* 80
		right = i*80 + 80
		for j in range(9):
			top = j*80
			bottom = j*80 + 80

			boxes.append(img[left+2:right-2,top+2:bottom-2])
	
	return boxes



def prepare_image(image_path: str) -> np.ndarray:
  
  img_thresh, img = preprocess_image(image_path)


  biggest_contour = find_biggest_contour(img_thresh)
  biggest_contour = sort_points(biggest_contour)

  src = np.float32(biggest_contour)
  dst = np.float32([[0, 0],[IMAGE_DIM, 0], [0, IMAGE_DIM],[IMAGE_DIM, IMAGE_DIM]])
  matrix = cv2.getPerspectiveTransform(src, dst)
  img_warped = cv2.warpPerspective(img, matrix, (IMAGE_DIM, IMAGE_DIM))
  img_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

  boxes = slice_image(img_warped)
  print(len(boxes))

  # image = cv2.drawContours(img, biggest_contour, -1, (0, 255, 0), 15)
  cv2.imshow('Warped Image', boxes[0])
  cv2.waitKey()


if __name__ == "__main__":
  prepare_image('images/image.png')
