import cv2
import numpy as np
import time
from PIL import ImageGrab

class Box:
    def __init__(self, left: int, top: int, width: int, height: int):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def __repr__(self) -> str:
        return f"Box(left={self.left}, top={self.top}, width={self.width}, height={self.height})"


class ImageArray:
    def __init__(self, path: str, image: np.ndarray):
        channels, width, height = image.shape[::-1]
        if channels == 4:
            b, g, r, a = cv2.split(image)
            mask = a
            array = cv2.merge((b, g, r))
        else:
            mask = None
            array = image

        self.name = path.split("/")[-1]
        self.array = array
        self.mask = mask
        self.width = width
        self.height = height


def is_valid_location(image: np.ndarray, x: int, y: int, w: int, h: int) -> bool:
    roi = image[y:y+h, x:x+w]
    return np.mean(roi) > 10


def match_template(image: np.ndarray, template: ImageArray, confidence: float) -> list[tuple[Box, float]]:
    start_time = time.time()
    result = cv2.matchTemplate(image, template.array, cv2.TM_CCOEFF_NORMED, mask=template.mask)
    locations = np.where((result >= confidence) & (~np.isinf(result)))
    coordinates = []
    for pt, corr in zip(zip(*locations[::-1]), result[locations]):
        if is_valid_location(image, pt[0], pt[1], template.width, template.height):
            box = Box(pt[0], pt[1], template.width, template.height)
            coordinates.append((box, corr))
    elapsed_time = time.time() - start_time
    print(f"Tiempo de ejecución de match_template: {elapsed_time:.4f} segundos")
    return coordinates


def locate_all_images(
    template: ImageArray,
    image: np.ndarray,
    confidence: float = 1.0,
    test_mode: bool = False
) -> list[Box]:
    if test_mode:
        coordinates = []
        data = match_template(image, template, confidence)
        for box, corr in data:
            print(f"{box} with value correlation {corr}")
            coordinates.append(box)
        return coordinates
    else:
        return [Box(pt[1], pt[0], template.width, template.height) for pt in zip(*np.where((cv2.matchTemplate(image, template.array, cv2.TM_CCOEFF_NORMED, mask=template.mask) >= confidence) & (~np.isinf(cv2.matchTemplate(image, template.array, cv2.TM_CCOEFF_NORMED, mask=template.mask)))[::-1]))]


def locate_image(
    template: ImageArray,
    image: np.ndarray,
    confidence: float = 1.0,
    test_mode: bool = False
) -> Box | None:
    if test_mode:
        data = match_template(image, template, confidence)
        if data:
            print(f"{data[0][0]} with value correlation {data[0][1]}")
            return data[0][0]
        else:
            return None
    else:
        data = locate_all_images(template, image, confidence)
        if data:
            return data[0]
        else:
            return None


def screenshot(bbox: tuple[int, int, int, int] | None = None) -> np.ndarray:
    image = ImageGrab.grab(bbox=bbox)
    image_array = np.array(image)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def visualize(image: np.ndarray, boxes: list[Box] | Box, save: bool = False):
    if isinstance(boxes, list):
        for box in boxes:
            top_left = box.left, box.top
            bottom_right = box.right, box.bottom
            cv2.rectangle(image, top_left, bottom_right, 255, 1)
    else:
        top_left = boxes.left, boxes.top
        bottom_right = boxes.right, boxes.bottom
        cv2.rectangle(image, top_left, bottom_right, 255, 1)

    cv2.imshow("Visualize Found Images", image)
    if save:
        cv2.imwrite("visualize_found_images.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def open_image(path: str) -> ImageArray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Template image not found or could not be loaded.")
    return ImageArray(path, image)