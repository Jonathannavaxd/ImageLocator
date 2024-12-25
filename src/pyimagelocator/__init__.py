import cv2
import numpy as np
import time
from PIL import ImageGrab
import random
from typing import Union, List, Optional

class Box:
    """
    Representa un área rectangular definida por sus coordenadas y dimensiones.

    La clase Box permite realizar operaciones sobre un área rectangular, como
    verificar si un punto está contenido dentro del área, calcular un área
    reducida a un porcentaje específico y generar coordenadas aleatorias dentro
    del área.

    Attributes:
        left (int): Coordenada izquierda del área.
        top (int): Coordenada superior del área.
        width (int): Ancho del área.
        height (int): Altura del área.
    """

    def __init__(self, left: int, top: int, width: int, height: int):
        """
        Inicializa una nueva instancia de la clase Box.

        Args:
            left (int): Coordenada izquierda del área.
            top (int): Coordenada superior del área.
            width (int): Ancho del área.
            height (int): Altura del área.
        """
        self.left = left
        self.top = top
        self.width = width
        self.height = height

    @property
    def right(self) -> int:
        """
        Coordenada derecha del área.

        Returns:
            int: La coordenada derecha calculada como left + width.
        """
        return self.left + self.width

    @property
    def bottom(self) -> int:
        """
        Coordenada inferior del área.

        Returns:
            int: La coordenada inferior calculada como top + height.
        """
        return self.top + self.height

    @property
    def area(self) -> int:
        """
        Calcula el área del rectángulo.

        Returns:
            int: El área calculada como width * height.
        """
        return self.width * self.height

    def contains(self, x: int, y: int) -> bool:
        """
        Verifica si un punto (x, y) está contenido dentro del área.

        Args:
            x (int): Coordenada x del punto a verificar.
            y (int): Coordenada y del punto a verificar.

        Returns:
            bool: True si el punto está dentro del área, False en caso contrario.
        """
        return self.left <= x < self.right and self.top <= y < self.bottom
    
    def area_percentage(self, area_percentage: int = 100) -> tuple[int, int, int, int]:
        """
        Limita el área a un porcentaje específico y devuelve las nuevas coordenadas.

        Args:
            area_percentage (int, opcional): Porcentaje al que se desea limitar el área (0 a 100).
                Por defecto es 100.

        Returns:
            tuple[int, int, int, int]: Nuevas coordenadas (left, top, right, bottom) del área limitada.

        Raises:
            ValueError: Si el porcentaje no está en el rango de 0 a 100.
        """
        if not (0 <= area_percentage <= 100):
            raise ValueError("El porcentaje debe estar entre 0 y 100.")

        new_width = self.width * (area_percentage / 100)
        new_height = self.height * (area_percentage / 100)
        new_left = self.left + (self.width - new_width) / 2
        new_top = self.top + (self.height - new_height) / 2
        new_right = new_left + new_width
        new_bottom = new_top + new_height
        return int(new_left), int(new_top), int(new_right), int(new_bottom)
    
    def random_coordinate(self, area_percentage: int = 100) -> tuple[int, int]:
        """
        Genera una coordenada aleatoria dentro del área limitada a un porcentaje específico.

        Args:
            area_percentage (int, opcional): Porcentaje al que se desea limitar el área (0 a 100).
                Por defecto es 100.

        Returns:
            tuple[int, int]: Coordenadas (x, y) aleatorias dentro del área limitada.
        """
        left, top, right, bottom = self.area_percentage(area_percentage)
        return (random.randint(left, right), random.randint(top, bottom))

    def __repr__(self) -> str:
        """
        Representación en cadena de la instancia Box.

        Returns:
            str: Una representación en cadena de la instancia Box.
        """
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

def LocateAllImages(
    template: Union[ImageArray, np.ndarray],
    image: Union[ImageArray, np.ndarray],
    confidence: float = 0.8) -> List[Box]:

    # Template have to be an ImageArray.
    if isinstance(template, ImageArray):
        pass
    elif isinstance(template, np.ndarray):
        template = ImageArray("default", template)
    else:
        raise ValueError("Template must be an ImageArray or a numpy array.")

    # Image have to be an np.ndarray.
    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, ImageArray):
        image = image.array
    else:
        raise ValueError("Image must be an ImageArray or a numpy array.")

    results = cv2.matchTemplate(image, template.array, cv2.TM_CCOEFF_NORMED, mask=template.mask)

    # Filtrar valores infinitos y NaN.
    valid_indices = np.where((~np.isnan(results)) & (~np.isinf(results)) & (results >= confidence))
    valid_results = results[valid_indices]

    # Ordenar las ubicaciones por valor de correlación de mayor a menor
    sorted_indices = np.argsort(-valid_results)
    sorted_valid_results = valid_results[sorted_indices]
    sorted_valid_indices = (valid_indices[0][sorted_indices], valid_indices[1][sorted_indices])

    # Verificar cada ubicación desde la mayor correlación hasta la menor
    boxes = []
    for i in range(len(sorted_valid_results)):
        x, y = sorted_valid_indices[1][i], sorted_valid_indices[0][i]
        roi = image[y:y+template.height, x:x+template.width]
        if np.mean(roi) > 10:
            box = Box(x, y, template.width, template.height)
            boxes.append(box)
    return boxes

def LocateImage(
    template: Union[ImageArray, np.ndarray],
    image: Union[ImageArray, np.ndarray],
    confidence: float = 0.8,
    test: bool = False) -> Optional[Box]:

    # Template have to be an ImageArray.
    if isinstance(template, ImageArray):
        pass
    elif isinstance(template, np.ndarray):
        template = ImageArray("default", template)
    else:
        raise ValueError("Template must be an ImageArray or a numpy array.")

    # Image have to be an np.ndarray.
    if isinstance(image, np.ndarray):
        pass
    elif isinstance(image, ImageArray):
        image = image.array
    else:
        raise ValueError("Image must be an ImageArray or a numpy array.")
    
    if test:
        start_time = time.time()
    
    results = cv2.matchTemplate(image, template.array, cv2.TM_CCOEFF_NORMED, mask=template.mask)

    # Filtrar valores infinitos y NaN antes de calcular el máximo
    valid_indices = np.where((~np.isnan(results)) & (~np.isinf(results)))
    valid_results = results[valid_indices]
    max_correlation = np.nanmax(valid_results) if valid_results.size > 0 else -1

    if max_correlation < confidence:
        if test:
            elapsed_time = time.time() - start_time
            print(f"No se encontró una correlación mayor o igual a la confianza: {elapsed_time:.4f} segundos, max cor: {max_correlation}")
        return None

    # Obtener la ubicación de la máxima correlación
    max_index = np.argmax(valid_results)
    max_loc = (valid_indices[1][max_index], valid_indices[0][max_index])

    # Crear la caja con la ubicación de la máxima correlación
    box = Box(max_loc[0], max_loc[1], template.width, template.height)

    if test:
        elapsed_time = time.time() - start_time
        print(f"Tiempo de ejecución de find_max_correlation_location: {elapsed_time:.4f} segundos, max cor: {max_correlation}")
    return box

def Screenshot(region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    if region:
        bbox = region[0], region[1], region[2] + region[0], region[3] + region[1]
    else:
        bbox = region
    
    image = ImageGrab.grab(bbox=bbox)
    image_array = np.array(image)
    return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)


def Visualize(boxes: list[Box] | Box, image: np.ndarray, save: bool = False):
    if not boxes:
        raise ValueError("No boxes found to visualize.")
    
    if isinstance(image, ImageArray):
        image = image.array
    
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


def ImageOpen(path: str) -> ImageArray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Template image not found or could not be loaded.")
    return ImageArray(path, image)