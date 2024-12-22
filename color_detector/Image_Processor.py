import cv2
import numpy as np
from typing import Tuple
class ImageProcessor:
    """Classe pour gérer le traitement des images haute résolution"""
    
    @staticmethod
    def calculate_optimal_size(image: np.ndarray, max_dimension: int = 1920) -> Tuple[int, int]:
        """Calcule la taille optimale pour l'affichage en conservant le ratio."""
        height, width = image.shape[:2]
        if max(width, height) <= max_dimension:
            return width, height
            
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
            
        return new_width, new_height
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int, height: int, 
                    interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        """Redimensionne l'image avec l'interpolation spécifiée."""
        return cv2.resize(image, (width, height), interpolation=interpolation)
