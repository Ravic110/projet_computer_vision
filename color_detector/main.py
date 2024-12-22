import cv2
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, Scale
from PIL import Image, ImageTk
import os
from typing import Tuple, Optional, Callable
from tkinter import messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import csv

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

class DragDropWindow(TkinterDnD.Tk):
    def __init__(self, on_image_loaded: Callable[[str], None]):
        super().__init__()
        
        self.on_image_loaded = on_image_loaded
        self.title("Color Identifier - Glissez une image")
        self.geometry("400x300")
        
        # Style
        self.style = ttk.Style()
        self.style.configure("Drop.TFrame", 
                           relief="solid", 
                           borderwidth=2,
                           background="lightgray")
        
        # Zone de drop
        self.drop_zone = ttk.Frame(self, style="Drop.TFrame")
        self.drop_zone.pack(padx=20, pady=20, expand=True, fill='both')
        
        # Labels
        self.label = ttk.Label(self.drop_zone,
                             text="Glissez une image ici\n\nFormats supportés:\nJPG, PNG, BMP",
                             font=('Arial', 12))
        self.label.pack(expand=True)
        
        # Configuration drag & drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.handle_drop)
        
    def handle_drop(self, event):
        """Gère le drop d'un fichier."""
        file_path = event.data
        # Nettoyage du chemin (enlève les accolades et guillemets si présents)
        file_path = file_path.strip('{}').strip('"')
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        if file_path.lower().endswith(valid_extensions):
            self.on_image_loaded(file_path)
        else:
            messagebox.showerror("Erreur", 
                               "Format non supporté.\nUtilisez JPG, PNG ou BMP.")

class ImageControlWindow(tk.Toplevel):
    """Fenêtre de contrôle pour les paramètres d'affichage"""
    
    def __init__(self, parent, callback):
        super().__init__(parent)
        self.title("Contrôles d'affichage")
        self.geometry("300x400")
        
        self.callback = callback
        
        # Taille maximale
        self.size_frame = ttk.LabelFrame(self, text="Taille d'affichage maximale")
        self.size_frame.pack(padx=10, pady=5, fill="x")
        
        self.size_scale = Scale(self.size_frame, 
                              from_=800, 
                              to=3840,
                              orient="horizontal",
                              label="Pixels",
                              command=self.on_size_change)
        self.size_scale.set(1920)
        self.size_scale.pack(padx=5, pady=5, fill="x")
        
        # Qualité
        self.quality_frame = ttk.LabelFrame(self, text="Qualité d'interpolation")
        self.quality_frame.pack(padx=10, pady=5, fill="x")
        
        self.quality_var = tk.StringVar(value="balanced")
        qualities = [
            ("Rapide (- ressources)", "fast"),
            ("Équilibrée", "balanced"),
            ("Haute qualité (+ ressources)", "high")
        ]
        
        for text, value in qualities:
            ttk.Radiobutton(self.quality_frame,
                           text=text,
                           value=value,
                           variable=self.quality_var,
                           command=self.on_quality_change).pack(anchor="w", padx=5, pady=2)
        
        # Informations
        self.info_frame = ttk.LabelFrame(self, text="Informations")
        self.info_frame.pack(padx=10, pady=5, fill="x")
        
        self.info_label = ttk.Label(self.info_frame,
                                  text="Double-cliquez sur l'image\npour identifier une couleur\n\n"
                                       "Appuyez sur 'Q' ou 'Échap'\npour changer d'image",
                                  justify="left")
        self.info_label.pack(padx=5, pady=5)
        
    def on_size_change(self, *args):
        self.callback(size=int(self.size_scale.get()))
    
    def on_quality_change(self, *args):
        self.callback(quality=self.quality_var.get())

class ColorIdentifier:
    def __init__(self, csv_file_path):
        """Initialise le ColorIdentifier avec le fichier de couleurs."""
        self.colors = self.load_colors(csv_file_path)
        self.selected_color_name: Optional[str] = None
        self.selected_rgb: Optional[Tuple[int, int, int]] = None
        self.history: list[Tuple[str, Tuple[int, int, int]]] = []
        self.original_image = None
        self.display_image = None
        
        # Paramètres d'affichage
        self.max_display_size = 1920
        self.interpolation_quality = "balanced"
        self.interpolation_methods = {
            "fast": cv2.INTER_NEAREST,
            "balanced": cv2.INTER_AREA,
            "high": cv2.INTER_CUBIC
        }
        
        # Création des fenêtres
        self.drag_window = DragDropWindow(self.load_new_image)
        self.control_window = None

    def load_colors(self, csv_file_path):
        colors = {}
        with open(csv_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                color_name = row[1]
                rgb = (int(row[3]), int(row[4]), int(row[5]))
                colors[color_name] = rgb
        return colors
        
    @staticmethod
    def load_color_dataset(colors_file: str) -> pd.DataFrame:
        """Charge le dataset des couleurs."""
        try:
            df = pd.read_csv(colors_file,
                           names=["color", "hex", "r", "g", "b"],
                           dtype={"color": str,
                                 "hex": str,
                                 "r": np.uint8,
                                 "g": np.uint8,
                                 "b": np.uint8})
            if df.isnull().any().any():
                raise ValueError("Le fichier de couleurs contient des valeurs manquantes")
            return df
        except Exception as e:
            raise ValueError(f"Erreur lors du chargement du fichier de couleurs: {e}")

    def load_new_image(self, image_path: str):
        """Charge une nouvelle image et lance l'analyse."""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            height, width = self.original_image.shape[:2]
            print(f"Image chargée - Dimensions: {width}x{height}")
            
            self.update_display_image()
            
            if self.control_window is None:
                self.control_window = ImageControlWindow(self.drag_window, self.update_display_settings)
            
            self.drag_window.withdraw()
            self.run()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image : {str(e)}")

    def update_display_settings(self, size=None, quality=None):
        """Met à jour les paramètres d'affichage et rafraîchit l'image."""
        if size is not None:
            self.max_display_size = size
        if quality is not None:
            self.interpolation_quality = quality
            
        self.update_display_image()

    def update_display_image(self):
        """Met à jour l'image d'affichage selon les paramètres actuels."""
        if self.original_image is None:
            return
            
        new_width, new_height = ImageProcessor.calculate_optimal_size(
            self.original_image, self.max_display_size)
            
        self.display_image = ImageProcessor.resize_image(
            self.original_image,
            new_width,
            new_height,
            self.interpolation_methods[self.interpolation_quality]
        )

    def get_color_name(self, r, g, b):
        """Retourne le nom de la couleur la plus proche."""
        min_distance = float('inf')
        color_name = None
        for name, (red, green, blue) in self.colors.items():
            distance = np.sqrt((r - red) ** 2 + (g - green) ** 2 + (b - blue) ** 2)
            if distance < min_distance:
                min_distance = distance
                color_name = name
        return color_name

    def handle_double_click(self, event, x, y, flags, param):
        """Gère le double-clic de la souris pour identifier la couleur."""
        if event == cv2.EVENT_LBUTTONDBLCLK:
            b, g, r = self.display_image[y, x]
            color_name = self.get_color_name(r, g, b)
            print(f"Color at ({x}, {y}): {color_name}")

    def create_sidebar(self) -> np.ndarray:
        """Crée la barre latérale avec les informations de couleur."""
        sidebar_width = 250
        sidebar_height = self.display_image.shape[0]
        sidebar = np.zeros((sidebar_height, sidebar_width, 3), dtype=np.uint8)

        # Titre
        cv2.putText(sidebar, "Color Information:", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        if self.selected_color_name and self.selected_rgb:
            y_offset = 70
            # Couleur sélectionnée
            cv2.putText(sidebar, f"Name: {self.selected_color_name}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(sidebar, f"RGB: {self.selected_rgb}",
                        (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Rectangle de couleur
            cv2.rectangle(sidebar, (10, y_offset + 40), (60, y_offset + 90),
                         self.selected_rgb, -1)

            # Historique
            cv2.putText(sidebar, "History:", (10, y_offset + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            for i, (hist_name, hist_rgb) in enumerate(reversed(self.history)):
                y_pos = y_offset + 150 + (i * 30)
                cv2.putText(sidebar, f"{hist_name}", (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(sidebar, (180, y_pos - 15), (210, y_pos + 5),
                             hist_rgb, -1)

        return sidebar

    def run(self):
        """Lance l'application."""
        if self.original_image is None:
            self.drag_window.mainloop()
            return

        window_name = "Color Identifier"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.handle_double_click)

        while True:
            if self.display_image is not None:
                sidebar = self.create_sidebar()
                resized_sidebar = cv2.resize(sidebar, (250, self.display_image.shape[0]))
                combined_image = np.hstack((self.display_image, resized_sidebar))
                cv2.imshow(window_name, combined_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                self.drag_window.deiconify()
                break
            
            # Mise à jour de Tkinter
            self.drag_window.update()

if __name__ == "__main__":
    try:
        color_identifier = ColorIdentifier(r"C:\Users\utilisateur\OneDrive\Bureau\victorien\projet ND\projet_computer_vision\color_detector\data\colors.csv")   
        color_identifier.run()
    except Exception as e:
        print(f"Erreur: {e}")