from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import ttk, messagebox
from tkinterdnd2 import DND_FILES
from typing import Callable
import tkinter as tk
from tkinter import ttk, Scale
from PIL import Image, ImageTk
import os

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
