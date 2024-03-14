import cv2
import pandas as pd
import numpy as np
import webcolors
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture

IMAGE_FILE = "colorpic.jpg"
COLORS_FILE = "colors.csv"


class ColorIdentifier:
    def __init__(self, colors_file):
        self.color_dataset = self.load_color_dataset(colors_file)

    def load_color_dataset(self, colors_file):
        return pd.read_csv(colors_file,
                           names=["color", "hex", "r", "g", "b"],
                           dtype={"color": str,
                                  "hex": str,
                                  "r": np.uint8,
                                  "g": np.uint8,
                                  "b": np.uint8})

    def get_color_name(self, r, g, b):
        def get_color_name(self, r, g, b):
            query_color = (r, g, b)
            try:
                color_name = webcolors.rgb_to_name(query_color)
            except ValueError:
                # Si la couleur n'est pas trouvée dans la bibliothèque, vous pouvez utiliser votre méthode actuelle
                color_values = self.color_dataset[["r", "g", "b"]].values.astype(np.uint8)
                color_distances = np.linalg.norm(color_values - np.array([r, g, b], dtype=np.uint8), axis=1)
                closest_color_index = np.argmin(color_distances)
                color_name = self.color_dataset.iloc[closest_color_index]["color"]
            print(color_name)
            return color_name


class ColorIdentifierApp(App):
    def build(self):
        self.color_identifier = ColorIdentifier(COLORS_FILE)
        self.layout = BoxLayout(orientation="vertical")
        self.image = KivyImage()
        self.layout.add_widget(self.image)
        self.load_image(IMAGE_FILE)
        return self.layout

    def load_image(self, image_file):
        cv_image = cv2.imread(image_file)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        cv_image = np.flipud(cv_image)
        height, width = cv_image.shape[:2]
        texture = Texture.create(size=(width, height))
        texture.blit_buffer(cv_image.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image.texture = texture
        self.image.size = width, height
        self.image.allow_stretch = True
        self.image.bind(on_touch_down=self.on_image_click)

    def on_image_click(self, instance, touch):
        if self.image.collide_point(*touch.pos):
            x, y = touch.pos
            x_ratio = x / self.image.width
            y_ratio = y / self.image.height
            texture = self.image.texture
            texture_size = list(texture.size)
            pixels = texture.pixels
            pixel_width = int(texture_size[0])
            pixel_height = int(texture_size[1])

            pixel_x = int(x_ratio * pixel_width)
            pixel_y = int(y_ratio * pixel_height)

            index = (pixel_y * pixel_width + pixel_x) * 3
            r, g, b = pixels[index + 2], pixels[index + 1], pixels[index]

            color_name = self.color_identifier.get_color_name(int(r * 255), int(g * 255), int(b * 255))
            print(f"Color name: {color_name}")


if __name__ == "__main__":
    ColorIdentifierApp().run()
