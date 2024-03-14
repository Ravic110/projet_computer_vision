import cv2
import pandas as pd
import numpy as np

IMAGE_FILE = "colorpic.jpg"
COLORS_FILE = "colors.csv"


class ColorIdentifier:
    def __init__(self, image_file, colors_file):
        self.image = cv2.imread(image_file)
        self.color_dataset = self.load_color_dataset(colors_file)
        self.selected_color_name = None


    @staticmethod
    def load_image(image_file):
        # Utilisation de l'option 0 pour lire l'image en niveaux de gris
        return cv2.imread(image_file, 0)

    @staticmethod
    def resize_image(image):
        # Redimensionnement sans changer les proportions
        width = 800
        ratio = width / image.shape[1]
        return cv2.resize(image, (width, int(image.shape[0] * ratio)))

    @staticmethod
    def load_color_dataset(colors_file):
        # Utilisation de dtype pour optimiser les types de données
        return pd.read_csv(colors_file,
                           names=["color", "hex", "r", "g", "b"],
                           dtype={"color": str,
                                  "hex": str,
                                  "r": np.uint8,
                                  "g": np.uint8,
                                  "b": np.uint8})

    def get_color_name(self, r, g, b):
        color_values = self.color_dataset[["r", "g", "b"]].values.astype(np.uint8)
        color_distances = np.sum(np.abs(color_values - np.array([r, g, b],
                                                                dtype=np.uint8)),
                                 axis=1)
        closest_color_index = np.argmin(color_distances)
        return self.color_dataset.iloc[closest_color_index]["color"]

    def handle_double_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                b, g, r = self.image[y, x]
                color_name = self.get_color_name(r, g, b)
                print(color_name)
                self.selected_color_name = color_name
                cv2.putText(self.image,
                            color_name, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

    def create_sidebar(self):
        sidebar_width = 200
        sidebar_height = self.image.shape[0]

        sidebar = np.zeros((sidebar_height, sidebar_width, 3), dtype=np.uint8)
        cv2.putText(sidebar, "Selected Color:",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        if self.selected_color_name is not None:
            cv2.putText(sidebar, self.selected_color_name,
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        return sidebar

    def run(self):
        cv2.namedWindow("image")
        resized_image = self.resize_image(self.image)
        cv2.setMouseCallback("image", self.handle_double_click)

        while True:
            sidebar_text = "Your text here"  # Mettez ici le texte que vous souhaitez afficher
            sidebar = self.create_sidebar()

            # Redimensionner la barre latérale pour qu'elle ait la même hauteur que l'image principale
            resized_sidebar = cv2.resize(sidebar, (200, resized_image.shape[0]))

            combined_image = np.hstack((resized_image, resized_sidebar))
            cv2.imshow("image", combined_image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27 or cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        color_identifier = ColorIdentifier(IMAGE_FILE, COLORS_FILE)
        color_identifier.run()
    except FileNotFoundError as e:
        print(f"Erreur: {e}. Vérifiez le chemin du fichier.")
    except Exception as e:
        print(f"Une erreur s'est produite: {e}")
