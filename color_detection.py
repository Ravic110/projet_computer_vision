import cv2
import pandas as pd
import numpy as np

IMAGE_FILE = "pic1.jpg"
COLORS_FILE = "colors.csv"


class ColorIdentifier:
    def __init__(self, image_file, colors_file):
        self.image = self.load_image(image_file)
        self.color_dataset = self.load_color_dataset(colors_file)

    def load_image(self, image_file):
        return cv2.imread(image_file)

    def resize_image(self, image):
        width = 1000
        height = int(image.shape[0] * width / image.shape[1])
        return cv2.resize(image, (width, height))

    def load_color_dataset(self, colors_file):
        return pd.read_csv(colors_file, names=["color", "hex", "r", "g", "b"])

    def get_color_name(self, r, g, b):
        # Calculez la distance euclidienne entre les valeurs RVB de l'image et celles de toutes les couleurs
        color_values = self.color_dataset[["r", "g", "b"]].values
        color_distances = np.linalg.norm(color_values - np.array([r, g, b]), axis=1)
        closest_color_index = np.argmin(color_distances)

        return self.color_dataset.iloc[closest_color_index]["color"]

    def handle_double_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            b, g, r = self.image[y, x]
            color_name = self.get_color_name(r, g, b)
            print(color_name)
            cv2.putText(self.image, color_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def run(self):
        cv2.namedWindow("image")
        resized_image = self.resize_image(self.image)
        cv2.setMouseCallback("image", self.handle_double_click)

        while True:
            cv2.imshow("image", resized_image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27 or cv2.getWindowProperty("image", cv2.WND_PROP_VISIBLE) < 1:
                cv2.destroyAllWindows()
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    color_identifier = ColorIdentifier(IMAGE_FILE, COLORS_FILE)
    color_identifier.run()
