import cv2
import pandas as pd
import numpy as np

IMAGE_FILE = "data/pic1.jpg"
COLORS_FILE = "data/colors.csv"

class ColorIdentifier:
    def __init__(self, image_file, colors_file):
        self.image_file = image_file
        self.colors_file = colors_file
        self.image = self.load_image(image_file)
        self.color_dataset = self.load_color_dataset(colors_file)
        self.resized_image = self.resize_image(self.image)

    def load_image(self, image_file):

        try:
            image = cv2.imread(image_file)
            if image is None:
                raise FileNotFoundError(f"Image file '{image_file}' not found or could not be loaded.")
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            exit(1)

    def resize_image(self, image, width=1000):
        height = int(image.shape[0] * width / image.shape[1])
        return cv2.resize(image, (width, height))

    def load_color_dataset(self, colors_file):
        try:
            dataset = pd.read_csv(colors_file, names=["color", "hex", "r", "g", "b"])
            if dataset.empty:
                raise ValueError(f"Color dataset '{colors_file}' is empty.")
            return dataset
        except Exception as e:
            print(f"Error loading color dataset: {e}")
            exit(1)

    def get_color_name(self, r, g, b):
        color_values = self.color_dataset[["r", "g", "b"]].values
        color_distances = np.linalg.norm(color_values - np.array([r, g, b]), axis=1)
        closest_color_index = np.argmin(color_distances)
        return self.color_dataset.iloc[closest_color_index]["color"]

    def handle_double_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            # Scale coordinates for resized image
            original_x = int(x * self.image.shape[1] / self.resized_image.shape[1])
            original_y = int(y * self.image.shape[0] / self.resized_image.shape[0])
            b, g, r = self.image[original_y, original_x]
            color_name = self.get_color_name(r, g, b)
            print(f"Clicked color: {color_name} (R={r}, G={g}, B={b})")
            
            # Draw text with background for better visibility
            font_scale = 0.6
            font_thickness = 2
            text = f"{color_name} (R={r}, G={g}, B={b})"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x, text_y = x, y
            box_coords = ((text_x, text_y - 10), (text_x + text_size[0], text_y + text_size[1]))
            cv2.rectangle(self.resized_image, box_coords[0], box_coords[1], (0, 0, 0), -1)
            cv2.putText(self.resized_image, text, (text_x, text_y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255, 255, 255), font_thickness)

    def run(self):
        print("Double-click on the image to identify a color. Press 'q' or 'Esc' to exit.")
        cv2.namedWindow("Image Viewer")
        cv2.setMouseCallback("Image Viewer", self.handle_double_click)

        while True:
            cv2.imshow("Image Viewer", self.resized_image)
            key = cv2.waitKey(1)
            if key & 0xFF in {ord('q'), 27} or cv2.getWindowProperty("Image Viewer", cv2.WND_PROP_VISIBLE) < 1:
                print("Exiting program...")
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    color_identifier = ColorIdentifier(IMAGE_FILE, COLORS_FILE)
    color_identifier.run()
