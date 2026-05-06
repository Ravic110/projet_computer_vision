"""Entry point for the text detection application."""

import tkinter as tk

from text_detector.text_detector import TextRecognitionApp
from text_detector.utils.logging_setup import setup_logging


def main() -> None:
    """Run the text recognition application."""
    setup_logging()
    root = tk.Tk()
    app = TextRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()