import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageTk

from predict import load_model, predict

model, meta = load_model()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIFAR-10 Classifier")
        self.resizable(False, False)

        ttk.Button(self, text="Choose Image...", command=self.choose).grid(
            row=0, column=0, columnspan=3, pady=10)

        self.preview = ttk.Label(self)
        self.preview.grid(row=1, column=0, columnspan=3, padx=10)

        self.rows = []
        for r in range(5):
            name = ttk.Label(self, text="", width=14, anchor="w")
            bar = ttk.Progressbar(self, length=220, maximum=100)
            pct = ttk.Label(self, text="", width=8, anchor="e")
            name.grid(row=r + 2, column=0, padx=(10, 5), pady=3)
            bar.grid(row=r + 2, column=1, pady=3)
            pct.grid(row=r + 2, column=2, padx=(5, 10), pady=3)
            self.rows.append((name, bar, pct))

    def choose(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        img = Image.open(path)

        thumb = img.convert("RGB")
        thumb.thumbnail((256, 256))
        self.tk_img = ImageTk.PhotoImage(thumb)  # keep a reference
        self.preview.configure(image=self.tk_img)

        results = predict(model, meta, img, k=5)
        for (name, bar, pct), (label, conf) in zip(self.rows, results):
            name.configure(text=label)
            bar["value"] = conf * 100
            pct.configure(text=f"{conf * 100:.2f}%")


if __name__ == "__main__":
    App().mainloop()