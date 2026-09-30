import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "custom_cnn_cifar10.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout(0.25))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path=MODEL_PATH, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CustomCNN(num_classes=len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    meta = {
        "classes": ckpt["classes"],
        "mean": torch.tensor(ckpt["mean"]).view(3, 1, 1),
        "std": torch.tensor(ckpt["std"]).view(3, 1, 1),
        "size": ckpt["input_size"],
    }
    return model, meta


def preprocess(image, meta):
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB").resize((meta["size"], meta["size"]), Image.LANCZOS)
    x = torch.from_numpy(__import__("numpy").array(image)).permute(2, 0, 1).float() / 255.0
    return ((x - meta["mean"]) / meta["std"]).unsqueeze(0)


@torch.no_grad()
def predict(model, meta, image, k=5, device=DEVICE):
    """image: file path or PIL image. Returns [(class_name, probability), ...]."""
    x = preprocess(image, meta).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    top_p, top_i = probs.topk(k)
    return [(meta["classes"][i], p.item()) for i, p in zip(top_i.tolist(), top_p)]


if __name__ == "__main__":
    import sys
    model, meta = load_model()
    for name, conf in predict(model, meta, sys.argv[1]):
        print(f"{name:<12} {conf * 100:6.2f}%")