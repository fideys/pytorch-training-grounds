import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train_marko import CLASSES, CNN, get_device


DEFAULT_MODEL_PATH = Path(__file__).with_name("marko_cifar10.pth")


def load_model(model_path, device):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            "Train the model first with train_marko.py."
        )

    model = CNN().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    with Image.open(image_path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)


def choose_image():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected_path = filedialog.askopenfilename(
        title="Choose an image for Marko",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not selected_path:
        raise SystemExit("No image selected.")
    return Path(selected_path)


def predict(model, image, device, top_k):
    with torch.no_grad():
        probabilities = torch.softmax(model(image.to(device)), dim=1)[0]

    scores, indices = probabilities.topk(top_k)
    return [
        (CLASSES[index.item()], score.item())
        for score, index in zip(scores, indices)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify an image with Marko's trained CIFAR-10 CNN."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help="path to the image to classify (opens a file picker when omitted)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"model checkpoint (default: {DEFAULT_MODEL_PATH.name})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        choices=range(1, len(CLASSES) + 1),
        metavar=f"1-{len(CLASSES)}",
        help="number of predictions to show (default: 3)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    try:
        model = load_model(args.model, device)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise SystemExit(error) from error

    image_path = args.image

    while True:
        image_path = image_path or choose_image()

        try:
            image = load_image(image_path)
        except (FileNotFoundError, OSError, RuntimeError) as error:
            print(error)
            image_path = None
            continue

        predictions = predict(model, image, device, args.top_k)

        print(f"\nUsing device: {device}")
        print(f"Image: {image_path}")
        print("Predictions:")
        for class_name, probability in predictions:
            print(f"  {class_name:<10} {probability:.2%}")

        choice = input(
            "\nPress Enter to choose another image, or type Q to exit: "
        ).strip().lower()
        if choice in {"q", "quit", "exit"}:
            break

        image_path = None


if __name__ == "__main__":
    main()
