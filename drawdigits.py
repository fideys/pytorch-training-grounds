from testmnist_cnn import CNN, device
import torch
import tkinter as tk
from PIL import Image, ImageDraw #pillow image. model can't see tkinter canvas, so we draw on a pillow image at the same time and turn that into a tensor
from torchvision import transforms

# make sure model loads
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.to(device)
model.eval()

print("Model loaded successfully!")

#tkinter stuff. i don't know how to use tkinter god bless chatgpt
root = tk.Tk()
root.title("Digit Recognizer")



canvas = tk.Canvas(
    root,
    width=560,
    height=560,
    bg="black"
)

canvas.pack()

prediction_label = tk.Label(
    root,
    text="Prediction: ? (Confidence: ?)",
    font=("Cascadia Code SemiBold", 40)
)

prediction_label.pack()

image = Image.new("L", (560, 560), 0)
draw = ImageDraw.Draw(image)

def paint(event):
    x = event.x
    y = event.y

    # draw on both the canvas and the pillow image
    canvas.create_oval(
        x-8, y-8,
        x+8, y+8,
        fill="white",
        outline="white"
    )
    draw.ellipse(
    [x-8, y-8, x+8, y+8],
    fill=255
    )

canvas.bind("<B1-Motion>", paint)

def clear_canvas():
    global image, draw

    canvas.delete("all")

    image = Image.new("L", (560, 560), 0)
    draw = ImageDraw.Draw(image)

    prediction_label.config(text="Prediction: ? (Confidence: ?)")

# do a ton of transforms to get the image into a tensor
def predict():
    small = image.resize((28, 28))

    tensor = transforms.ToTensor()(small)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    with torch.no_grad():
        output = model(tensor)

    probabilities = torch.softmax(output, dim=1)

    prediction = probabilities.argmax(dim=1).item()
    confidence = probabilities.max().item()

    prediction_label.config(
        text=f"Prediction: {prediction} (Confidence: {confidence:.2f})"
    )
    
predict_button = tk.Button(
    root,
    text="Predict",
    command=predict,
    width=20,
    height=2
)

predict_button.pack()

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    width=10,
    height=1
)

clear_button.pack()

# always put at the end of the tkinter code
root.mainloop()
