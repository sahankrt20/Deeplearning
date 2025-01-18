import torch
from PIL import Image
from torchvision import transforms
from models.cnn_model import CNNModel

# Configurations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "cnn_model.pth"
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load Model
model = CNNModel(num_classes=10).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Preprocess Image
def preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,))])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict
def predict(image_path):
    image = preprocess_image(image_path).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
    return CLASS_NAMES[predicted.item()]

image_path = "path/to/sample_image.jpg"
prediction = predict(image_path)
print(f"Predicted Class: {prediction}")
