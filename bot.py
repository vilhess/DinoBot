import keyboard
import cv2
import numpy as np
from PIL import ImageGrab
import torch
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor, Normalize
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'
path = 'efficientnet_v2_s.pth'

saved_model = torchvision.models.efficientnet_v2_s()
saved_model.classifier = torch.nn.Linear(in_features=1280, out_features=2)
saved_model.load_state_dict(torch.load(path))
saved_model = saved_model.to(device)
saved_mode = saved_model.eval()


transformer = Compose([
    Resize((480, 480)),
    CenterCrop(480),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def action(pred):
    pred = pred.detach().cpu().numpy()[0]
    # if pred=1 then jump
    if pred == 1:
        keyboard.press('space')
        keyboard.release('space')
        print('jump')
    else:
        print('no jump')


def pred():
    while (not keyboard.is_pressed("esc")):

        # Capture image and save to the 'captures' folder with time and date along with the key being pressed
        image = cv2.cvtColor(np.array(ImageGrab.grab(
            bbox=(450, 260, 1055, 380))), cv2.COLOR_RGB2BGR)  # Adapt the bbox to your screen
        image = ToTensor()(image)
        image = transformer(image)
        with torch.no_grad():
            image = image.to(device)
            image = image.unsqueeze(0)
            pred = saved_model(image)
            pred = torch.softmax(pred, dim=1).argmax(dim=1)
            action(pred)


if __name__ == "__main__":
    pred()
