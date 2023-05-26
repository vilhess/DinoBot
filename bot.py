import keyboard
import cv2
import numpy as np
from PIL import ImageGrab
import torch
from torchvision.transforms import CenterCrop, Resize, Compose, ToTensor, Normalize
import torchvision
from inceptionV3 import InceptionV3


device = 'mps'
path_v2 = 'efficientnet_v2_s.pth'
path_v3 = 'efficientnet_v3.pth'

try:
    saved_model_v2 = torchvision.models.efficientnet_v2_s()
    saved_model_v2.classifier = torch.nn.Linear(
        in_features=1280, out_features=2)
    saved_model_v2.load_state_dict(torch.load(path_v2))
    saved_model_v2 = saved_model_v2.to(device)
    saved_model_v2 = saved_model_v2.eval()

except:
    print('train model first')


try:
    saved_model_v3 = InceptionV3()
    saved_model_v3.load_state_dict(torch.load(path_v3))
    saved_model_v3 = saved_model_v3.to(device)
    saved_model_v3 = saved_model_v3.eval()

except:
    print('train model first')

# change to saved_model_v3 if you want to use inception v3 but you have to train it first
saved_model = saved_model_v2


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
