# DinoBot

DinoBot is a bot designed to play the Chrome Dinosaur Game using the EfficientNet model. It is highly inspired by the code from Akshay Ballal on [https://betterprogramming.pub/how-to-build-an-ai-powered-game-bot-with-pytorch-and-efficientnet-f0d47733a0e7]

## Requirements

To run DinoBot, you need to install the required dependencies by using 

`pip install -r requirements.txt`

1. Clone this repository:

`git clone https://github.com/vilhess/DinoBot.git`

2. Navigate to the project directory:

`cd DinoBot`

## How it Works

DinoBot utilizes the EfficientNet model, a state-of-the-art deep learning model, to analyze the game screen and make decisions. The model takes a screenshot of the game and classifies it into different actions: jump or nothing (for the moment).

The bot uses opencv to interact with the Chrome browser and capture screenshots of the game. It then preprocesses the images and feeds them into the EfficientNet model for prediction. Based on the predicted action, the bot sends the appropriate keyboard command to the game.

## Data Collection

Before using the classification system, you need to collect training data. Run the `capture.py` script to capture images of the Dino Run game while pressing the keys corresponding to jump actions. The captured images will be saved in the `captures` folder. Make sure the `captures` folder exists before running the script.

## Model Training

Once you have collected enough images, you can train the classification model. Run the `train.py` script to automatically split the data into training and testing sets, and train the model on the training data. The trained model will be saved to a file named `efficientnet_v2_s.pth`.

## Using the Trained Model

After training the model, you can use it to predict jump actions in the Dino Run game. Run the `predict.py` script to capture screenshots of the game screen, preprocess them, and feed them into the trained model for prediction. Based on the predictions, the script will perform the necessary jump actions using the `keyboard` library.

Make sure you have the trained model file `efficientnet_v2_s.pth` in the same directory as the `predict.py` script before running it.

## Configuration of the Game Screen

Please note that the `predict.py` script uses the `ImageGrab.grab()` function from the `PIL` library to capture the game screen. You may need to adjust the coordinates of the `bbox` parameter in this function in the script to match the area of the screen where the Dino Run game is displayed.

## Acknowledgements

This project was inspired by [akshayballal95](https://github.com/akshayballal95/dino). I borrowed the basic structure and some ideas from their code repository. Many thanks to the contributors of InspirationBot for their valuable work.


