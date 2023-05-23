# DinoBot (readme not finished)

DinoBot is a bot designed to play the Chrome Dinosaur Game using the EfficientNet model. It is highly inspired by the code from Akshay Ballal on [https://betterprogramming.pub/how-to-build-an-ai-powered-game-bot-with-pytorch-and-efficientnet-f0d47733a0e7]

## Requirements

To run DinoBot, you need to install the required dependencies by using 

`pip install -r requirements.txt`

1. Clone this repository:

`git clone https://github.com/vilhess/DinoBot.git`

2. Navigate to the project directory:

`cd DinoBot`

3. Start the Chrome Dinosaur Game in your browser.

4. Run the bot:

`python bot.py`

The bot will automatically connect to the game and start playing.

5. To stop the bot, simply press `Ctrl + C` in the terminal.

## How it Works

DinoBot utilizes the EfficientNet model, a state-of-the-art deep learning model, to analyze the game screen and make decisions. The model takes a screenshot of the game and classifies it into different actions: jump or nothing (for the moment).

The bot uses opencv to interact with the Chrome browser and capture screenshots of the game. It then preprocesses the images and feeds them into the EfficientNet model for prediction. Based on the predicted action, the bot sends the appropriate keyboard command to the game.

## Acknowledgements

This project was inspired by [akshayballal95](https://github.com/akshayballal95/dino). I borrowed the basic structure and some ideas from their code repository. Many thanks to the contributors of InspirationBot for their valuable work.


