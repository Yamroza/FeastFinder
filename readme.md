# State Dialog Manager for Restaurant Suggestions

## Overview

The State Dialog Manager is a conversational AI system designed to assist users in finding restaurant suggestions based on their preferences. The system employs a state machine architecture to guide the conversation and facilitate user interactions. 

## Getting Started

### Prerequisites

To run the system, you need to have the following installed:

- Python 3.x
- Required libraries (see requirements.txt)

### Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/restaurant-dialog-manager.git
    cd restaurant-dialog-manager
    ```

2. Install the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure your settings:
    - Open the `config.ini` file to adjust configurable parameters such as delay, style, capital output, and restart options according to your preferences.

### Training the Models

The current version of the Dialog Manager works with the feedforward neural network machine learning model, using the BoW approach. However, it is possible to switch to another model. This can be done by using model_training.py and specifying which model, with what kind of preprocessing you would like to use. The possible combinations are: FN-BoW, LR-BoW, FFN-Doc2Vec and LR-Doc2Vec.

### Running the Dialog Manager

Once the models have been trained, you can start the dialog manager by running the `main.py` script:

```bash
python main.py
