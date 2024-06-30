
# Marketing Guru Chatbot

An AI-Powered Marketing Assistant for Churn Prediction

## Overview

The Marketing Guru Chatbot is designed to help businesses predict customer churn and interact with users through a chatbot interface. This repository contains scripts for preprocessing data, training a churn prediction model, testing the model, deploying an API, and creating a chatbot.

## Directory Structure

- `data/` - Contains the dataset used for training and testing.
- `model/` - Stores the trained model, label encoders, and scalers.
- `utils/` - Utility functions used throughout the project.
- `0-preprocess.py` - Script for preprocessing the data.
- `1-train.py` - Script for training the churn prediction model.
- `2-test.py` - Script for testing the trained model.
- `3-api.py` - FastAPI script to create an API for the churn prediction model.
- `4-chatbot.py` - Script to deploy a chatbot that uses the churn prediction API.
- `requirements.txt` - List of dependencies required for the project.

## Getting Started

### Prerequisites

Ensure you have Python installed on your system. You can install the required dependencies using:

\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Data Preprocessing

Before training the model, preprocess the data using:

\`\`\`bash
python 0-preprocess.py
\`\`\`

This script will clean the data and prepare it for model training.

### Training the Model

Train the churn prediction model with:

\`\`\`bash
python 1-train.py
\`\`\`

This script will train the model and save it along with label encoders and scalers in the `model/` directory.

### Testing the Model

Test the performance of the trained model using:

\`\`\`bash
python 2-test.py
\`\`\`

This script will evaluate the model on test data and print the performance metrics.

### Deploying the API

Deploy the model as an API with FastAPI:

\`\`\`bash
python 3-api.py
\`\`\`

The API will be accessible at `http://127.0.0.1:8000`.

### Running the Chatbot

Launch the chatbot interface using:

\`\`\`bash
python 4-chatbot.py
\`\`\`

The chatbot will interact with users and predict churn based on the model.

## Contributing

Contributions are welcome. Please fork the repository and create a pull request.

## License

This project is licensed under the MIT License.
