# Image-Captioning-using-VGG16-LSTM-Attention-Layer

📌 Project Idea
In this project, I created an image captioning system that takes an image as input and generates a descriptive sentence as output. The model combines a pre-trained VGG16 CNN to extract image features, an LSTM network for sequence generation, and an attention mechanism to improve focus on important parts of the image during captioning.

🎯 Objective
Extract high-level image features using VGG16 convolutional neural network.

Use LSTM to generate captions word by word.

Integrate attention layer to help the model focus on relevant regions in the image.

Train the model on the Flickr8k dataset to learn meaningful captions.

🛠️ Technologies I Used
Python

Libraries: TensorFlow / Keras, numpy, matplotlib, nltk

Dataset: Flickr8k (images + captions)

Jupyter Notebook for development and experimentation

🗂️ Dataset
Flickr8k dataset with 8,000 images and five captions per image.

Images are used for feature extraction, while captions provide ground truth sentences.

🔄 My Workflow
Preprocessing Captions

Tokenized, cleaned, and padded text sequences.

Feature Extraction

Passed images through pre-trained VGG16, extracted feature vectors from the last convolutional layers.

Model Architecture

Combined extracted image features with LSTM sequence generator.

Added an attention mechanism to improve accuracy by allowing the model to focus dynamically on parts of the image while generating each word.

Training

Trained the model on image-caption pairs.

Used teacher forcing by providing previous words as input for next word prediction.

Inference

Given a new image, the model generates captions word by word until reaching an end token or max sequence length.

📊 Results
The model successfully generates human-like captions describing the contents of unseen images.

Adding attention improves the relevance of generated captions compared to baseline LSTM-only models.

