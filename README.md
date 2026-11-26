# Solving the SMS Classification Challenge
The challenge required us to build a machine learning model that classifies SMS messages as either "ham" or "spam". A "ham" message is a normal message sent by a friend, while a "spam" message is an advertisement or a message sent by a company. The task was to create a function called predict_message that takes a message string as an argument and returns a list where the first element is the probability that the message is "ham or spam" and the second element is the either the word "ham" or "spam"

# Overview of the Solution

My solution containing all my code and detailed notes was implemented using Google Colaboratory, a free cloud service that provides an environment to run Jupyter Notebooks. The first step was to import the necessary libraries such as TensorFlow, pandas, numpy, and matplotlib.

The next step was to load the SMS Spam Collection dataset. This dataset has already been split into training and test datasets. The data was read into pandas DataFrames for ease of processing and inspection.

# Data Cleaning and Preprocessing
During the data inspection phase, it was found that the training dataset had no missing values. The class columns in both training and testing datasets were converted from text ('spam' and 'ham') to numeric values (0 and 1) using one-hot encoding. The messages were then checked for length and plotted for a better understanding of the data distribution. Duplicate messages were dropped to avoid re-training on the same data.

# Model Building
The next phase was building the machine learning model. The chosen model was a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units. This model type is suitable for text classification tasks due to its ability to capture long-term dependencies in sequence data. The model included an embedding layer, two bidirectional LSTM layers, a dense layer, and a final output layer. The model was compiled with a binary cross-entropy loss function and Adam optimizer.

# Training the Model
The model was trained on the training data for 10 epochs. The training data was shuffled and batched for better training performance. At the end of each epoch, the model's performance was evaluated on the test data. The model achieved a validation accuracy of 98.91% by the 16th epoch, indicating a high level of performance in SMS message classification.

# Conclusion
In conclusion, the challenge of classifying SMS messages as "spam" or "ham" was successfully tackled using Google Colaboratory and TensorFlow. The solution utilized an RNN with LSTM units, which performed well in capturing the patterns in the sequence data inherent in text messages. The trained model achieved a high level of accuracy, demonstrating its effectiveness in this classification task.
