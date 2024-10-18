import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt
import numpy 
from sklearn.model_selection import train_test_split


# Simple ANN
class IrisModel(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super(IrisModel, self).__init__()

        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Fetch the data
def get_data(url):
    my_df = pd.read_csv(url)
    return my_df

# change last colum from string to integers
def str_to_int(my_df):
    my_df["variety"] = my_df["variety"].replace("Setosa", 0)
    my_df['variety'] = my_df['variety'].replace('Versicolor', 1)
    my_df['variety'] = my_df['variety'].replace('Virginica', 2)
    return my_df

# Train Test Split!  Set X, y
def train_test(my_df):
    # Split X, y
    X = my_df.drop('variety', axis=1)
    y = my_df['variety']

    # Convert these to numpy arrays
    X = X.values
    y = y.values

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

    # Convert X features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # Convert y labels to tensors long
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    return X_train, X_test, y_train, y_test 


if __name__ == "__main__":
    torch.manual_seed(41)
    model = IrisModel()
    print(f"The Iris Model: {model}")
    print(f"The Iris Model: {model.fc1}")

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
df = get_data(url=url)
df = str_to_int(df)
print(df.head())

X_train, X_test, y_train, y_test = train_test(df)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Set the criterion of model
criterion = nn.CrossEntropyLoss()

# Choose Adam Optimizer, lr = learning rate 
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100
losses = []

for epoch in range(epochs):
    # Go forward and get a prediction
    y_pred = model.forward(X_train) 
    
    # Measure the loss/error
    loss = criterion(y_pred, y_train) 
   
    losses.append(loss.detach().numpy())  # Keep Track of our losses

    # Calculate accuracy
    with torch.no_grad():  # No need to calculate gradients for accuracy
        predictions = torch.argmax(y_pred, dim=1)  # Get the index of max log-probability
        correct = (predictions == y_train).sum().item()  # Count correct predictions
        accuracy = correct / y_train.size(0)  # Calculate accuracy

    # Print every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%")

    # Do backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Graph it out
plt.plot(range(epochs), losses)
plt.ylabel("loss/ERROR")
plt.xlabel("Epochs")
plt.show()