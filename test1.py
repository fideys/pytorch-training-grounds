# Import the main PyTorch library
import torch

# Import the neural network tools from PyTorch
import torch.nn as nn

# -------------------------------
# CREATE SOME FAKE DATA
# -------------------------------

# Create a tensor with random numbers
# Shape: 10 rows, 3 columns
# Think of this as: 10 data samples, each with 3 features
x = torch.randn(10, 3)

# Create target values (what the model should predict)
# Shape must match the output of the model later
y = torch.randn(10, 1)

print("input shape:", x.shape)  # Should print: torch.Size([10, 3])
print("target shape:", y.shape)  # Should print: torch.Size([10, 1])

# -------------------------------
# DEFINE THE MODEL
# -------------------------------

# nn.Linear creates a fully connected layer
# It takes:
#   - 3 input values
#   - outputs 1 value
#
# Internally, it creates:
#   weights (3 numbers)
#   bias (1 number)
#
# output = (input * weights) + bias
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

z = model(x)
# -------------------------------
# DEFINE LOSS FUNCTION
# -------------------------------

# Mean Squared Error:
# Measures how far predictions are from the real values
loss_fn = nn.MSELoss()


# -------------------------------
# DEFINE OPTIMIZER
# -------------------------------

# The optimizer updates the model's weights
# model.parameters() = all weights & biases inside the model
# lr = learning rate (how big each update step is)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# -------------------------------
# TRAINING LOOP
# -------------------------------

# Repeat training 100 times
for i in range(100):

    # 1. Forward pass
    # Feed input data through the model
    predictions = model(x)
    print("Predictions shape:", predictions.shape)  # Should print: torch.Size([10, 1])

    # 2. Calculate how wrong the predictions are
    loss = loss_fn(predictions, y)
    print("Loss shape:", loss.shape)  # Should print: torch.Size([])
    # 3. Clear old gradients
    # (PyTorch accumulates them by default)
    optimizer.zero_grad()

    # 4. Compute new gradients
    # This tells every weight how much it contributed to the error
    loss.backward()

    # 5. Update the weights using the gradients
    optimizer.step()

    # Print progress every 10 steps
    if i % 10 == 0:
        print("Step:", i, "Loss:", loss.item())
