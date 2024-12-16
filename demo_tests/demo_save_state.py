import torch
from torch import nn, optim
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
import os

from accelerate.utils import ProjectConfiguration
import torch

# Define the ProjectConfiguration
config = ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=1000)

# Initialize the Accelerator with the configuration
accelerator = Accelerator(project_config=config, project_dir='./training_checkpoints')


# Define the project directory where the state will be saved
# project_dir = "./training_checkpoints"
# os.makedirs(project_dir, exist_ok=True)

# # Initialize the Accelerator with the project directory specified
# accelerator = Accelerator(project_dir=project_dir)

# Create a simple linear regression model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Generate some random data for training
X_train = torch.randn(100, 1)
y_train = 3 * X_train + torch.randn(100, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = LinearModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Prepare everything with the accelerator
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Unpack the batch
        x_batch, y_batch = batch
        
        # Forward pass
        predictions = model(x_batch)
        loss = loss_fn(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss at the end of each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Save the state after each epoch to the specified project directory
    accelerator.save_state()
    # print(f"State saved to {project_dir}")
