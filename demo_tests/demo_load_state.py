import torch
from torch import nn, optim
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
import os

from accelerate.utils import ProjectConfiguration
import torch

def find_latest_save_iteration(project_dir):
    # the ckpt dir have many dirs: checkpoints/checkpoint_0/1/2/... find the biggest number
    # return the biggest number
    names = os.listdir(os.path.join(project_dir, "checkpoints"))
    names = [name for name in names if name.startswith("checkpoint_")]
    return max([int(name.split("_")[-1]) for name in names])

project_dir = "./training_checkpoints/with_step"

try:
    latest_save_iter = find_latest_save_iteration(project_dir)
except:
    latest_save_iter = 0

# Define the ProjectConfiguration
config = ProjectConfiguration(automatic_checkpoint_naming=True, total_limit=1000, iteration=latest_save_iter)



# Initialize the Accelerator with the configuration
accelerator = Accelerator(project_config=config, project_dir=project_dir)

print(f"accelerator project_dir: {accelerator.project_dir}")



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
        self.register_buffer("step", torch.tensor(0))

    def forward(self, x):
        return self.linear(x)

# Generate some random data for training
X_train = torch.randn(1000, 1)
y_train = 3 * X_train + torch.randn(1000, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Instantiate the model, loss function, and optimizer
model = LinearModel()
# optimizer = optim.SGD(model.parameters(), lr=0.01)
# adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Prepare everything with the accelerator
model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

print(f"1 accelerator save iteration: {accelerator.save_iteration}")


# load the state

# Load the state from the specified project directory
# accelerator.load_state()



print(f"latest_save_iter: {latest_save_iter}")

accelerator.project_configuration.iteration += (latest_save_iter + 1)

print(f"2 accelerator save iteration: {accelerator.save_iteration}")

# train_loader = accelerator.skip_first_batches(train_loader, model.step)

print(f"3 accelerator save iteration: {accelerator.save_iteration}")

print(f"State loaded from {project_dir}")

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # Unpack the batch
        x_batch, y_batch = batch
        
        # Forward pass
        predictions = model(x_batch)
        print(f"Step: {model.step}")
        model.step += 1
        loss = loss_fn(predictions, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"4 accelerator save iteration: {accelerator.save_iteration}")

    # Print loss at the end of each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    # Save the state after each epoch to the specified project directory
    accelerator.save_state()
    # print(f"State saved to {project_dir}")
