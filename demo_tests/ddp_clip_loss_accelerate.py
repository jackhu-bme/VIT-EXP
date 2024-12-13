# implement the clip loss for ddp in two ways, compare the two implementations, this script is for the second implementation by accelerate
# this should be tested on two nvidia gpus

import torch
import random

# set random seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)

# create data tensors
x = torch.rand(8, 2) # stands for 8 samples with 5 features, such as 8 images, and for each gpu it has 4 images
y = torch.rand(8, 2) # texts for the images, 4 texts for each gpu

# create model
linear_model = torch.nn.Linear(2, 2)

# torch dataset
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


custom_dataset = CustomDataset(x, y)
custom_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=False)

# clip loss implemented by torch

from clip_loss_acc import ClipLossAcc







from accelerate import Accelerator

accelerator = Accelerator()

# acc_model = accelerator.prepare_model(linear_model)

optim = torch.optim.SGD(linear_model.parameters(), lr=0.001)


custom_dataloader, acc_optim, acc_model = accelerator.prepare(custom_dataloader, optim, linear_model)

# acc_optim = accelerator.prepare_optimizer(optim)

# scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=1.0)

# acc_scheduler = accelerator.prepare_scheduler(scheduler)

# clip loss implemented by accelerate


# loss_fn = nn.MSELoss()

loss_fn = ClipLossAcc(smoothing=0.)

print(f"before , linear model: {linear_model}")

acc_optim.zero_grad()

print(f"x on process id: {accelerator.process_index} is {x}")

outputs = acc_model(x)

print(f"output on process id: {accelerator.process_index} is {outputs}")

print(f"y on process id: {accelerator.process_index} is {y}")

loss = loss_fn(outputs, y, accelerator)

print(f"loss on process id: {accelerator.process_index} is {loss}")

accelerator.backward(loss)

acc_optim.step()

print(f"after , linear model: {linear_model}")



# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         outputs = model(**batch)
#         loss = outputs.loss
#         accelerator.backward(loss)

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)



# usage
# accelerate launch demo_tests/ddp_clip_loss_accelerate.py
