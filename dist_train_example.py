import os
import torch
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.optim.lr_scheduler as lr_scheduler

# Define a simple dataset class
class RandomDataset(Dataset):
    def __init__(self, num_samples=1000, input_size=10):
        self.num_samples = num_samples
        self.inputs = torch.randn(num_samples, input_size)
        self.labels = torch.randint(0, 2, (num_samples,))  # Binary classification

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Training function
def train():
    # Get environment variables from torchrun
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set device
    torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create model and move to GPU
    model = SimpleModel(input_size=10, output_size=2).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Create dataset and dataloader
    dataset = RandomDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Training loop
    for epoch in range(10):
        sampler.set_epoch(epoch)  # Ensure different shuffling per epoch
        model.train()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(local_rank), labels.to(local_rank)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Step the scheduler
        scheduler.step()

        # Print loss and learning rate from rank 0
        if rank == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch+1}/10, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    # Save the model only from rank 0
    if rank == 0:
        save_path = '/mnt/pvc-blob-fuse-out/yuhang/checkpoint.pth'
        if not os.path.exists('/mnt/pvc-blob-fuse-out/yuhang/'):
            os.makedirs('/mnt/pvc-blob-fuse-out/yuhang/')
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    # Cleanup
    if rank == 0:
        print('Training Done!')
    dist.destroy_process_group()

if __name__ == "__main__":
    train()