import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Training function
def train():
    # Initialize distributed environment variables (set by torchrun)
    rank = int(os.environ["RANK"])            # Global rank of the process
    world_size = int(os.environ["WORLD_SIZE"])  # Total number of processes
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank within the current node

    print('rank = {}, world_size = {}, local_rank = {}'.format(rank, world_size, local_rank))

    # Set device
    torch.cuda.set_device(local_rank)

    # Initialize process group
    print('initializing the dist process group')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create model and move to GPU
    model = SimpleModel(input_size=10, output_size=2).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Learning rate scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Create dummy dataset
    dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Training loop
    print('start training loop!')
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
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    train()