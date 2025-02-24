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
def train(rank, world_size, epochs=10, batch_size=32, save_path="checkpoint.pth"):
    # Set master address & port (important for multi-node training)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")  # Ensure this port is open

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create model and move to GPU
    model = SimpleModel(input_size=10, output_size=2).to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)  # Avoid unused param issues

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Add LR Scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Decay LR every 5 epochs

    # Create dummy dataset
    dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Training loop
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # Ensures different shuffling per epoch
        model.train()

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(rank), labels.to(rank)

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
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")

    # Save the model only from rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    # Cleanup
    dist.destroy_process_group()

# Entry point
def main():
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))  # Default to available GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()