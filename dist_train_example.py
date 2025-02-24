import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
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
def train(local_rank, world_size, master_addr, master_port):
    # Set master address & port
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # Retrieve global rank and local rank
    rank = int(os.environ["RANK"])  # Global rank
    torch.cuda.set_device(local_rank)  # Assign correct GPU based on local rank

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create model and move to GPU
    model = SimpleModel(input_size=10, output_size=2).to(local_rank)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)  

    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Add LR Scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Create dummy dataset
    dataset = TensorDataset(torch.randn(1000, 10), torch.randint(0, 2, (1000,)))
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
        torch.save(model.module.state_dict(), save_path)
        print(f"Model saved at {save_path}")

    # Cleanup
    dist.destroy_process_group()

# Entry point
def main():
    parser = argparse.ArgumentParser(description="Distributed Training Example")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address")
    parser.add_argument("--master_port", type=str, default="12355", help="Master node port")
    args = parser.parse_args()

    world_size = int(os.environ["WORLD_SIZE"])  # Get world size from environment
    local_rank = int(os.environ["LOCAL_RANK"])  # Local rank within the node

    # Spawn processes on each GPU of the node
    mp.spawn(train, args=(world_size, args.master_addr, args.master_port), nprocs=torch.cuda.device_count(), join=True)

if __name__ == "__main__":
    main()