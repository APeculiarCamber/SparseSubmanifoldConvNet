import torch
import wandb
from sp_modelnet_dataset import ModelNetDataset
from torch.utils.data import DataLoader
from sp_resnet import ResNet, SpResNet
import time
from sparse_conv3d_atomic import SparseConv3d_Atomic
from sparse_conv3d_inplace import SparseConv3d_InPlace
from torch.optim import lr_scheduler

def make_dataloaders(batch_size, pin_memory=False):
    test_dataset = ModelNetDataset("ModelNet10", train=False)
    train_dataset = ModelNetDataset("ModelNet10", train=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=pin_memory)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=pin_memory)

    return train_loader, test_loader

def adjust_input(input : torch.Tensor, grid_size):
    if input.shape[2] > grid_size:
        bs = input.shape[2] // grid_size
        gs = grid_size
        input = input.reshape(-1, gs, bs, gs, bs, gs, bs).permute(0, 1, 3, 5, 2, 4, 6).reshape(-1,gs,gs,gs,bs*bs*bs).contiguous()
        input = torch.any(input, dim=-1)
    input = input.unsqueeze(1)
    return input.to(torch.float)

def DEBUG_print_grads(model):
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Gradient: \n{param.grad}")

def train(model, criterion, optimizer, train_loader, grid_size, device):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = adjust_input(inputs, grid_size)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        assert not loss.isnan()

        loss.backward()
        # DEBUG_print_grads(model)
        optimizer.step()

        running_loss += loss.item()

        inputs, labels, loss = None, None, None
        torch.cuda.empty_cache()

    return running_loss / len(train_loader)

def test(model, criterion, test_loader, grid_size, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = adjust_input(inputs, grid_size)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    return accuracy, avg_loss

def run_main(config, model):
    batch_size = config['batch_size']
    lr = config['lr']
    grid_size = config['grid_size']
    check_step = config['step']
    step_gamma = config['step_gamma']

    train_loader, test_loader = make_dataloaders(batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stepper = lr_scheduler.StepLR(optimizer, check_step, step_gamma)

    best_accuracy = 0.0
    for epoch in range(config['epochs']):
        st_time = time.time()
        train_loss = train(model, criterion, optimizer, train_loader, grid_size, device)
        train_time = time.time() - st_time
        torch.cuda.empty_cache()
        st_time = time.time()
        accuracy, test_loss = test(model, criterion, test_loader, grid_size, device)
        test_time = time.time() - st_time
        torch.cuda.empty_cache()
        
        device = torch.cuda.current_device()
        
        # Log metrics using WandB
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'train_time': train_time,
            'test_time': test_time,
        })
        print({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'train_time': train_time,
            'test_time': test_time,
        })

        # Checkpoint the model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            checkpoint_path = f"checkpoint_epoch_best_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)

        stepper.step()
        
def run_std_resnet(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet([2, 2, 2, 2], config['grid_size'], 16, num_classes=10).to(device)
    run_main(config, model)

def run_std_resnet(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet([2, 2, 2, 2], config['grid_size'], 16, num_classes=10).to(device)
    run_main(config, model)

import argparse
if __name__ == "__main__":
    # Configurations for your training
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--step", type=int, default=6)
    parser.add_argument("--step_gamma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--model_type", type=str, default="dense", help="dense,inplace,atomic")
    parser.add_argument("--model_size", type=int, default=16)
    args = parser.parse_args()
    
    config = vars(args)
    
    if True:
        wandb.login()
        wandb.init(project='submanifold', name="MODEL_TYPE_"+str(args.model_type)+"__GRID_SIZE_"+str(args.grid_size)+"__MODEL_SIZE_"+str(args.model_size), config=config)
        wandb.config.update(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model_type == "dense":
        model = ResNet([2, 2, 2, 2], config['grid_size'], args.model_size, num_classes=10).to(device)
    if args.model_type == "inplace":
        model = SpResNet([2, 2, 2, 2], config['grid_size'], args.model_size, ConvOp=SparseConv3d_InPlace, num_classes=10).to(device)
    if args.model_type == "atomic":
        model = SpResNet([2, 2, 2, 2], config['grid_size'], args.model_size, ConvOp=SparseConv3d_Atomic, num_classes=10).to(device)

    run_main(config, model)
