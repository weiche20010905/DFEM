import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import resnet18

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_loaders(data_root, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--ckpt', default='./checkpoints/teacher_resnet18_cifar10.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    device = torch.device(args.device)

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)
    model = resnet18(num_classes=10).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss, running_correct, running_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_total += y.size(0)

        scheduler.step()
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        test_acc = evaluate(model, test_loader, device)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(),
                        'test_acc': test_acc,
                        'epoch': epoch + 1}, args.ckpt)

        print(f'epoch {epoch+1:3d}/{args.epochs} | '
              f'lr {optimizer.param_groups[0]["lr"]:.4f} | '
              f'train_loss {train_loss:.4f} train_acc {train_acc*100:.2f} | '
              f'test_acc {test_acc*100:.2f} (best {best_acc*100:.2f}) | '
              f'{time.time()-t0:.1f}s')

    print(f'Best teacher test accuracy: {best_acc*100:.2f}%')
    print(f'Saved to {args.ckpt}')


if __name__ == '__main__':
    main()
