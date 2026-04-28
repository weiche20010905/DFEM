"""Data-free model extraction.

Teacher: CIFAR10-pretrained ResNet18 (white-box outputs only).
Query data: CIFAR100 (disjoint classes from CIFAR10 — no CIFAR10 samples used).
Student: same architecture, trained on KD loss against teacher's soft labels.
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import resnet18

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar100_query_loader(data_root, batch_size, num_workers, train=True):
    """CIFAR100 surrogate. Same shape/stats as CIFAR10 inputs; disjoint label space."""
    if train:
        tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    ds = datasets.CIFAR100(data_root, train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=True, drop_last=train)


def get_cifar_test_loader(data_root, batch_size, num_workers):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = datasets.CIFAR10(data_root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


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


@torch.no_grad()
def agreement(student, teacher, loader, device):
    """Fraction of inputs where student top-1 matches teacher top-1."""
    student.eval()
    teacher.eval()
    match = total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        s_pred = student(x).argmax(1)
        t_pred = teacher(x).argmax(1)
        match += (s_pred == t_pred).sum().item()
        total += x.size(0)
    return match / total


def kd_loss(student_logits, teacher_logits, temperature):
    T = temperature
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean',
    ) * (T * T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--teacher-ckpt', default='./checkpoints/teacher_resnet18_cifar10.pth')
    parser.add_argument('--student-ckpt', default='./checkpoints/student_resnet18_cifar100query.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.student_ckpt), exist_ok=True)
    device = torch.device(args.device)

    teacher = resnet18(num_classes=10).to(device)
    state = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(state['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = resnet18(num_classes=10).to(device)

    query_loader = get_cifar100_query_loader(args.data_root, args.batch_size, args.num_workers, train=True)
    cifar_test_loader = get_cifar_test_loader(args.data_root, 512, args.num_workers)

    teacher_acc_on_cifar = evaluate(teacher, cifar_test_loader, device)
    print(f'Teacher CIFAR10 test acc: {teacher_acc_on_cifar*100:.2f}%')

    optimizer = optim.SGD(student.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay,
                          nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_agree = 0.0
    for epoch in range(args.epochs):
        student.train()
        t0 = time.time()
        running_loss, running_count = 0.0, 0
        for x, _ in query_loader:
            x = x.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = kd_loss(s_logits, t_logits, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_count += x.size(0)
        scheduler.step()

        avg_loss = running_loss / running_count
        s_acc = evaluate(student, cifar_test_loader, device)
        agree = agreement(student, teacher, cifar_test_loader, device)
        print(f'epoch {epoch+1:3d}/{args.epochs} | '
              f'lr {optimizer.param_groups[0]["lr"]:.4f} | '
              f'kd_loss {avg_loss:.4f} | '
              f'student_cifar_acc {s_acc*100:.2f} | '
              f'agreement_with_teacher {agree*100:.2f} | '
              f'{time.time()-t0:.1f}s')

        if agree > best_agree:
            best_agree = agree
            torch.save({'model_state_dict': student.state_dict(),
                        'cifar_test_acc': s_acc,
                        'agreement': agree,
                        'epoch': epoch + 1,
                        'temperature': args.temperature}, args.student_ckpt)

    print(f'Best agreement: {best_agree*100:.2f}%')
    print(f'Saved student to {args.student_ckpt}')


if __name__ == '__main__':
    main()
