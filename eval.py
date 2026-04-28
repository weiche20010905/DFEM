"""Compare student vs. teacher on CIFAR10 test set."""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import resnet18

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


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
    student.eval()
    teacher.eval()
    match = total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        match += (student(x).argmax(1) == teacher(x).argmax(1)).sum().item()
        total += x.size(0)
    return match / total


def load(model, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--teacher-ckpt', default='./checkpoints/teacher_resnet18_cifar10.pth')
    parser.add_argument('--student-ckpt', default='./checkpoints/student_resnet18_extracted.pth')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

    teacher = resnet18(num_classes=10).to(device)
    student = resnet18(num_classes=10).to(device)
    load(teacher, args.teacher_ckpt, device)
    load(student, args.student_ckpt, device)

    t_acc = evaluate(teacher, loader, device)
    s_acc = evaluate(student, loader, device)
    agree = agreement(student, teacher, loader, device)

    print(f'Teacher CIFAR10 test acc: {t_acc*100:.2f}%')
    print(f'Student CIFAR10 test acc: {s_acc*100:.2f}%')
    print(f'Student↔Teacher agreement: {agree*100:.2f}%')
    print(f'Fidelity gap (teacher - student): {(t_acc - s_acc)*100:.2f} pp')


if __name__ == '__main__':
    main()
