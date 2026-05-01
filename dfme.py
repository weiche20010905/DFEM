"""Black-box Data-Free Model Extraction (DFME, Truong et al. 2021).

Threat model: teacher only exposes a query API returning logits/soft labels.
The attacker cannot backprop through the teacher.

Pipeline:
  - Generator G(z) produces synthetic 3x32x32 inputs.
  - Student S is trained by KD against teacher's responses on G(z) (white-box for S).
  - Generator is trained adversarially to maximize student-teacher disagreement.
  - Generator gradient w.r.t. its inputs is estimated via forward differences
    over `num_directions` random Gaussian directions (no teacher backprop needed).
"""
import argparse
import gc
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import Generator, resnet18


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_cifar_test_loader(data_root, batch_size, num_workers):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    ds = datasets.CIFAR10(data_root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def discrepancy(s_logits, t_logits):
    """Per-sample L1 distance between softmax outputs."""
    s = F.softmax(s_logits, dim=1)
    t = F.softmax(t_logits, dim=1)
    return F.l1_loss(s, t, reduction='none').sum(dim=1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def agreement(student, teacher, loader, device):
    student.eval(); teacher.eval()
    match = total = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        match += (student(x).argmax(1) == teacher(x).argmax(1)).sum().item()
        total += x.size(0)
    return match / total


def estimate_x_grad(student, teacher, x, eps, num_directions):
    """Forward-difference estimate of grad_x of -mean(discrepancy).

    Black-box w.r.t. teacher: only forward queries on x and x+eps*u.
    Returns a tensor of the same shape as x.
    """
    x_d = x.detach()
    with torch.no_grad():
        t0 = teacher(x_d)
        s0 = student(x_d)
        L0 = -discrepancy(s0, t0)  # [B]

        grad = torch.zeros_like(x_d)
        for _ in range(num_directions):
            u = torch.randn_like(x_d)
            x_pert = x_d + eps * u
            t_p = teacher(x_pert)
            s_p = student(x_pert)
            L_p = -discrepancy(s_p, t_p)
            grad = grad + ((L_p - L0) / eps).view(-1, 1, 1, 1) * u
        grad = grad / num_directions
    return grad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--teacher-ckpt', default='./checkpoints/teacher_resnet18_cifar10.pth')
    parser.add_argument('--student-ckpt', default='./checkpoints/student_resnet18_dfme.pth')
    parser.add_argument('--generator-ckpt', default='./checkpoints/generator_dfme.pth')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--iters-per-epoch', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--n-g', type=int, default=1, help='generator inner steps per iter')
    parser.add_argument('--n-s', type=int, default=5, help='student inner steps per iter')
    parser.add_argument('--num-directions', type=int, default=1, help='# random directions for grad estimation')
    parser.add_argument('--eps', type=float, default=1e-3, help='forward-difference epsilon')
    parser.add_argument('--lr-s', type=float, default=0.1)
    parser.add_argument('--lr-g', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--num-workers', type=int, default=0, help='set to 0 to avoid DataLoader deadlock with long-running DataParallel')
    parser.add_argument('--gpus', default='0', help='comma-separated GPU ids, e.g. "0,1,2,3"')
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.student_ckpt), exist_ok=True)
    device_ids = [int(g) for g in args.gpus.split(',')]
    device = torch.device(f'cuda:{device_ids[0]}')

    teacher = resnet18(num_classes=10).to(device)
    state = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher.load_state_dict(state['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = resnet18(num_classes=10).to(device)
    G = Generator(nz=args.nz).to(device)

    if len(device_ids) > 1:
        teacher = nn.DataParallel(teacher, device_ids=device_ids)
        student = nn.DataParallel(student, device_ids=device_ids)
        G = nn.DataParallel(G, device_ids=device_ids)
        print(f'Using DataParallel across GPUs {device_ids}')

    test_loader = get_cifar_test_loader(args.data_root, 512, num_workers=2)
    teacher_acc = evaluate(teacher, test_loader, device)
    print(f'Teacher CIFAR10 test acc: {teacher_acc*100:.2f}%')

    opt_S = optim.SGD(student.parameters(), lr=args.lr_s,
                      momentum=args.momentum, weight_decay=args.weight_decay,
                      nesterov=True)
    opt_G = optim.Adam(G.parameters(), lr=args.lr_g)
    sched_S = optim.lr_scheduler.CosineAnnealingLR(opt_S, T_max=args.epochs)
    sched_G = optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.epochs)

    queries_per_iter = (1 + args.num_directions) * args.n_g + args.n_s
    print(f'Teacher queries per iter (per batch): {queries_per_iter} '
          f'(n_g={args.n_g}, n_s={args.n_s}, directions={args.num_directions})')
    total_queries = args.epochs * args.iters_per_epoch * queries_per_iter * args.batch_size
    print(f'Total query budget: ~{total_queries/1e6:.1f}M samples')

    best_agree = 0.0
    for epoch in range(args.epochs):
        student.train()
        G.train()
        t0 = time.time()
        running_g_loss = running_s_loss = 0.0
        running_g_n = running_s_n = 0

        for it in range(args.iters_per_epoch):
            for _ in range(args.n_g):
                z = torch.randn(args.batch_size, args.nz, device=device)
                x = G(z)
                grad_x = estimate_x_grad(student, teacher, x, args.eps, args.num_directions)
                opt_G.zero_grad()
                x.backward(grad_x / args.batch_size)
                opt_G.step()
                with torch.no_grad():
                    g_loss = -discrepancy(student(x.detach()), teacher(x.detach())).mean().item()
                running_g_loss += g_loss
                running_g_n += 1

            for _ in range(args.n_s):
                with torch.no_grad():
                    z = torch.randn(args.batch_size, args.nz, device=device)
                    x = G(z).detach()
                    t_log = teacher(x)
                s_log = student(x)
                loss = discrepancy(s_log, t_log).mean()
                opt_S.zero_grad()
                loss.backward()
                opt_S.step()
                running_s_loss += loss.item()
                running_s_n += 1

            if (it + 1) % args.log_every == 0:
                print(f'  ep{epoch+1} iter {it+1}/{args.iters_per_epoch} | '
                      f'g_loss {running_g_loss/max(running_g_n,1):.4f} | '
                      f's_loss {running_s_loss/max(running_s_n,1):.4f}',
                      flush=True)
                torch.cuda.empty_cache()

        sched_S.step(); sched_G.step()
        gc.collect()
        torch.cuda.empty_cache()

        s_acc = evaluate(student, test_loader, device)
        agree = agreement(student, teacher, test_loader, device)
        elapsed = time.time() - t0
        print(f'epoch {epoch+1:3d}/{args.epochs} | '
              f'lr_s {opt_S.param_groups[0]["lr"]:.4f} lr_g {opt_G.param_groups[0]["lr"]:.5f} | '
              f'g_loss {running_g_loss/max(running_g_n,1):.4f} '
              f's_loss {running_s_loss/max(running_s_n,1):.4f} | '
              f'student_cifar_acc {s_acc*100:.2f} agreement {agree*100:.2f} | {elapsed:.1f}s')

        latest_student = args.student_ckpt.replace('.pth', '_latest.pth')
        latest_generator = args.generator_ckpt.replace('.pth', '_latest.pth')
        torch.save({'model_state_dict': unwrap(student).state_dict(),
                    'cifar_test_acc': s_acc,
                    'agreement': agree,
                    'epoch': epoch + 1}, latest_student)
        torch.save({'model_state_dict': unwrap(G).state_dict(),
                    'nz': args.nz,
                    'epoch': epoch + 1}, latest_generator)

        if agree > best_agree:
            best_agree = agree
            torch.save({'model_state_dict': unwrap(student).state_dict(),
                        'cifar_test_acc': s_acc,
                        'agreement': agree,
                        'epoch': epoch + 1}, args.student_ckpt)
            torch.save({'model_state_dict': unwrap(G).state_dict(),
                        'nz': args.nz,
                        'epoch': epoch + 1}, args.generator_ckpt)

    print(f'Best agreement: {best_agree*100:.2f}%')
    print(f'Saved student to {args.student_ckpt}')
    print(f'Saved generator to {args.generator_ckpt}')


if __name__ == '__main__':
    main()
