"""
Phase 1: Unit Cell Training with Hard Programmatic Acceptance Gates.
Trains at fixed T=8, checks T=1 against the empirical local ceiling, and evaluates a separate test split.
"""
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.core import LatentReasoner
from data.graph_task import generate_parity_benchmark

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

def evaluate(model, loader, steps):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_A, _, b_A, _, _ in loader:
            logits, _ = model(x_A, steps=steps)
            preds = logits.argmax(dim=-1)
            correct += (preds == b_A).sum().item()
            total += b_A.size(0)
    return correct / total

def train_phase1():
    set_seed(42)
    print("Generating datasets...")
    train_ds, train_hashes = generate_parity_benchmark(num_samples=4000, seed=42)
    val_ds, val_hashes = generate_parity_benchmark(num_samples=1000, seed=1337, existing_hashes=train_hashes)
    test_ds, _ = generate_parity_benchmark(num_samples=1000, seed=9999, existing_hashes=train_hashes.union(val_hashes))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = LatentReasoner(num_nodes=16, d_input=32, d_model=128)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print("Training LatentReasoner at fixed recurrent depth T=8...")
    for epoch in range(1, 26):
        model.train()
        for x_A, _, b_A, _, _ in train_loader:
            optimizer.zero_grad()
            logits, _ = model(x_A, steps=8)
            loss = criterion(logits, b_A)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            acc_8 = evaluate(model, val_loader, steps=8)
            print(f"Epoch {epoch:02d} | Val Acc (T=8): {acc_8*100:.1f}%")

    print("\n--- Phase 1 Acceptance Gate Evaluation ---")
    acc_1 = evaluate(model, val_loader, steps=1)
    acc_8 = evaluate(model, val_loader, steps=8)
    print(f"Gate 1: T=1 Accuracy = {acc_1*100:.1f}% (Required: <= 56.5%)")
    print(f"Gate 2: T=8 Accuracy = {acc_8*100:.1f}% (Required: >= 90.0%)")

    if acc_1 > 0.565:
        print(f"FAIL: Gate 1 missed ({acc_1*100:.1f}% > 56.5%)!", file=sys.stderr)
        sys.exit(1)
    if acc_8 < 0.90:
        print(f"FAIL: Gate 2 missed ({acc_8*100:.1f}% < 90.0%)!", file=sys.stderr)
        sys.exit(1)

    acc_test_8 = evaluate(model, test_loader, steps=8)
    print(f"SUCCESS: Phase 1 validation gates passed. Held-out Test Acc (T=8): {acc_test_8*100:.1f}%")
    torch.save(model.state_dict(), "checkpoint_phase1.pt")

if __name__ == "__main__":
    train_phase1()
