"""
Phase 2a: 1-Bit Latent Swarm with Stratified Acceptance Gates.
Evaluates all four (b_A, b_B) quadrants, stratified message-flip and probe recovery.
Reports validation acceptance and held-out test results.
"""
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.swarm import LatentSwarm
from data.graph_task import generate_parity_benchmark

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)

def evaluate_swarm_stratified(swarm, loader, **kwargs):
    swarm.eval()
    quad_correct = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    quad_total = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    msg_correct = {0: 0, 1: 0}
    msg_total = {0: 0, 1: 0}
    total_correct = 0
    total = 0
    total_msg_correct = 0

    with torch.no_grad():
        for x_A, x_B, b_A, b_B, Y in loader:
            p_logits, _, _, _, m = swarm(x_A, x_B, steps_A=8, steps_B=8, **kwargs)
            preds = p_logits.argmax(dim=-1)
            total_correct += (preds == Y).sum().item()
            total += Y.size(0)

            for ba, bb, p, y in zip(b_A.tolist(), b_B.tolist(), preds.tolist(), Y.tolist()):
                quad_total[(ba, bb)] += 1
                if p == y:
                    quad_correct[(ba, bb)] += 1

            m_bits = (m.squeeze(-1) > 0).long().tolist()
            for ba, mb in zip(b_A.tolist(), m_bits):
                msg_total[ba] += 1
                if mb == ba:
                    msg_correct[ba] += 1
                    total_msg_correct += 1

    quad_accs = {q: quad_correct[q] / max(quad_total[q], 1) for q in quad_correct}
    probe_accs = {b: msg_correct[b] / max(msg_total[b], 1) for b in msg_correct}
    overall_probe = total_msg_correct / max(total, 1)
    return total_correct / max(total, 1), quad_accs, probe_accs, overall_probe

def evaluate_flip_stratified(swarm, loader):
    """Fraction of initially correct predictions remaining correct after m is flipped, by b_B."""
    swarm.eval()
    flip_correct = {0: 0, 1: 0}
    base_correct = {0: 0, 1: 0}
    with torch.no_grad():
        for x_A, x_B, _, b_B, Y in loader:
            p_base, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8)
            p_flip, _, _, _, _ = swarm(x_A, x_B, steps_A=8, steps_B=8, flip_message=True)
            base_preds = p_base.argmax(dim=-1)
            flip_preds = p_flip.argmax(dim=-1)
            for bb, bp, fp, y in zip(b_B.tolist(), base_preds.tolist(), flip_preds.tolist(), Y.tolist()):
                if bp == y:
                    base_correct[bb] += 1
                    if fp == y:
                        flip_correct[bb] += 1
    return {bb: flip_correct[bb] / max(base_correct[bb], 1) for bb in (0, 1)}

def _print_stratified(prefix, overall, quads, flip, zero, probes, probe_overall):
    print(f"{prefix} Connected Accuracy: {overall*100:.2f}%")
    for q in ((0, 0), (0, 1), (1, 0), (1, 1)):
        print(f"{prefix} Quadrant {q} Accuracy: {quads[q]*100:.2f}%")
    print(f"{prefix} Flip residual b_B=0: {flip[0]*100:.2f}%")
    print(f"{prefix} Flip residual b_B=1: {flip[1]*100:.2f}%")
    print(f"{prefix} Zero-message accuracy: {zero*100:.2f}%")
    print(f"{prefix} Probe overall: {probe_overall*100:.2f}%")
    print(f"{prefix} Probe b_A=0: {probes[0]*100:.2f}%")
    print(f"{prefix} Probe b_A=1: {probes[1]*100:.2f}%")

def train_phase2():
    set_seed(42)
    train_ds, train_hashes = generate_parity_benchmark(num_samples=4000, seed=42)
    val_ds, val_hashes = generate_parity_benchmark(num_samples=1000, seed=1337, existing_hashes=train_hashes)
    test_ds, _ = generate_parity_benchmark(num_samples=1000, seed=9999, existing_hashes=train_hashes.union(val_hashes))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    swarm = LatentSwarm(num_nodes=16, d_input=32, d_model=128)
    try:
        # State dict contains tensors only; all computation is CPU-only.
        p1_state = torch.load("checkpoint_phase1.pt", map_location="cpu", weights_only=True)
        swarm.core_A.load_state_dict(p1_state)
        swarm.core_B.load_state_dict(p1_state)
        print("Warm-started Core A and Core B from checkpoint_phase1.pt")
    except FileNotFoundError:
        print("Error: checkpoint_phase1.pt not found! Run train_unit.py first.", file=sys.stderr)
        sys.exit(1)

    optimizer = torch.optim.AdamW(swarm.parameters(), lr=8e-4, weight_decay=1e-2)
    crit_parity = nn.CrossEntropyLoss()
    crit_aux = nn.CrossEntropyLoss()
    crit_msg = nn.BCEWithLogitsLoss()

    print("Training Phase 2a at T_A=8, T_B=8...")
    for epoch in range(1, 21):
        swarm.train()
        for x_A, x_B, b_A, b_B, Y in train_loader:
            optimizer.zero_grad()
            p_logits, raw_m, aux_A, aux_B, _ = swarm(x_A, x_B, steps_A=8, steps_B=8)
            l_parity = crit_parity(p_logits, Y)
            l_aux_A = crit_aux(aux_A, b_A)
            l_aux_B = crit_aux(aux_B, b_B)
            l_msg = crit_msg(raw_m.squeeze(-1), b_A.float())
            loss = l_parity + 0.5 * l_aux_A + 0.5 * l_aux_B + 0.5 * l_msg
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            acc, _, _, _ = evaluate_swarm_stratified(swarm, val_loader)
            print(f"Epoch {epoch:02d} | Validation parity accuracy: {acc*100:.2f}%")

    print("\n--- Phase 2a Validation Acceptance Evaluation ---")
    acc_connected, quads, probes, probe_overall = evaluate_swarm_stratified(swarm, val_loader)
    acc_zero, _, _, _ = evaluate_swarm_stratified(swarm, val_loader, zero_message=True)
    flip_strat = evaluate_flip_stratified(swarm, val_loader)
    _print_stratified("Validation", acc_connected, quads, flip_strat, acc_zero, probes, probe_overall)

    # Hard validation gates; do not relax these to force a pass.
    failed = []
    if acc_connected < 0.82:
        failed.append("overall connected accuracy < 82%")
    if any(q_acc < 0.75 for q_acc in quads.values()):
        failed.append("one or more quadrant accuracies < 75%")
    if flip_strat[0] > 0.15 or flip_strat[1] > 0.15:
        failed.append("stratified message-flip residual > 15%")
    if acc_zero > 0.53:
        failed.append("zero-message accuracy > 53%")
    if probe_overall < 0.95 or probes[0] < 0.90 or probes[1] < 0.90:
        failed.append("overall or stratified message probe below acceptance threshold")
    if failed:
        for reason in failed:
            print(f"FAIL: {reason}", file=sys.stderr)
        sys.exit(1)

    print("SUCCESS: All Phase 2a validation gates passed.")

    # Final held-out test evaluation; report all metrics but do not tune on test data.
    print("\n--- Final Held-Out Test Set Evaluation ---")
    t_acc, t_quads, t_probes, t_probe_overall = evaluate_swarm_stratified(swarm, test_loader)
    t_zero, _, _, _ = evaluate_swarm_stratified(swarm, test_loader, zero_message=True)
    t_flip = evaluate_flip_stratified(swarm, test_loader)
    _print_stratified("Test", t_acc, t_quads, t_flip, t_zero, t_probes, t_probe_overall)

if __name__ == "__main__":
    train_phase2()
