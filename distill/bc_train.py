#!/usr/bin/env python3
"""Train the clone and write it as an rsl_rl checkpoint.

Fits the actor mean net (45 -> 128 -> 128 -> 64 -> 12, ELU -- the same
architecture PPO trains) by MSE on the collector's (obs45, act45) pairs, then
writes ``model_0.pt`` in the schema ``OnPolicyRunner.load(strict=True)``
expects: the trained actor (with the exploration std set to the PPO cfg's
init_std 0.8 -- cloning never touches it), a fresh critic, a fresh Adam
state over the 17 parameters in PPO's construction order, iter 0.

    python distill/bc_train.py --dataset runs/distill/pairs.npz --output runs/distill/bc_clone.pt --epochs 40 --augment

``--augment`` adds uniform input jitter at the clean-observation noise floor
so the clone learns to recover from its own small deviations; the shipped
clone used it. Pure torch, a few seconds on a GPU. Validation MSE is not a
useful gate on its own -- evaluate the clone closed-loop (phase2/run_distill.sh).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HID = [128, 128, 64]
JITTER = (((0, 3), 0.05), ((3, 6), 0.02), ((9, 21), 0.01), ((21, 33), 0.2), ((33, 45), 0.05))


def mlp(inp: int, out: int) -> nn.Sequential:
    layers, prev = [], inp
    for h in HID:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers += [nn.Linear(prev, out)]
    return nn.Sequential(*layers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", required=True, help="path to write the checkpoint (e.g. bc_clone.pt)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--augment", action="store_true", help="uniform input jitter at the clean noise floor")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    z = np.load(args.dataset)
    obs = torch.as_tensor(z["obs45"], dtype=torch.float32)
    act = torch.as_tensor(z["act45"], dtype=torch.float32)
    assert obs.shape[1] == 45 and act.shape[1] == 12
    n = obs.shape[0]
    perm = torch.randperm(n)
    n_val = max(n // 10, 1)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    actor = mlp(45, 12).to(args.device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    obs_d, act_d = obs.to(args.device), act.to(args.device)

    for epoch in range(args.epochs):
        actor.train()
        ep_perm = tr_idx[torch.randperm(tr_idx.shape[0])]
        losses = []
        for i in range(0, ep_perm.shape[0], args.batch):
            idx = ep_perm[i : i + args.batch]
            x = obs_d[idx]
            if args.augment:
                x = x.clone()
                for (a, b), amp in JITTER:
                    x[:, a:b] += (torch.rand_like(x[:, a:b]) * 2 - 1) * amp
            pred = actor(x)
            loss = nn.functional.mse_loss(pred, act_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        actor.eval()
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(actor(obs_d[val_idx]), act_d[val_idx]).item()
        print(f"epoch {epoch:3d} train {np.mean(losses):.6f} val {val_mse:.6f}")

    # rsl_rl checkpoint schema
    actor_cpu = actor.cpu()
    actor_sd = {f"mlp.{k}": v for k, v in actor_cpu.state_dict().items()}
    actor_sd["distribution.std_param"] = torch.full((12,), 0.8)
    critic = mlp(45, 1)
    critic_sd = {f"mlp.{k}": v for k, v in critic.state_dict().items()}

    std_param = nn.Parameter(actor_sd["distribution.std_param"].clone())
    all_params = list(actor_cpu.parameters()) + [std_param] + list(critic.parameters())
    assert len(all_params) == 17, len(all_params)
    fresh_opt = torch.optim.Adam(all_params, lr=1e-3)
    ckpt = {
        "actor_state_dict": actor_sd,
        "critic_state_dict": critic_sd,
        "optimizer_state_dict": fresh_opt.state_dict(),
        "iter": 0,
        "infos": None,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    meta = {
        "dataset": str(Path(args.dataset).resolve()),
        "rows": int(n),
        "epochs": args.epochs,
        "augment": bool(args.augment),
        "final_val_mse": float(val_mse),
        "seed": args.seed,
        "std_param": 0.8,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("BC_TRAIN_META " + json.dumps(meta))
    print("BC_TRAIN: PASS")


if __name__ == "__main__":
    main()
