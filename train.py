"""CORTEX training entrypoint — ONE hand-written loop (no Trainer/Lightning).

Stage 1: trains the dense baseline on the needle task and reports retrieval
accuracy vs. chance. Writes run.json provenance (§3). Prints a PASS/FAIL banner.

    python train.py --config configs/dense_baseline.yaml

CLI overrides let the SAME code run at a smaller scale for quick verification
on CPU-only boxes without forking a config (ablations stay config-only, §2 r5):
    python train.py --config ... --max-steps 400 --batch-size 16 --seq-len 128
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

from data.synthetic import (CorefSpec, NeedleSpec, make_batch, make_coref_batch,
                            retrieval_accuracy)
from data.language import LanguageDataset
from eval.calibration import calibration_metrics
from eval.faithfulness import faithfulness_coref
from eval.flops import dense_flops_from_cfg
from eval.length_gen import length_gen_eval
from modern_nlp_architectire.dyn_patch import patch_stats
from modern_nlp_architectire.losses import (calibration_loss, lm_loss, mtp_loss, ponder_cost)
from modern_nlp_architectire.modeling import CortexForCausalLM
from modern_nlp_architectire.configuration import CortexConfig
from utils import (banner, get_device, get_logger, load_config, set_seed,
                   write_run_json)

log = get_logger("train")


needle_loss = lm_loss


def build_task(task: str, mcfg: Dict[str, Any], dcfg: Dict[str, Any]):
    """Return (spec, make_fn, chance). make_fn(bs, spec, gen, device)->batch."""
    if task == "needle":
        spec = NeedleSpec(vocab_size=mcfg["vocab_size"],
                          n_values=dcfg.get("n_values", 32),
                          seq_len=dcfg["seq_len"])
        return spec, make_batch, spec.chance
    if task == "coref":
        spec = CorefSpec(vocab_size=mcfg["vocab_size"],
                         n_keys=dcfg.get("n_keys", 16),
                         n_values=dcfg.get("n_values", 32),
                         n_pairs=dcfg.get("n_pairs", 4),
                         seq_len=dcfg["seq_len"])
        return spec, make_coref_batch, spec.chance
    raise ValueError(f"unknown task: {task}")


def lr_at(step: int, base_lr: float, warmup: int, total: int) -> float:
    """Linear warmup then cosine decay to 0."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if step >= total:
        return 0.0
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, make_fn, spec, n_batches: int, batch_size: int,
             device, seed: int = 12345) -> Dict[str, Any]:
    """Retrieval accuracy + loss (+ controller g_t stats) on fixed-seed eval data."""
    model.eval()
    g = torch.Generator().manual_seed(seed)  
    accs, losses = [], []
    g_hard, g_easy, g_all = [], [], []       
    pstats = []                               
    psteps_hard, psteps_easy = [], []        
    moe_imb, moe_k, loads = [], [], []       
    for _ in range(n_batches):
        b = make_fn(batch_size, spec, g, device=device)
        logits = model(b.x)                              
        if hasattr(b, 'answer_pos'):
            accs.append(retrieval_accuracy(logits, b))
        else:
            mask = b.difficulty > 0
            if mask.any():
                top5_preds = logits.topk(5, dim=-1).indices
                correct_in_top5 = (top5_preds[mask] == b.y[mask].unsqueeze(-1)).any(dim=-1)
                accs.append(correct_in_top5.float().mean().item())
            else:
                accs.append(0.0)
        losses.append(needle_loss(logits, b).item())
        if model.controller is not None:
            gt = model.last_signal.g                     
            hard = b.difficulty > 0                       
            g_hard.append(gt[hard])
            g_easy.append(gt[~hard])
            g_all.append(gt.reshape(-1))
        if model.patcher is not None:
            pstats.append(patch_stats(model.last_patch, b.difficulty, b.x.shape[1]))
        if model.latent_ponder is not None:
            es = model.last_ponder.expected_steps               
            hard = b.difficulty > 0
            psteps_hard.append(es[hard]); psteps_easy.append(es[~hard])
        if model.blocks[0].is_moe:
            imbs = [blk.mlp.last_info.imbalance for blk in model.blocks]
            moe_imb.append(sum(imbs) / len(imbs))
            moe_k.append(model.blocks[0].mlp.last_info.mean_k)
            loads.append(torch.stack([blk.mlp.last_info.load for blk in model.blocks]).mean(0))
    model.train()
    out: Dict[str, Any] = {"answer_acc": sum(accs) / len(accs),
                           "eval_loss": sum(losses) / len(losses)}
    if g_all:
        allg = torch.cat(g_all)
        hist = torch.histc(allg, bins=10, min=0.0, max=1.0)  
        out["g_mean"] = allg.mean().item()
        out["g_std"] = allg.std().item()
        out["g_hard_mean"] = torch.cat(g_hard).mean().item()
        out["g_easy_mean"] = torch.cat(g_easy).mean().item()
        out["g_hist"] = (hist / hist.sum()).tolist()
    if pstats:
        keys = pstats[0].keys()
        for k in keys:
            out[f"patch_{k}"] = sum(p[k] for p in pstats) / len(pstats)
    if psteps_hard:
        hh, ee = torch.cat(psteps_hard), torch.cat(psteps_easy)
        alls = torch.cat([hh, ee])
        out["ponder_hard"] = hh.mean().item()
        out["ponder_easy"] = ee.mean().item()
        out["ponder_mean"] = alls.mean().item()
        out["ponder_min"] = alls.min().item()
        out["ponder_max"] = alls.max().item()
    if moe_imb:
        out["moe_imbalance"] = sum(moe_imb) / len(moe_imb)
        out["moe_mean_k"] = sum(moe_k) / len(moe_k)
        out["moe_load"] = (torch.stack(loads).mean(0)).tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="CORTEX trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint.pt if it exists")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="output dir for run.json")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--controller", dest="controller", action="store_true",
                    default=None, help="force-enable the (inert) controller")
    ap.add_argument("--no-controller", dest="controller", action="store_false",
                    help="force-disable the controller")
    ap.add_argument("--pos", choices=["learned", "nope"], default=None,
                    help="override positional scheme (#2)")
    ap.add_argument("--dyn-patch", dest="dyn_patch", action="store_true",
                    default=None, help="force-enable dynamic patching (#3)")
    ap.add_argument("--no-dyn-patch", dest="dyn_patch", action="store_false",
                    help="force-disable dynamic patching")
    ap.add_argument("--patch-aux", type=float, default=None,
                    help="boundary-supervision weight for dynamic patching")
    ap.add_argument("--task", choices=["needle", "coref"], default=None)
    ap.add_argument("--attn-budget", dest="attn_budget", action="store_true",
                    default=None, help="force-enable budgeted attention (#1)")
    ap.add_argument("--no-attn-budget", dest="attn_budget", action="store_false")
    ap.add_argument("--structured-residual", dest="structured_residual",
                    action="store_true", default=None, help="layerscale residual (#7)")
    ap.add_argument("--budget-cost", type=float, default=None,
                    help="compute cost lambda on mean g_t (pushes span down)")
    ap.add_argument("--latent-ponder", dest="latent_ponder", action="store_true",
                    default=None, help="enable adaptive-depth latent loop (#5+#9)")
    ap.add_argument("--no-latent-ponder", dest="latent_ponder", action="store_false")
    ap.add_argument("--ponder-kl", type=float, default=None,
                    help="weight on PonderNet KL-to-geometric prior")
    ap.add_argument("--moe", dest="moe", action="store_true", default=None,
                    help="enable difficulty-scaled MoE FFN (#4)")
    ap.add_argument("--no-moe", dest="moe", action="store_false")
    ap.add_argument("--memory", dest="memory", action="store_true", default=None,
                    help="enable g-gated KV memory (#6)")
    ap.add_argument("--no-memory", dest="memory", action="store_false")
    ap.add_argument("--moe-experts", type=int, default=None)
    ap.add_argument("--moe-expert-ff", type=int, default=None)
    ap.add_argument("--mtp", dest="mtp", action="store_true", default=None,
                    help="enable multi-token prediction head (#8)")
    ap.add_argument("--no-mtp", dest="mtp", action="store_false")
    ap.add_argument("--mtp-weight", type=float, default=None)
    ap.add_argument("--cal-weight", type=float, default=None,
                    help="calibration loss weight (align g_t with NLL, #10)")
    ap.add_argument("--ponder-cost", type=float, default=None,
                    help="lambda_c: compute cost on E[ponder steps] (frontier knob)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    
    mcfg: Dict[str, Any] = dict(cfg["model"])
    ocfg: Dict[str, Any] = dict(cfg["optim"])
    dcfg: Dict[str, Any] = dict(cfg["data"])
    ecfg: Dict[str, Any] = dict(cfg.get("eval", {"interval": 100, "batches": 8}))

    if args.max_steps is not None:
        ocfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        dcfg["batch_size"] = args.batch_size
    if args.seq_len is not None:
        dcfg["seq_len"] = args.seq_len

    device = torch.device("cpu") if args.cpu else get_device()
    task = cfg.get("data", {}).get("task", "needle")
    lang_dataset = None
    if task == "language":
        
        ds_name = cfg.get("data", {}).get("dataset_name", "roneneldan/TinyStories")
        log.info(f"Setting up real language streaming from {ds_name}...")
        lang_dataset = iter(LanguageDataset(ds_name, seq_len=dcfg["seq_len"], batch_size=dcfg["batch_size"], device=device))
        
        
        chance = 1.0 / 100257 
        make_fn = lambda bs, spec, gen, device: next(lang_dataset)
        spec = None
        
    else:
        spec, make_fn, chance = build_task(task, mcfg, dcfg)

    
    subs = cfg.get("subsystems", {})
    enable_controller = (bool(subs.get("controller", False))
                         if args.controller is None else args.controller)
    pos = ("nope" if subs.get("nope", False) else mcfg.get("pos", "learned"))
    if args.pos is not None:
        pos = args.pos
    dyn_patch = (bool(subs.get("dyn_patch", False))
                 if args.dyn_patch is None else args.dyn_patch)
    attn_budget = (bool(subs.get("attn_budget", False))
                   if args.attn_budget is None else args.attn_budget)
    structured_res = (bool(subs.get("structured_residual", False))
                      if args.structured_residual is None else args.structured_residual)
    residual = "layerscale" if structured_res else "vanilla"
    pcfg = dict(cfg.get("patch", {}))
    for k in ("patch_hidden", "patch_ctx_weight", "patch_use_signal"):
        if k in pcfg:
            mcfg[k] = pcfg[k]
    patch_aux = (args.patch_aux if args.patch_aux is not None
                 else float(pcfg.get("aux_weight", 0.0)))
    budget_cost = (args.budget_cost if args.budget_cost is not None
                   else float(cfg.get("budget", {}).get("cost", 0.0)))
    
    latent_ponder = (bool(subs.get("halting", False) or subs.get("latent_loop", False))
                     if args.latent_ponder is None else args.latent_ponder)
    poncfg = dict(cfg.get("ponder", {}))
    for k in ("ponder_max_steps", "ponder_min_steps", "ponder_prior_lambda",
              "ponder_g_bias"):
        if k in poncfg:
            mcfg[k] = poncfg[k]
    ponder_kl = (args.ponder_kl if args.ponder_kl is not None
                 else float(poncfg.get("kl_weight", 0.01)))
    moe = (bool(subs.get("moe", False)) if args.moe is None else args.moe)
    memory = (bool(subs.get("memory", False)) if args.memory is None else args.memory)
    moecfg = dict(cfg.get("moe", {}))
    for k in ("moe_experts", "moe_k_max", "moe_expert_ff"):
        if k in moecfg:
            mcfg[k] = moecfg[k]
    if args.moe_experts is not None:
        mcfg["moe_experts"] = args.moe_experts
    if args.moe_expert_ff is not None:
        mcfg["moe_expert_ff"] = args.moe_expert_ff
    if "mem_slots" in cfg.get("mem", {}):
        mcfg["mem_slots"] = cfg["mem"]["mem_slots"]
    lb_weight = float(moecfg.get("lb_weight", 0.01))
    
    objcfg = dict(cfg.get("objective", {}))
    mtp = (bool(objcfg.get("mtp", False)) if args.mtp is None else args.mtp)
    mtp_weight = (args.mtp_weight if args.mtp_weight is not None
                  else float(objcfg.get("mtp_weight", 0.5)))
    cal_weight = (args.cal_weight if args.cal_weight is not None
                  else float(objcfg.get("cal_weight", 0.0)))
    lambda_c = (args.ponder_cost if args.ponder_cost is not None
                else float(objcfg.get("ponder_cost", 0.0)))

    
    cortex_cfg = {
        "model": {
            "pos": pos,
            "span_min": mcfg.get("span_min", 4),
            "span_temp": mcfg.get("span_temp", 1.0),
            "residual": residual,
            "moe_experts": mcfg.get("moe_experts", 4),
            "moe_k_max": mcfg.get("moe_k_max", 2),
            "moe_expert_ff": mcfg.get("moe_expert_ff", 0),
            "ponder_max_steps": mcfg.get("ponder_max_steps", 4),
            "ponder_min_steps": mcfg.get("ponder_min_steps", 1),
            "mem_slots": mcfg.get("mem_slots", 32),
        },
        "subsystems": {
            "controller": enable_controller,
            "dyn_patch": dyn_patch,
            "attn_budget": attn_budget,
            "latent_loop": latent_ponder,
            "moe": moe,
            "memory": memory,
        },
        "patch": {
            "patch_hidden": mcfg.get("patch_hidden", 32),
            "patch_ctx_weight": mcfg.get("patch_ctx_weight", 0.5),
        },
        "objective": {
            "mtp": mtp,
        }
    }
    
    hf_config = CortexConfig(
        vocab_size=mcfg["vocab_size"],
        d_model=mcfg["d_model"],
        n_layers=mcfg["n_layers"],
        n_heads=mcfg["n_heads"],
        d_ff=mcfg["d_ff"],
        max_seq_len=mcfg["max_seq_len"],
        dropout=mcfg.get("dropout", 0.0),
        norm=mcfg.get("norm", "pre"),
        tie_embeddings=mcfg.get("tie_embeddings", True),
        cortex_cfg=cortex_cfg
    )
    
    model = CortexForCausalLM(hf_config).to(device)
    
    ckpt_path = Path("results") / cfg.get("run_name", "run") / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        log.info(f"Resuming from checkpoint {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
    n_params = model.num_params()
    flops = dense_flops_from_cfg(mcfg, dcfg["seq_len"])
    log.info("run=%s device=%s params=%.2fM control_mode=%s task=%s | controller=%s "
             "pos=%s dyn_patch=%s attn_budget=%s residual=%s | patch_aux=%.2f "
             "budget_cost=%.3f",
             cfg.get("run_name", "?"), device, n_params / 1e6,
             cfg.get("control_mode"), task, enable_controller, pos, dyn_patch,
             attn_budget, residual, patch_aux, budget_cost)
    log.info("chance=%.4f train/seq=%.3f GFLOP", chance, flops.per_seq_train / 1e9)

    opt = torch.optim.AdamW(model.parameters(), lr=ocfg["lr"],
                            weight_decay=ocfg["weight_decay"], betas=(0.9, 0.95))

    torch.autograd.set_detect_anomaly(True)

    total_steps = ocfg["max_steps"]
    warmup = ocfg.get("warmup_steps", 0)
    grad_clip = ocfg.get("grad_clip", 1.0)
    use_amp = (device.type == "cuda") and (ocfg.get("precision", "bf16") == "bf16")
    gpu_delay = float(ocfg.get("gpu_delay", 0.0))

    train_gen = torch.Generator().manual_seed(seed + 1)  
    history = []
    t0 = time.time()
    model.train()
    for step in range(total_steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step, ocfg["lr"], warmup, total_steps)

        b = make_fn(dcfg["batch_size"], spec, train_gen, device=device)
        
        if (b.y == -100).all():
            print(f"[{step}] WARNING: Batch entirely ignored. Skipping step.")
            continue
            
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(b.x, gt_difficulty=b.difficulty, patch_aux_weight=patch_aux)
            loss = lm_loss(logits, b)                          
            if model.mtp_head is not None and mtp_weight > 0.0:
                loss = loss + mtp_weight * mtp_loss(model.last_mtp, b)  
            if model.controller is not None and cal_weight > 0.0:
                
                loss = loss + cal_weight * calibration_loss(
                    logits, model.last_signal.logits[..., 0], b)
            if model.latent_ponder is not None and lambda_c > 0.0:
                
                loss = loss + lambda_c * ponder_cost(model.last_ponder.expected_steps)
            if model.patcher is not None and model.last_patch.aux_loss is not None:
                loss = loss + model.last_patch.aux_loss   
            if model.controller is not None and budget_cost > 0.0:
                loss = loss + budget_cost * model.last_signal.g.mean()  
            if model.latent_ponder is not None and ponder_kl > 0.0:
                
                loss = loss + ponder_kl * model.last_ponder.kl.mean()   
            if model.last_moe_aux is not None and lb_weight > 0.0:
                loss = loss + lb_weight * model.last_moe_aux   
        opt.zero_grad(set_to_none=True)
        if loss.isnan():
            print(f"[{step}] WARNING: NaN loss detected. Skipping step.")
            opt.zero_grad(set_to_none=True)
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"[{step}] WARNING: Invalid grad norm {grad_norm}. Skipping step.")
            opt.zero_grad(set_to_none=True)
            continue
            
        opt.step()

        if gpu_delay > 0.0 and device.type == "cuda":
            time.sleep(gpu_delay)

        if (step + 1) % ecfg["interval"] == 0 or step == 0:
            m = evaluate(model, make_fn, spec, ecfg["batches"], dcfg["batch_size"], device)
            sps = (step + 1) / (time.time() - t0)
            gtxt = (f" | g_t {m['g_mean']:.3f}±{m['g_std']:.3f} "
                    f"(hard {m['g_hard_mean']:.3f}/easy {m['g_easy_mean']:.3f})"
                    if "g_mean" in m else "")
            ptxt = (f" | patch_len {m['patch_avg_patch_len']:.2f} "
                    f"(bnd hard {m['patch_p_boundary_hard']:.2f}/"
                    f"easy {m['patch_p_boundary_easy']:.2f})"
                    if "patch_avg_patch_len" in m else "")
            dtxt = (f" | ponder hard {m['ponder_hard']:.2f}/easy {m['ponder_easy']:.2f} "
                    f"[{m['ponder_min']:.1f},{m['ponder_max']:.1f}]"
                    if "ponder_mean" in m else "")
            mtxt = (f" | moe imbal {m['moe_imbalance']:.2f} k {m['moe_mean_k']:.2f}"
                    if "moe_imbalance" in m else "")
            log.info("step %5d | loss %.4f | eval_loss %.4f | answer_acc %.3f "
                     "(chance %.3f) | %.2f it/s%s%s%s%s",
                     step + 1, loss.item(), m["eval_loss"], m["answer_acc"],
                     chance, sps, gtxt, ptxt, dtxt, mtxt)
            history.append({"step": step + 1, "train_loss": loss.item(), **m})
            
            # Save checkpoint
            if not ckpt_path.parent.exists():
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = ckpt_path.with_suffix('.tmp.pt')
            torch.save(model.state_dict(), tmp_path)
            tmp_path.replace(ckpt_path)
            log.info(f"Saved checkpoint to {ckpt_path}")

    final = evaluate(model, make_fn, spec, max(ecfg["batches"], 16),
                     dcfg["batch_size"], device)

    
    lengen = None
    if task == "needle":
        L = dcfg["seq_len"]
        lengen = length_gen_eval(model, mcfg["vocab_size"], dcfg.get("n_values", 32),
                                 [L, 2 * L, 3 * L, 4 * L], dcfg["batch_size"],
                                 device, n_batches=8)
        log.info("length-gen (%s): %s", pos, {k: round(v, 3) for k, v in lengen.items()})

    
    faith = None
    if task == "coref" and model.controller is not None:
        faith = faithfulness_coref(model, spec, dcfg["batch_size"], device, n_batches=8)
        log.info("faithfulness: g_hard=%.3f g_easy=%.3f sep=%.3f auroc=%.3f passed=%s",
                 faith["g_hard_mean"], faith["g_easy_mean"], faith["separation"],
                 faith["auroc"], faith["passed"])

    
    calib = None
    if model.controller is not None:
        calib = calibration_metrics(model, make_fn, spec, dcfg["batch_size"], device)
        log.info("calibration: corr(g,NLL)=%.3f ece=%.3f mean_g=%.3f mean_err=%.3f",
                 calib["corr_g_nll"], calib["ece"], calib["mean_g"], calib["mean_err"])
    wall = time.time() - t0

    
    beats_chance = final["answer_acc"] > max(5 * chance, 0.25)
    
    passed = beats_chance and (faith is None or faith["passed"])

    metrics = {
        "task": task,
        "final_answer_acc": final["answer_acc"],
        "final_eval_loss": final["eval_loss"],
        "chance": chance,
        "beats_chance": beats_chance,
        "passed": passed,
        "controller_enabled": enable_controller,
        "pos": pos,
        "dyn_patch": dyn_patch,
        "attn_budget": attn_budget,
        "residual": residual,
        "budget_cost": budget_cost,
        "n_params": n_params,
        "train_flops_per_seq": flops.per_seq_train,
        "wall_seconds": wall,
        "steps": total_steps,
        "history": history,
    }
    
    metrics["objective"] = {"mtp": mtp, "mtp_weight": mtp_weight,
                            "cal_weight": cal_weight, "lambda_c": lambda_c}
    metrics["compute"] = {
        "mean_g": final.get("g_mean"),
        "ponder_mean": final.get("ponder_mean"),
        "moe_mean_k": final.get("moe_mean_k"),
    }
    if lengen is not None:
        metrics["length_gen"] = {str(k): v for k, v in lengen.items()}
    if faith is not None:
        metrics["faithfulness"] = faith
    if calib is not None:
        metrics["calibration"] = calib
    if "g_mean" in final:
        metrics["controller"] = {
            "g_mean": final["g_mean"], "g_std": final["g_std"],
            "g_hard_mean": final["g_hard_mean"], "g_easy_mean": final["g_easy_mean"],
            "g_hist_10bins": final["g_hist"],
        }
    if "patch_avg_patch_len" in final:
        metrics["patch"] = {
            "avg_patch_len": final["patch_avg_patch_len"],
            "p_boundary_hard": final["patch_p_boundary_hard"],
            "p_boundary_easy": final["patch_p_boundary_easy"],
            "mean_num_patches": final["patch_mean_num_patches"],
        }
    if "ponder_mean" in final:
        
        max_steps = mcfg.get("ponder_max_steps", 4)
        min_steps = mcfg.get("ponder_min_steps", 1)
        no_collapse = (min_steps + 0.05 < final["ponder_mean"] < max_steps - 0.05)
        metrics["ponder"] = {
            "hard": final["ponder_hard"], "easy": final["ponder_easy"],
            "mean": final["ponder_mean"], "min": final["ponder_min"],
            "max": final["ponder_max"], "max_steps": max_steps,
            "min_steps": min_steps, "no_collapse": bool(no_collapse),
            "hard_gt_easy": bool(final["ponder_hard"] > final["ponder_easy"]),
        }
    if "moe_imbalance" in final:
        
        metrics["moe"] = {
            "imbalance": final["moe_imbalance"], "mean_k": final["moe_mean_k"],
            "load": final["moe_load"], "n_experts": mcfg.get("moe_experts", 4),
            "balanced": bool(final["moe_imbalance"] < 1.5),
        }
    if memory and "g_mean" in final:
        
        metrics["memory"] = {
            "read_gate_hard": final["g_hard_mean"],
            "read_gate_easy": final["g_easy_mean"],
            "fires_on_hard": bool(final["g_hard_mean"] > final["g_easy_mean"]),
            "n_slots": mcfg.get("mem_slots", 32),
        }
    out_dir = Path(args.out) if args.out else Path("results") / cfg.get("run_name", "run")
    run_path = write_run_json(out_dir / "run.json",
                              config={**cfg, "_overrides": {
                                  "task": task, "max_steps": ocfg["max_steps"],
                                  "batch_size": dcfg["batch_size"],
                                  "seq_len": dcfg["seq_len"]}},
                              seed=seed, metrics=metrics)
    log.info("wrote %s", run_path)
    ftxt = (f" | faithfulness sep={faith['separation']:.3f} auroc={faith['auroc']:.3f}"
            if faith is not None else "")
    banner(passed, f"CORTEX run ({cfg.get('run_name','run')}, task={task})",
           f"answer_acc={final['answer_acc']:.3f} vs chance={chance:.3f}"
           f"{ftxt} | attn_budget={attn_budget} | {total_steps} steps | {wall:.0f}s")


if __name__ == "__main__":
    main()
