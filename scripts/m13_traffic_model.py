"""M13 analytic traffic model: per-buffer formulas vs measured counters.

The runnable model of record for the M13 backward decision (memo §4): it
validates the segmented-backward traffic formulas against the M13-T0 ncu
counters on the four T0 shapes (dev fp16, the 16x512 bf16 corner, one real
query record, one real doc record). Formulas are in (B, S, V, D, f, elt)
plus the selected config blocks; distribution statistics (destination-run
count, mixed-chunk fraction, active fraction) are computed from the
*actual inputs* — the same seeds and bundle records the profiled runs used
— never estimated.

The MEASURED tables below are the M13-T0 counters of record, transcribed
from the deposited ncu transcripts (M13 memo §2; session artifacts). The
CONFIG table pins the autotune selections those transcripts ran under —
if the autotuner's selections move (config-list edits re-key the caches),
re-profile and update both tables together: the model is only meaningful
against counters from the configs it assumes.

Real-record rows need the capture bundles (default
``tests/data/bundles/``, regenerable via
``scripts/capture_index_distributions.py`` — but regenerated bundles
contain different records, so the embedded measured counters then no
longer correspond; see ``tests/data/README.md``). Sector = 32 B. Output:
the expected-vs-measured table per buffer per shape and per-kernel time
floors with named anchors (memo §4.2).

Usage (CUDA required; bundle-dependent rows are skipped with a notice if
the bundles are absent):

    env $ENV PYTHONPATH=src python -u scripts/m13_traffic_model.py \
        [--bundle-dir tests/data/bundles]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from bench_backward import make_synthetic_case, make_real_case  # noqa: E402

SECT = 32.0
L2_ACHIEVABLE = 6.6e12     # B/s, M12-measured fabric rate at 89-91% busy
L2_CONSERVATIVE = 5.1e12   # ~70% of implied peak: conservative attainment
DRAM_ACHIEVABLE = 1.5e12   # B/s (prep kernel measured 84% of nameplate)

# Measured counters of record (M13-T0 ncu transcripts; memo §2).
# Sectors are counts; dram_* in MB; durations in microseconds (ncu regime —
# structure only, never compared with do_bench latencies).
MEASURED = {
    "dev_fp16": {
        "seg": dict(lts_read=48_861_752, red=3_486_240, dram_read=68.19, dram_write=0.0, dur_us=765.15),
        "emb": dict(lts_read=37_115_615, lts_write=2_933_943, l1_ld=49_269_888, dram_read=12.62, dram_write=53.19, dur_us=214.18),
    },
    "corner_bf16": {
        "seg": dict(lts_read=170_174_528, red=6_396_160, dram_read=372.38, dram_write=22.02, dur_us=2590.0),
        "emb": dict(lts_read=158_498_256, lts_write=19_466_812, l1_ld=164_692_224, dram_read=32.19, dram_write=583.86, dur_us=759.46),
    },
    "query_r0": {
        "seg": dict(lts_read=209_927_708, red=1_546_752, dram_read=750.82, dram_write=2.02, dur_us=2800.0),
        "emb": dict(lts_read=56_459_507, lts_write=24_031_459, l1_ld=157_306_992, dram_read=32.60, dram_write=726.72, dur_us=625.60),
    },
    "doc_r1": {
        "seg": dict(lts_read=201_263_299, red=12_460_800, dram_read=746.61, dram_write=8.21, dur_us=3310.0),
        "emb": dict(lts_read=147_305_819, lts_write=24_031_455, l1_ld=200_782_128, dram_read=36.34, dram_write=727.85, dur_us=894.78),
    },
}

# Autotune selections the measured counters ran under (M13 memo §2 tables).
CONFIG = {
    "dev_fp16": dict(seg_chunk=32, seg_bd=128, emb_bv=64, emb_bd=64),
    "corner_bf16": dict(seg_chunk=64, seg_bd=64, emb_bv=32, emb_bd=64),
    "query_r0": dict(seg_chunk=256, seg_bd=64, emb_bv=32, emb_bd=64),
    "doc_r1": dict(seg_chunk=32, seg_bd=128, emb_bv=64, emb_bd=64),
}

DIMS = {"dev_fp16": (768, 2), "corner_bf16": (1024, 2),
        "query_r0": (768, 2), "doc_r1": (768, 2)}


def case_stats(case, chunk):
    """Exact distribution stats the model consumes."""
    scores, idx, mask = case["max_scores"], case["max_idx"], case["mask"]
    B, V = scores.shape
    S = mask.shape[1]
    active = scores > 0
    f = active.float().mean().item()
    # destination keys of active entries, sorted -- exactly what the kernel sees
    b_index = torch.arange(B, device=scores.device).unsqueeze(1).expand(B, V)
    keys = torch.where(active, b_index * S + idx.to(torch.int64),
                       torch.tensor(B * S, device=scores.device))
    keys_sorted, _ = torch.sort(keys.flatten())
    total = B * V
    n_active = int(active.sum().item())
    live = keys_sorted[:n_active]
    runs = int((live[1:] != live[:-1]).sum().item()) + (1 if n_active else 0)
    n_chunks_live = (n_active + chunk - 1) // chunk
    starts = torch.arange(0, n_active, chunk, device=live.device)
    ends = torch.clamp(starts + chunk - 1, max=n_active - 1)
    mixed = int((live[starts] != live[ends]).sum().item())
    return dict(B=B, S=S, V=V, total=total, f=f, n_active=n_active,
                runs=runs, chunks_live=n_chunks_live, mixed_chunks=mixed)


def model_segmented(st, D, elt, chunk, bd):
    N, f = st["total"], st["f"]
    T_d = -(-D // bd)
    m = st["mixed_chunks"]
    gather = f * N * D * elt / SECT
    streams = 3 * (N * 4 / SECT) * T_d
    prevnext = 2 * (m * chunk * 4 / SECT) * T_d
    lts_read = gather + streams + prevnext
    red = (st["chunks_live"] + 2 * st["runs"]) * (D * 4 / SECT)
    table_mb = st["V"] * D * elt / 1e6
    streams_once_mb = 3 * N * 4 / 1e6
    return dict(lts_read=lts_read, red=red,
                dram_compulsory_mb=table_mb + streams_once_mb,
                T_d=T_d, table_mb=table_mb)


def model_embed(st, D, elt, bv, bd):
    N, f = st["total"], st["f"]
    T_d = -(-D // bd)
    streams = (N * 8 / SECT) * T_d              # g + idx, dense tile loads
    gather_req = f * N * D * elt / SECT          # hidden rows, requested at L1
    stores = st["V"] * D * 4 / SECT + st["V"] * 4 / SECT
    return dict(l1_ld=gather_req + streams, streams=streams,
                gather_req=gather_req, lts_write=stores,
                dram_write_mb=st["V"] * D * 4 / 1e6, T_d=T_d)


def time_floors(meas):
    l2_bytes = (meas["lts_read"] + meas.get("lts_write", 0)
                + meas.get("red", 0)) * SECT
    dram_bytes = (meas["dram_read"] + meas["dram_write"]) * 1e6
    return dict(
        t_l2_ms=l2_bytes / L2_ACHIEVABLE * 1e3,
        t_l2_cons_ms=l2_bytes / L2_CONSERVATIVE * 1e3,
        t_dram_ms=dram_bytes / DRAM_ACHIEVABLE * 1e3,
    )


def build_cases(bundle_dir: Path):
    out = {}
    out["dev_fp16"] = make_synthetic_case(
        source="uniform", batch_size=32, seq_len=128, dim=768, vocab=30522,
        density=0.75, active_fraction=1.0, zipf_s=1.1, dtype=torch.float16,
        bias_on=True, seed=0)
    out["corner_bf16"] = make_synthetic_case(
        source="uniform", batch_size=16, seq_len=512, dim=1024, vocab=151936,
        density=0.75, active_fraction=1.0, zipf_s=1.1, dtype=torch.bfloat16,
        bias_on=True, seed=0)
    steps0 = bundle_dir / "swimir_de_steps0.pt"
    steps150 = bundle_dir / "swimir_de_steps150.pt"
    for name, path, record in (("query_r0", steps0, 0), ("doc_r1", steps150, 1)):
        if path.exists():
            bundle = torch.load(path, weights_only=False)
            out[name] = make_real_case(bundle["records"][record],
                                       dtype=torch.float16, bias_on=True, seed=0)
        else:
            print(f"SKIP {name}: bundle missing ({path}); regenerate via "
                  f"scripts/capture_index_distributions.py — note the embedded "
                  f"measured counters correspond to the original records only",
                  flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=str,
                        default=str(REPO_ROOT / "tests" / "data" / "bundles"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("m13_traffic_model.py requires CUDA (case "
                           "construction matches the profiled inputs)")

    torch.manual_seed(0)
    cases = build_cases(Path(args.bundle_dir))
    print(f"{'shape':12s} {'kernel':5s} {'buffer':22s} {'model':>14s} "
          f"{'measured':>14s} {'resid%':>7s}")
    for name, case in cases.items():
        D, elt = DIMS[name]
        cfg = CONFIG[name]
        st = case_stats(case, cfg["seg_chunk"])
        ms = model_segmented(st, D, elt, cfg["seg_chunk"], cfg["seg_bd"])
        me = model_embed(st, D, elt, cfg["emb_bv"], cfg["emb_bd"])
        meas = MEASURED[name]
        rows = [
            ("seg", "lts_read sectors", ms["lts_read"], meas["seg"]["lts_read"]),
            ("seg", "red sectors", ms["red"], meas["seg"]["red"]),
            ("seg", "dram_read MB (compuls.)", ms["dram_compulsory_mb"], meas["seg"]["dram_read"]),
            ("emb", "l1 load sectors", me["l1_ld"], meas["emb"]["l1_ld"]),
            ("emb", "lts_write sectors", me["lts_write"], meas["emb"]["lts_write"]),
            ("emb", "dram_write MB", me["dram_write_mb"], meas["emb"]["dram_write"]),
        ]
        print(f"-- {name}: f={st['f']:.4f} runs={st['runs']} "
              f"chunks_live={st['chunks_live']} mixed={st['mixed_chunks']} "
              f"T_d(seg)={ms['T_d']} T_d(emb)={me['T_d']}")
        for kern, label, model, measured in rows:
            resid = (model - measured) / measured * 100 if measured else float("nan")
            print(f"{name:12s} {kern:5s} {label:22s} {model:14,.0f} "
                  f"{measured:14,.0f} {resid:6.1f}%")
        fs, fe = time_floors(meas["seg"]), time_floors(meas["emb"])
        kappa = meas["seg"]["dram_read"] / ms["table_mb"]
        print(f"{name:12s} seg   t_floor L2/cons/DRAM    "
              f"{fs['t_l2_ms']:.3f} / {fs['t_l2_cons_ms']:.3f} / "
              f"{fs['t_dram_ms']:.3f} ms   measured(ncu) "
              f"{meas['seg']['dur_us']/1e3:.3f} ms   kappa_dram={kappa:.2f}")
        print(f"{name:12s} emb   t_floor L2/cons/DRAM    "
              f"{fe['t_l2_ms']:.3f} / {fe['t_l2_cons_ms']:.3f} / "
              f"{fe['t_dram_ms']:.3f} ms   measured(ncu) "
              f"{meas['emb']['dur_us']/1e3:.3f} ms")
        alpha = 1 - (meas["emb"]["lts_read"] - me["streams"]) / me["gather_req"]
        print(f"{name:12s} emb   measured gather L1-absorption alpha = {alpha:.2f}")
    print("MODEL_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
