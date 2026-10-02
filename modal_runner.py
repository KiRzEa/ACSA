"""Modal runner: trains baseline + CAGE jobs on Modal GPU containers in parallel, writes results to a
Modal Volume (survives between runs), matching the same outputs/<name>/seed_<s>/ layout as the Kaggle
sessions (scripts/kaggle_queue.py) and the CAGE shell runners (run_cage_5seeds_description.sh,
run_cage_zeroshot.sh, run_cage_checkpoint.sh). Two GPU tiers: A100 for mT5-large (needs the VRAM), A10
for everything else (viT5-large, viT5-base, CodeT5-base, CAGE/PhoBERT-base -- all comfortably fit).

Setup (already done on this machine): `pip install modal`, `modal setup`.
Pricing (Starter plan, 2026-10-01): A10 $0.000306/s ~= $1.10/h, A100-40GB $0.000583/s ~= $2.10/h.
Concurrency cap: 10 GPUs at once -- dispatch below submits every job up front via .spawn() (not a
blocking .starmap() per tier) so both tiers' jobs queue and run concurrently, not one tier after the other.

Usage:
    modal run modal_runner.py --phase baseline_gap   # mt5large:beauty, the last open baseline gap (4 jobs)
    modal run modal_runner.py --phase ckpt           # CAGE checkpoint-for-paraphrase-eval, 1 seed/domain (5 jobs)
    modal run modal_runner.py --phase zeroshot       # CAGE leave-one-category-out, 3 seeds x 15 combos (45 jobs)
    modal run modal_runner.py --phase all_remaining  # all of the above together (54 jobs), max parallelism
    modal run modal_runner.py --job "mt5large:beauty:7"       # one baseline job (job_key:seed)
    modal run modal_runner.py --job "zeroshot_beauty_mau_sac:42"  # one CAGE job (job_key:seed)
    modal volume get acsa-outputs <out_dir>/seed_<s>/test_predictions.jsonl ./local_path
        # (fetching whole directories is flaky on this modal client version -- fetch files by name)
"""
import modal

app = modal.App("acsa-baselines")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers==4.57.6", "accelerate", "sentencepiece", "pyvi",
        "scikit-learn", "scipy", "numpy",
    )
    .add_local_dir(".", remote_path="/root/acsa", ignore=["outputs", ".git", "*.pdf", "*.zip"])
)

volume = modal.Volume.from_name("acsa-outputs", create_if_missing=True)

DOMAIN_DATA = {"restaurant": "Res_ABSA", "hotel": "Hotel_ABSA", "phone": "Phone_ABSA",
               "education": "Education_ABSA", "beauty": "Beauty_ABSA"}
DOMAIN_PROMPT = {"education": "university course evaluation", "beauty": "beauty product"}  # t5 instruction-tuning jobs only


def _t5(domain, model, batch_size=8, eval_batch_size=4):
    d = DOMAIN_DATA[domain]
    return ["baselines/t5_seq2seq_baseline.py",
        "--train_path", f"{d}/Train.txt", "--dev_path", f"{d}/Dev.txt", "--test_path", f"{d}/Test.txt",
        "--model_name", model, "--batch_size", str(batch_size), "--eval_batch_size", str(eval_batch_size),
        "--epochs", "10", "--gradient_checkpointing"]


def _instr(domain, fmt, lang):
    d = DOMAIN_DATA[domain]
    return ["baselines/t5_instruction_tuning.py", "--domain", DOMAIN_PROMPT[domain],
        "--format", fmt, "--lang", lang,
        "--train_path", f"{d}/Train.txt", "--dev_path", f"{d}/Dev.txt", "--test_path", f"{d}/Test.txt"]


def _cage(domain, holdout_category=None):
    """CAGE fixed-fusion, description text -- same protocol as run_cage_5seeds_description.sh /
    run_cage_zeroshot.sh / run_cage_checkpoint.sh (PhoBERT-base-v2, 10 epochs, batch 16, max_length 256)."""
    d = DOMAIN_DATA[domain]
    args = ["train_mtl_acsa_v2.py",
        "--train_path", f"{d}/Train.txt", "--dev_path", f"{d}/Dev.txt", "--test_path", f"{d}/Test.txt",
        "--model_name", "vinai/phobert-base-v2", "--domain", domain, "--category_text", "description",
        "--epochs", "10", "--batch_size", "16", "--max_length", "256"]
    if holdout_category:
        args += ["--holdout_category", holdout_category]
    return args


# job_key -> dict(args=[script + args, WITHOUT --seeds/--output_dir -- _run() adds those], out_dir=
# <canonical name under outputs/>, gpu="small"|"large", keep_ckpt=bool). out_dir is explicit per job
# (not derived from job_key) specifically to avoid the naming-convention bug a previous version of
# this file had (job_key.replace(":", "_") silently produced "nl_vi_beauty" instead of the
# "t5_beauty_nl_vi" layout scripts/cage_stats.py actually looks for).
JOBS = {
    "smoke": dict(args=["baselines/t5_seq2seq_baseline.py",
        "--train_path", "Education_ABSA/Train_smoke.txt", "--dev_path", "Education_ABSA/Dev_smoke.txt",
        "--test_path", "Education_ABSA/Test_smoke.txt", "--model_name", "VietAI/vit5-base",
        "--batch_size", "4", "--eval_batch_size", "4", "--epochs", "1"], out_dir="smoke", gpu="small"),

    # batch_size=16 (not the small-tier default of 8): A100-40GB measured only ~12.5GB/GPU at batch=8
    # (observed live on the dashboard while the 4 currently-running mt5large:beauty jobs trained under the
    # old default -- GPU utilization sat ~20%, well under the A10/Kaggle-T4-era settings' headroom on this
    # much bigger card). 2x batch keeps a comfortable margin under 40GB while roughly doubling throughput.
    # Only applies to FUTURE dispatches of this job_key, not jobs already running.
    "mt5large:education": dict(args=_t5("education", "google/mt5-large", batch_size=16, eval_batch_size=8), out_dir="mt5large_education", gpu="large"),
    "mt5large:beauty": dict(args=_t5("beauty", "google/mt5-large", batch_size=16, eval_batch_size=8), out_dir="mt5large_beauty", gpu="large"),
    "vit5large:education": dict(args=_t5("education", "VietAI/vit5-large"), out_dir="vit5large_education", gpu="small"),
    "vit5large:beauty": dict(args=_t5("beauty", "VietAI/vit5-large"), out_dir="vit5large_beauty", gpu="small"),

    "nl_vi:beauty": dict(args=_instr("beauty", "nl", "vi"), out_dir="t5_beauty_nl_vi", gpu="small"),
    "nl_en:beauty": dict(args=_instr("beauty", "nl", "en"), out_dir="t5_beauty_nl_en", gpu="small"),
    "code_vi:beauty": dict(args=_instr("beauty", "code", "vi"), out_dir="t5_beauty_code_vi", gpu="small"),
    "code_en:beauty": dict(args=_instr("beauty", "code", "en"), out_dir="t5_beauty_code_en", gpu="small"),
}

# CAGE checkpoint-for-paraphrase-eval (run_cage_checkpoint.sh's protocol): 1 seed per domain, best_model.pt kept.
for _d in DOMAIN_DATA:
    JOBS[f"ckpt_{_d}"] = dict(args=_cage(_d), out_dir=f"cage_ckpt_{_d}_fixed", gpu="small", keep_ckpt=True)

# CAGE zero-shot leave-one-category-out (run_cage_zeroshot.sh's TABLE, identical to
# scripts/zeroshot_stats.py's TABLE -- keep all three in sync if this changes).
ZEROSHOT_TABLE = [
    ("restaurant", "FOOD#QUALITY", "food_quality"),
    ("restaurant", "RESTAURANT#MISCELLANEOUS", "restaurant_miscellaneous"),
    ("restaurant", "DRINKS#PRICES", "drinks_prices"),
    ("hotel", "SERVICE#GENERAL", "service_general"),
    ("hotel", "FOOD&DRINKS#STYLE&OPTIONS", "food_drinks_style_options"),
    ("hotel", "FACILITIES#CLEANLINESS", "facilities_cleanliness"),
    ("phone", "GENERAL", "general"),
    ("phone", "PRICE", "price"),
    ("phone", "STORAGE", "storage"),
    ("education", "Hành vi", "hanh_vi"),
    ("education", "Nói chung", "noi_chung"),
    ("education", "Cung cấp tài liệu", "cung_cap_tai_lieu"),
    ("beauty", "Màu sắc", "mau_sac"),
    ("beauty", "Bao bì", "bao_bi"),
    ("beauty", "Độ bền màu", "do_ben_mau"),
]
for _domain, _category, _slug in ZEROSHOT_TABLE:
    JOBS[f"zeroshot_{_domain}_{_slug}"] = dict(
        args=_cage(_domain, holdout_category=_category),
        out_dir=f"cage_zeroshot_{_domain}_{_slug}", gpu="small", keep_ckpt=False)

# ---------------------------------------------------------------------------------------------------
# Phases: (job_key, seed) pairs.
# ---------------------------------------------------------------------------------------------------
BASELINE_GAP_PHASE = [("mt5large:beauty", s) for s in ("7", "99", "123", "2024")]  # last open baseline gap
CKPT_PHASE = [(f"ckpt_{d}", "42") for d in DOMAIN_DATA]
ZEROSHOT_PHASE = [(f"zeroshot_{d}_{slug}", s) for d, _, slug in ZEROSHOT_TABLE for s in ("42", "123", "2024")]
ALL_REMAINING_PHASE = BASELINE_GAP_PHASE + CKPT_PHASE + ZEROSHOT_PHASE

# Budget-constrained resumption (2026-10-01, ~$2.70 left): of ZEROSHOT_PHASE's 45 jobs, only these 13
# genuinely completed (verified against the volume, not just the crashed run's own "done" list --
# hotel_facilities_cleanliness:42 LOOKS like it ran but only ever wrote best_model.pt, never
# metrics.json/test_predictions.jsonl, so it does NOT count as done and is included below again):
#   restaurant_food_quality, restaurant_restaurant_miscellaneous, restaurant_drinks_prices,
#   hotel_service_general (all 3 seeds each); beauty_mau_sac (seed 42 only, from the earlier
#   standalone calibration run, predates and is separate from the 45-job batch's own tally).
# Remaining 32 jobs, reordered cheap-domain-first (Beauty/Education/Phone finished faster per job than
# Hotel/Restaurant in the completed data so far -- plausibly because their category count K is smaller,
# which scales the per-step category-query re-encoding cost) so a tight budget banks more COMPLETE
# domains rather than stalling mid-way through an expensive one again.
_DONE = {("restaurant", "food_quality", s) for s in ("42", "123", "2024")} \
    | {("restaurant", "restaurant_miscellaneous", s) for s in ("42", "123", "2024")} \
    | {("restaurant", "drinks_prices", s) for s in ("42", "123", "2024")} \
    | {("hotel", "service_general", s) for s in ("42", "123", "2024")} \
    | {("beauty", "mau_sac", "42")} \
    | {("beauty", "bao_bi", s) for s in ("42", "123", "2024")}  # completed 2026-10-01 19:58
_CHEAP_FIRST_DOMAIN_ORDER = {"beauty": 0, "education": 1, "phone": 2, "restaurant": 3, "hotel": 4}
_remaining = sorted(
    ((d, slug, s) for d, _, slug in ZEROSHOT_TABLE for s in ("42", "123", "2024") if (d, slug, s) not in _DONE),
    key=lambda t: (_CHEAP_FIRST_DOMAIN_ORDER[t[0]], t[1], t[2]))
ZEROSHOT_REMAINING_CHEAP_FIRST_PHASE = [(f"zeroshot_{d}_{slug}", s) for d, slug, s in _remaining]

PHASES = {"baseline_gap": BASELINE_GAP_PHASE, "ckpt": CKPT_PHASE, "zeroshot": ZEROSHOT_PHASE,
          "all_remaining": ALL_REMAINING_PHASE, "zeroshot_remaining_cheap_first": ZEROSHOT_REMAINING_CHEAP_FIRST_PHASE}


def _run(job_key: str, seed: str):
    import subprocess
    from pathlib import Path
    job = JOBS[job_key]
    script, args = job["args"][0], job["args"][1:]
    out_dir = job["out_dir"]
    # train_mtl_acsa_v2.py (CAGE) creates its OWN seed_<s>/ subdir internally whenever --seeds is passed
    # (even a single value) -- base_output_dir/seed_<s>/ is where it writes metrics.json/test_predictions.jsonl.
    # The t5 baseline scripts do the opposite: they write those files flat at whatever --output_dir is
    # given, with no self-appended seed level. So CAGE jobs get the bare out_dir (its own seed_<s>/
    # append then lands at the expected outputs/<out_dir>/seed_<s>/); t5 jobs get the seed suffix added
    # here. Passing the seed suffix to BOTH double-nests CAGE into seed_<s>/seed_<s>/ (caught via a real
    # calibration run on Modal -- see git history/plan notes).
    is_cage = script == "train_mtl_acsa_v2.py"
    base_output_dir = f"outputs/{out_dir}" if is_cage else f"outputs/{out_dir}/seed_{seed}"
    cmd = ["python3", script, *args, "--seeds", seed, "--output_dir", base_output_dir]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, cwd="/root/acsa", check=True)
    if not job.get("keep_ckpt", False):
        # CAGE checkpoints are ~0.5GB/seed; only the "ckpt_*" family (experiment b, paraphrase-robustness
        # eval) needs one kept -- everything else (t5 baselines never produce one anyway; zero-shot
        # doesn't need one) gets it dropped to keep the volume lean, same rationale as cage_common.sh.
        for p in Path(f"/root/acsa/outputs/{out_dir}/seed_{seed}").rglob("best_model.pt"):
            p.unlink()
    volume.commit()
    return f"{out_dir}/seed_{seed}"


@app.function(image=image, gpu="A10", volumes={"/root/acsa/outputs": volume}, timeout=6 * 60 * 60)
def run_small(job_key: str, seed: str):
    return _run(job_key, seed)


@app.function(image=image, gpu="A100-40GB", volumes={"/root/acsa/outputs": volume}, timeout=6 * 60 * 60)
def run_large(job_key: str, seed: str):
    return _run(job_key, seed)


@app.local_entrypoint()
def main(job: str = "", phase: str = "", budget_hours: float = 0.0):
    if job:
        job_key, seed = job.rsplit(":", 1)
        fn = run_large if JOBS[job_key]["gpu"] == "large" else run_small
        print("done:", fn.remote(job_key, seed))
        return
    pairs = PHASES.get(phase, [])
    if not pairs:
        print(f'pass --job "key:seed" or --phase {"|".join(PHASES)}'); return

    if budget_hours > 0:
        # Hard cost cap: run jobs ONE AT A TIME (blocking .remote(), never .spawn()) so at most a single
        # A10 container is ever billed concurrently, and stop dispatching the NEXT job once elapsed
        # wall-clock time would exceed budget_hours -- a deliberately conservative local safety net on
        # top of Modal's own account-level spend limit (that limit hard-blocked everything once already,
        # see [[modal_runner_workflow]] memory note / mt5large_beauty's original empty-seed incident).
        # At ~$1.10/h (A10, Starter plan), budget_hours should be set to (remaining credits / 1.10), e.g.
        # --budget_hours 4 for ~$4.40 of headroom.
        import time
        start = time.time()
        done, skipped = [], []
        print(f"serial mode: budget={budget_hours}h (~${budget_hours * 1.10:.2f} at A10 rate), {len(pairs)} jobs queued")
        for job_key, seed in pairs:
            elapsed_h = (time.time() - start) / 3600
            if elapsed_h >= budget_hours:
                skipped = [f"{k}:{s}" for k, s in pairs[len(done):]]
                print(f"\nbudget reached ({elapsed_h:.2f}h elapsed) -- stopping before {job_key}:{seed}. "
                      f"{len(done)} done, {len(skipped)} skipped (rerun later with the same --phase).")
                break
            fn = run_large if JOBS[job_key]["gpu"] == "large" else run_small
            print(f"[{elapsed_h:.2f}h elapsed] running {job_key}:{seed} ...")
            try:
                r = fn.remote(job_key, seed)
                print(f"  done: {job_key}:{seed} -> {r}")
                done.append(r)
            except Exception as e:
                print(f"  FAILED: {job_key}:{seed} -> {e}")
        else:
            print(f"\nall {len(pairs)} jobs finished within budget ({(time.time() - start) / 3600:.2f}h elapsed)")
        print("done:", done)
        if skipped:
            print("skipped (budget):", skipped)
        return
    # Dispatch EVERY job up front with .spawn() (non-blocking) across both GPU tiers at once, instead of
    # a blocking run_small.starmap(...) followed by run_large.starmap(...) -- the latter would not even
    # submit the large-tier jobs to Modal until every small-tier job finished, under-using the GPU pool
    # whenever one tier has more queued work than the other. Modal's own scheduler enforces the
    # concurrency cap across whatever is actually dispatched.
    handles = []
    for job_key, seed in pairs:
        fn = run_large if JOBS[job_key]["gpu"] == "large" else run_small
        handles.append((job_key, seed, fn.spawn(job_key, seed)))
    print(f"dispatched {len(handles)} jobs ({sum(1 for k,_,_ in handles if JOBS[k]['gpu']=='large')} large A100 + "
          f"{sum(1 for k,_,_ in handles if JOBS[k]['gpu']=='small')} small A10), waiting for completion...")
    results = []
    for job_key, seed, h in handles:
        try:
            r = h.get()
            print(f"  done: {job_key}:{seed} -> {r}")
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {job_key}:{seed} -> {e}")
    print("done:", results)
