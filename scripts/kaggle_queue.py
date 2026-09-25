#!/usr/bin/env python3
"""Seed top-up queue for the baselines we ran ourselves (5 seeds: 42, 123, 2024, 7, 99), split into Kaggle sessions.

At `plan` time (locally) seeds that already have a result are left out: `<dir>/seed_<s>/...` (new layout) or a flat `<dir>/multi_seed_summary.json`
whose per_seed list contains the seed (existing seed-42 runs).  Rows taken from earlier papers are not queued.

  python3 scripts/kaggle_queue.py list                       inventory: done / to run / estimated hours per group
  python3 scripts/kaggle_queue.py plan  [--budget-hours 10.5] [--gpus 2] [--calibrate outputs/_job_times.tsv]
                                                             writes scripts/rerun_plan.json (jobs split into sessions)
  python3 scripts/kaggle_queue.py run --session 1 [--budget-hours 10.5]     (on Kaggle, from the repo root)

`run` (on Kaggle, where outputs/ is empty, so no already-done check) never starts a job whose estimate no longer fits in the budget (it is left for the next session), writes each job's
log to outputs/_logs/, appends the measured minutes to outputs/_job_times.tsv (feed it back with `plan --calibrate`), and
re-packs the small result files (scripts/pack_results.sh) after every finished job, so a cut session loses at most one job.
Estimates are NOT measured: they assume Kaggle 2 x T4 and are meant to be replaced by --calibrate after the first session.
"""
import argparse, json, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "scripts" / "rerun_plan.json"
SEEDS = [42, 123, 2024, 7, 99]
DOM = {  # data dir, domain phrase used inside instruction prompts
    "restaurant": ("Res_ABSA", "restaurant"), "hotel": ("Hotel_ABSA", "hotel"), "phone": ("Phone_ABSA", "mobile phone"),
    "education": ("Education_ABSA", "university course evaluation"), "beauty": ("Beauty_ABSA", "beauty product")}
ORDER = ["bert", "cnn", "t5base", "instr", "t5large"]
# Unmeasured first guesses, minutes per (job) on one T4; large models use both GPUs.
EST = {  # minutes per job; base-size numbers scaled from round 1 (BERT ~90 min, instruction ~100 min on Restaurant); large ones are guesses
    "bert": {"restaurant": 90, "hotel": 90, "phone": 95, "education": 65, "beauty": 130},          # PhoBERT + XLM-R + ensemble
    "cnn": {"beauty": 10},
    "t5base": {"education": 110, "beauty": 300},
    "instr": {"restaurant": 100, "hotel": 100, "phone": 108, "education": 65, "beauty": 180},
    "t5large": {"mt5large/education": 300, "mt5large/beauty": 720, "vit5large/education": 200, "vit5large/beauty": 480},
}


def seed_done(d, seed, marker="multi_seed_summary.json"):
    d = ROOT / "outputs" / d
    if (d / f"seed_{seed}" / marker).exists():
        return True
    flat = d / "multi_seed_summary.json"
    if flat.exists():
        try:
            return seed in [p.get("seed") for p in json.loads(flat.read_text()).get("per_seed", [])]
        except Exception:
            return False
    return seed == 42 and marker == "test_predictions.jsonl" and (d / marker).exists()


def build_jobs(calib):
    jobs = []

    def add(group, name, dom, seed, est, gpus, cmds, dirs):
        key = f"{group}/{name}/{dom}"
        jobs.append({"id": f"{group}:{name}:{dom}:s{seed}", "group": group, "domain": dom, "seed": seed, "gpus": gpus,
                     "est_min": calib.get(key, est), "cmds": cmds, "dirs": dirs})

    for d, (dd, prompt_dom) in DOM.items():
        tr = f"--train_path {dd}/Train.txt --dev_path {dd}/Dev.txt --test_path {dd}/Test.txt"
        for s in SEEDS:
            if not seed_done(f"bert_{d}_ensemble", s, "test_predictions.jsonl"):
                ph, xl, en = (f"outputs/bert_{d}_phobert/seed_{s}", f"outputs/bert_{d}_xlmr/seed_{s}", f"outputs/bert_{d}_ensemble/seed_{s}")
                add("bert", "pho+xlmr", d, s, EST["bert"][d], 1, [
                    f"python3 baselines/bert_baseline.py --mode train {tr} --seeds {s} --output_dir {ph} --model_name vinai/phobert-base-v2 --segmenter pyvi",
                    f"python3 baselines/bert_baseline.py --mode train {tr} --seeds {s} --output_dir {xl} --model_name xlm-roberta-base --segmenter none",
                    f"python3 baselines/bert_baseline.py --mode ensemble --checkpoints {ph}/best_model.pt {xl}/best_model.pt --test_path {dd}/Test.txt --output_dir {en}",
                    f"rm -f {ph}/best_model.pt {xl}/best_model.pt"], [f"outputs/bert_{d}_ensemble/seed_{s}/test_predictions.jsonl"])
        if d == "beauty":
            for s in SEEDS:
                for name, extra in (("cnn", ""), ("bilstm_cnn", " --use_bilstm")):
                    if not seed_done(f"{name}_beauty", s):
                        out = f"outputs/{name}_beauty/seed_{s}"
                        add("cnn", name, d, s, EST["cnn"][d], 1,
                            [f"python3 baselines/cnn_baseline.py {tr} --seeds {s} --output_dir {out}{extra}"], [f"{out}/multi_seed_summary.json"])
        if d in ("education", "beauty"):
            for s in SEEDS:
                if not seed_done(f"vit5base_{d}", s):
                    out = f"outputs/vit5base_{d}/seed_{s}"
                    add("t5base", "vit5base", d, s, EST["t5base"][d], 1,
                        [f"python3 baselines/t5_seq2seq_baseline.py {tr} --output_dir {out} --model_name VietAI/vit5-base --seeds {s}"], [f"{out}/multi_seed_summary.json"])
                for name, model in (("mt5large", "google/mt5-large"), ("vit5large", "VietAI/vit5-large")):
                    if not seed_done(f"{name}_{d}", s):
                        out = f"outputs/{name}_{d}/seed_{s}"
                        add("t5large", name, d, s, EST["t5large"][f"{name}/{d}"], 2, [
                            f"python3 baselines/t5_seq2seq_baseline.py {tr} --output_dir {out} --model_name {model} --seeds {s} "
                            f"--batch_size 8 --eval_batch_size 4 --epochs 10 --gradient_checkpointing --device_map_auto",
                            "rm -rf ~/.cache/huggingface/hub"], [f"{out}/multi_seed_summary.json"])
        for fmt in ("code", "nl"):
            for lang in ("vi", "en"):
                for s in SEEDS:
                    dirname = f"t5_{d}_{fmt}_{lang}"
                    if not seed_done(dirname, s):
                        out = f"outputs/{dirname}/seed_{s}"
                        add("instr", f"{fmt}_{lang}", d, s, EST["instr"][d], 1,
                            [f"python3 baselines/t5_instruction_tuning.py --domain '{prompt_dom}' --format {fmt} --lang {lang} {tr} --output_dir {out} --seeds {s}"], [f"{out}/multi_seed_summary.json"])
    rank = {g: i for i, g in enumerate(ORDER)}
    jobs.sort(key=lambda j: (rank[j["group"]], j["seed"] != 42, list(DOM).index(j["domain"]), j["seed"]))
    return jobs


def load_calibration(path):
    if not path or not Path(path).exists():
        return {}
    acc = {}
    for line in Path(path).read_text().splitlines():
        jid, minutes = line.split("\t")[:2]
        g, name, dom, _ = jid.split(":")
        acc.setdefault(f"{g}/{name}/{dom}", []).append(float(minutes))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def pack_sessions(jobs, budget_min, gpus):
    sessions, remaining = [], list(jobs)
    while remaining:
        free, placed, sess = [0.0] * gpus, [], []
        for j in list(remaining):
            need = min(j["gpus"], gpus)
            free.sort()
            start = free[need - 1]
            if start + j["est_min"] <= budget_min:
                for i in range(need):
                    free[i] = start + j["est_min"]
                sess.append(j); remaining.remove(j)
        if not sess:  # a single job longer than the budget
            sess = [remaining.pop(0)]
        sessions.append({"jobs": [j["id"] for j in sess], "hours": round(max(free) / 60, 1)})
    return sessions


def cmd_list(args, jobs):
    tot = {}
    for j in jobs:
        g = tot.setdefault(j["group"], [0, 0.0]); g[0] += 1; g[1] += j["est_min"] * j["gpus"] / 60
    print(f"{'group':8s}{'jobs':>6s}{'GPU-hours (est)':>17s}")
    for g in ORDER:
        if g in tot:
            print(f"{g:8s}{tot[g][0]:6d}{tot[g][1]:17.1f}")
    print(f"{'total':8s}{sum(v[0] for v in tot.values()):6d}{sum(v[1] for v in tot.values()):17.1f}")
    by = {}
    for j in jobs:
        by.setdefault((j["group"], j["domain"]), []).append(j["seed"])
    print("\nremaining seeds per (group, domain), summed over the rows of the group:")
    for (g, d), s in sorted(by.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1])):
        print(f"  {g:8s}{d:11s}{len(s):3d} jobs")


def cmd_plan(args, jobs):
    sessions = pack_sessions(jobs, args.budget_hours * 60, args.gpus)
    PLAN.write_text(json.dumps({"round": args.round, "budget_hours": args.budget_hours, "gpus": args.gpus, "sessions": sessions,
                                "jobs": {j["id"]: j for j in jobs}}, indent=1))
    print(f"{len(jobs)} jobs -> {len(sessions)} sessions of <= {args.budget_hours} h ({args.gpus} GPUs):")
    for i, s in enumerate(sessions, 1):
        groups = {}
        for jid in s["jobs"]:
            groups[jid.split(":")[0]] = groups.get(jid.split(":")[0], 0) + 1
        print(f"  session {i:2d}: ~{s['hours']:4.1f} h, {len(s['jobs']):3d} jobs {groups}")
    for j in jobs:
        if j["est_min"] > args.budget_hours * 60:
            print(f"WARNING {j['id']}: estimated {j['est_min']/60:.1f} h > budget, it will be cut by the hard deadline; measure a smaller job first or change its settings")
    print("wrote", PLAN.relative_to(ROOT))


def cmd_run(args, _):
    plan = json.loads(PLAN.read_text())
    ids = plan["sessions"][args.session - 1]["jobs"]
    ids = sorted(ids, key=lambda i: plan["jobs"][i]["gpus"] != 2)  # both-GPU jobs first: they wait for idle GPUs otherwise
    tag = f"queue_{plan.get('round', 'r1')}_s{args.session}"
    if args.dry_run:
        for jid in ids:
            j = plan["jobs"][jid]
            print(f"# {jid}  est {j['est_min']:.0f} min, {j['gpus']} GPU(s)")
            print(*j["cmds"], sep="\n")
        return
    budget = (args.budget_hours or plan["budget_hours"]) * 3600
    logs = ROOT / "outputs" / "_logs"; logs.mkdir(parents=True, exist_ok=True)
    t0, lock, running, deferred = time.time(), threading.Lock(), [], []
    workers = plan["gpus"]

    def finish(job, minutes, rc):
        with lock:
            with open(ROOT / "outputs" / "_job_times.tsv", "a") as f:
                f.write(f"{job['id']}\t{minutes:.1f}\t{rc}\n")
        subprocess.run(["bash", "scripts/pack_results.sh", tag], cwd=ROOT, stdout=subprocess.DEVNULL)

    procs, stop = set(), threading.Event()

    def watchdog():  # a notebook that hits Kaggle's 12 h wall is cancelled and its output may be lost: stop cleanly first
        while not stop.is_set():
            if time.time() - t0 > hard:
                print(f"[{time.strftime('%H:%M:%S')}] HARD DEADLINE ({hard/3600:.2f} h): killing running jobs so the notebook can finish", flush=True)
                stop.set()
                for pr in list(procs):
                    try:
                        os.killpg(pr.pid, 9)
                    except Exception:
                        pass
                return
            time.sleep(10)

    def work(job, gpu):
        env = dict(os.environ)
        if job["gpus"] == 1 and workers > 1:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        start = time.time()
        rc = 0
        with open(logs / (job["id"].replace(":", "_") + ".log"), "w") as log:
            for c in job["cmds"]:
                if rc != 0 and not c.startswith("rm "):
                    continue  # after a failure only the cleanup commands still run
                pr = subprocess.Popen(c, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                procs.add(pr)
                r = pr.wait()
                procs.discard(pr)
                rc = rc or r
        minutes = (time.time() - start) / 60
        print(f"[{time.strftime('%H:%M:%S')}] {'DONE' if rc == 0 else 'FAIL'} {job['id']} ({minutes:.0f} min)", flush=True)
        finish(job, minutes, rc)

    hard = budget + args.grace_hours * 3600
    threading.Thread(target=watchdog, daemon=True).start()
    free_gpus = list(range(workers))
    for jid in ids:
        job = plan["jobs"][jid]
        if stop.is_set() or time.time() - t0 + job["est_min"] * 60 > budget:
            deferred.append(jid); continue
        need = min(job["gpus"], workers)
        while True:  # wait until enough GPUs are free
            for item in list(running):
                if not item[0].is_alive():
                    running.remove(item); free_gpus.extend(item[1])
            if len(free_gpus) >= need:
                break
            time.sleep(5)
        gs = [free_gpus.pop(0) for _ in range(need)]
        th = threading.Thread(target=work, args=(job, gs[0])); th.start()
        running.append((th, gs))
        print(f"[{time.strftime('%H:%M:%S')}] START {jid} on GPU {gs}", flush=True)
    for t, _g in running:
        t.join()
    stop.set()
    if deferred:
        print(f"{len(deferred)} jobs did not fit the budget and were left for the next session:", *deferred, sep="\n  ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "plan", "run"])
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--calibrate", default=None)
    ap.add_argument("--round", default="r2", help="plan: tag put in the zip names (queue_<round>_s<N>.zip) so rounds do not overwrite each other")
    ap.add_argument("--tag", action="store_true", help="print the zip tag of --session and exit")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--grace-hours", type=float, default=0.75, help="run: hard-kill running jobs this long after --budget-hours")
    a = ap.parse_args()
    if a.cmd == "run" and a.tag:
        print(f"queue_{json.loads(PLAN.read_text()).get('round', 'r1')}_s{a.session}")
    elif a.cmd == "run":
        cmd_run(a, None)
    else:
        a.budget_hours = a.budget_hours or 10.5
        jobs = build_jobs(load_calibration(a.calibrate))
        {"list": cmd_list, "plan": cmd_plan}[a.cmd](a, jobs)
