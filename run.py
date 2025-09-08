#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import subprocess
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional

import torch

# ---------------------------------------------------------------------
# 간단 유틸
# ---------------------------------------------------------------------
def get_git_commit_or_none() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def update_latest_link(base_dir: str, target_dir: str) -> None:
    latest_path = os.path.join(base_dir, "latest.txt")
    try:
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(target_dir)
    except Exception as e:
        print(f"[WARN] Could not update latest link file: {e}")

def parse_gpu_list(arg: Optional[str]) -> List[int]:
    if arg is None or arg == "": return []
    if isinstance(arg, int): return [arg]
    parts = [p.strip() for p in str(arg).split(",") if p.strip() != ""]
    return [int(p) for p in parts if p.isdigit()]

# ---------------------------------------------------------------------
# 병렬 실행 로직
# ---------------------------------------------------------------------
def _model_supports_residual(model: str, root_path: str) -> bool:
    residual_file = os.path.join(root_path, "tests", model, "user_pgdpo_residual.py")
    return os.path.isfile(residual_file)

def orchestrate_all(
    *,
    model: str,
    gpus: List[int],
    plots_base: str,
    cli_args: argparse.Namespace,
):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_dir = os.path.join(ROOT, plots_base, model, "multi", ts)
    ensure_dir(multi_dir)

    supports_residual = _model_supports_residual(model, ROOT)
    modes = ["base", "run", "projection"]
    if supports_residual:
        modes.append("residual")

    if not gpus:
        gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [-1]
    gpu_map = {mode: gpus[i % len(gpus)] for i, mode in enumerate(modes)}

    procs = []
    print(f"🧩 Orchestrating multi-run for model '{model}':")
    for mode_key in modes:
        gpu_id = gpu_map[mode_key]
        cmd = [sys.executable, os.path.abspath(__file__), model, f"--{mode_key}"]
        
        # 전달받은 모든 하이퍼파라미터 인자를 자식 프로세스에 그대로 전달
        if cli_args.d_override is not None: cmd.extend(["-d", str(cli_args.d_override)])
        if cli_args.k_override is not None: cmd.extend(["-k", str(cli_args.k_override)])
        if cli_args.epochs_override is not None: cmd.extend(["--epochs", str(cli_args.epochs_override)])
        if cli_args.batch_size_override is not None: cmd.extend(["--batch-size", str(cli_args.batch_size_override)])
        if cli_args.lr_override is not None: cmd.extend(["--lr", str(cli_args.lr_override)])
        if cli_args.seed_override is not None: cmd.extend(["--seed", str(cli_args.seed_override)])
        cmd.extend(["--plots", plots_base, "--tag", f"multi_{ts}"])

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_id >= 0 and torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd.extend(["--gpu", "0"]) # 자식은 자신의 0번 GPU를 사용
            gpu_label = f"cuda:{gpu_id}"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            cmd.extend(["--gpu", "-1"])
            gpu_label = "cpu"

        log_path = os.path.join(multi_dir, f"{mode_key}_gpu{gpu_id if gpu_id >= 0 else 'cpu'}.log")
        print(f"  • Launching [{mode_key}] on GPU={gpu_label} (log: {os.path.basename(log_path)})")
        
        with open(log_path, "w", encoding="utf-8") as lf:
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
            procs.append((mode_key, p))

    for mode_key, p in procs:
        p.wait()
        status = "✅ PASSED" if p.returncode == 0 else f"❌ FAILED (code: {p.returncode})"
        print(f"  • Finished [{mode_key}]: {status}")


# ---------------------------------------------------------------------
# 단일 실행 로직
# ---------------------------------------------------------------------
def single_run(
    *,
    mode_key: str,
    model: str,
    gpu_sel: str,
    plots_base: str,
    tag: Optional[str] = None,
    d_override: Optional[int] = None,
    k_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    lr_override: Optional[float] = None,
    seed_override: Optional[int] = None
) -> None:
    # 1. 모델별 k=0 강제 플래그 설정
    MODELS_FORCE_K0 = {"vpp", "mt_1d", "mt_nd", "sir", "harvest", "park"}
    if model in MODELS_FORCE_K0:
        os.environ["PGDPO_FORCE_K0"] = "1"
        if k_override is not None and k_override != 0:
            print(f"[INFO] Model '{model}' forces k=0. User input k={k_override} will be ignored.")

    # 2. 모든 외부 파라미터를 환경변수로 주입
    if d_override is not None: os.environ["PGDPO_D"] = str(d_override)
    if k_override is not None: os.environ["PGDPO_K"] = str(k_override)
    if epochs_override is not None: os.environ["PGDPO_EPOCHS"] = str(epochs_override)
    if batch_size_override is not None: os.environ["PGDPO_BATCH_SIZE"] = str(batch_size_override)
    if lr_override is not None: os.environ["PGDPO_LR"] = str(lr_override)
    if seed_override is not None: os.environ["PGDPO_SEED"] = str(seed_override)

    ROOT = os.path.dirname(os.path.abspath(__file__))
    CORE_PATH = os.path.join(ROOT, "core")
    TEST_PATH = os.path.join(ROOT, "tests", model)

    # 3. 경로 추가 후 모듈 로드
    sys.path.insert(0, CORE_PATH)
    sys.path.insert(0, TEST_PATH)

    print(f"✅ Running with model: {model}")
    overrides = []
    if d_override is not None: overrides.append(f"d={d_override}")
    if k_override is not None: overrides.append(f"k={k_override}")
    if epochs_override is not None: overrides.append(f"epochs={epochs_override}")
    if batch_size_override is not None: overrides.append(f"batch_size={batch_size_override}")
    if lr_override is not None: overrides.append(f"lr={lr_override}")
    if seed_override is not None: overrides.append(f"seed={seed_override}")
    if overrides:
        print(f"   Overrides: {', '.join(overrides)}")

    # GPU 선택
    sel = "cuda" if torch.cuda.is_available() else "cpu"
    if gpu_sel.startswith("cuda:"):
        try:
            idx = int(gpu_sel.split(":")[1])
            torch.cuda.set_device(idx)
            sel = f"cuda:{idx}"
        except Exception:
            pass # 기본 cuda 사용
    elif gpu_sel == "cpu":
        sel = "cpu"
    print(f"✅ Selected device: {sel}")

    # --- ✨✨✨ 수정된 부분 ✨✨✨ ---
    # 설정이 적용된 모듈을 먼저 로드해서 d, k 값을 가져옵니다.
    from pgdpo_base import d, k, CRN_SEED_EU

    # d, k 값을 포함하여 출력 경로 생성
    param_str = f"d{d}_k{k}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_name = ts if tag is None else f"{ts}_{tag}"
    
    # 최종 출력 경로: plots/<model>/<mode>/d<d>_k<k>/<timestamp>
    OUT_DIR = os.path.join(ROOT, plots_base, model, mode_key, param_str, ts_name)
    ensure_dir(OUT_DIR)
    print(f"✅ Output dir: {OUT_DIR}")
    # --- ✨✨✨ 여기까지 수정 ✨✨✨ ---

    mode_human = {
        "run": "Variance-Reduced Simulation",
        "projection": "P-PGDPO with Projection",
        "base": "Basic Simulation",
        "residual": "Residual Learning",
    }[mode_key]
    print(f"🚀 Mode: {mode_human}")

    # 나머지 필요한 변수들을 로드합니다.
    from pgdpo_base import (
        seed, T, m,
        epochs, batch_size, lr,
        DIM_X, DIM_U,
    )

    # 메타데이터 기록
    meta = {
        "model": model, "mode": mode_key, "timestamp": ts, "outdir": OUT_DIR,
        "selected_gpu": sel, "torch_version": torch.__version__,
        "git_commit": get_git_commit_or_none(),
        "final_params": {
            "d": int(d), "k": int(k), "DIM_X": int(DIM_X), "DIM_U": int(DIM_U),
            "epochs": int(epochs), "batch_size": int(batch_size), "lr": float(lr),
            "seed": int(seed), "T": float(T), "m": int(m), "CRN_SEED_EU": int(CRN_SEED_EU),
        },
        "overrides": {
            "d": d_override, "k": k_override, "epochs": epochs_override,
            "batch_size": batch_size_override, "lr": lr_override, "seed": seed_override,
        }
    }
    write_json(os.path.join(OUT_DIR, "run.json"), meta)

    base_mode_dir = os.path.join(ROOT, plots_base, model, mode_key, param_str) # 경로에 param_str 추가
    update_latest_link(base_mode_dir, OUT_DIR)

    # 모드별 실행
    from pgdpo_base import run_common
    
    # 기본 rmse_kwargs 설정
    rmse_kwargs = {"seed_eval": CRN_SEED_EU, "outdir": OUT_DIR}

    if mode_key == "run":
        from pgdpo_run import train_stage1_run, print_policy_rmse_and_samples_run
        from pgdpo_with_projection import REPEATS, SUBBATCH
        train_fn, rmse_fn = train_stage1_run, print_policy_rmse_and_samples_run
        rmse_kwargs.update({"repeats": REPEATS, "sub_batch": SUBBATCH})
    
    elif mode_key == "projection":
        from pgdpo_base import train_stage1_base
        from pgdpo_with_projection import print_policy_rmse_and_samples_direct, REPEATS, SUBBATCH
        train_fn, rmse_fn = train_stage1_base, print_policy_rmse_and_samples_direct
        rmse_kwargs.update({"repeats": REPEATS, "sub_batch": SUBBATCH})

    elif mode_key == "base":
        from pgdpo_base import train_stage1_base, print_policy_rmse_and_samples_base
        train_fn, rmse_fn = train_stage1_base, print_policy_rmse_and_samples_base
    
    else: # "residual"
        from pgdpo_residual import train_residual_stage1
        from pgdpo_run import print_policy_rmse_and_samples_run
        from pgdpo_with_projection import REPEATS, SUBBATCH
        train_fn, rmse_fn = train_residual_stage1, print_policy_rmse_and_samples_run
        rmse_kwargs.update({"repeats": REPEATS, "sub_batch": SUBBATCH})

    run_common(
        train_fn=train_fn, rmse_fn=rmse_fn, seed_train=None,
        train_kwargs={"outdir": OUT_DIR},
        rmse_kwargs=rmse_kwargs,
    )
    write_text(os.path.join(OUT_DIR, "_DONE"), "ok\n")

# ---------------------------------------------------------------------
# 엔트리 포인트
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PG-DPO runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --all 모드를 위해 모델을 위치 인수가 아닌 옵션으로 변경
    parser.add_argument("model", nargs='?', default=None, help="Name of the model to run (e.g., vpp, ko_nd)")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--all", action='store_true', help="Run all supported modes concurrently for a model")
    mode_group.add_argument("--run", action='store_true', help="Variance-reduced simulation mode")
    mode_group.add_argument("--projection", action='store_true', help="P-PGDPO with projection mode")
    mode_group.add_argument("--base", action='store_true', help="Basic simulation mode")
    mode_group.add_argument("--residual", action='store_true', help="Residual learning mode")

    # 모든 외부 파라미터 인수 추가
    parser.add_argument("--gpu", type=str, default=None, help="GPU index or CSV list for --all mode (e.g., 0 or 0,1,2). Use -1 for CPU.")
    parser.add_argument("--plots", type=str, default="plots", help="Base plots directory")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag for output folder name")
    parser.add_argument("-d", type=int, dest="d_override", default=None, help="Override dimension 'd'")
    parser.add_argument("-k", type=int, dest="k_override", default=None, help="Override dimension 'k'")
    parser.add_argument("--epochs", type=int, dest="epochs_override", default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size_override", default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, dest="lr_override", default=None, help="Override learning rate")
    parser.add_argument("--seed", type=int, dest="seed_override", default=None, help="Override random seed")
    
    args = parser.parse_args()
    
    if args.model is None:
        parser.error("the following arguments are required: model")

    # --all 모드 실행
    if args.all:
        gpus = parse_gpu_list(args.gpu)
        orchestrate_all(model=args.model, gpus=gpus, plots_base=args.plots, cli_args=args)
        return

    # 단일 모드 실행
    if args.run: MODE = "run"
    elif args.projection: MODE = "projection"
    elif args.base: MODE = "base"
    else: MODE = "residual"

    gpu_sel = "cuda" if torch.cuda.is_available() else "cpu"
    if args.gpu is not None:
        gpu_sel = "cpu" if args.gpu == "-1" else f"cuda:{parse_gpu_list(args.gpu)[0]}"

    single_run(
        mode_key=MODE, model=args.model, gpu_sel=gpu_sel,
        plots_base=args.plots, tag=args.tag,
        d_override=args.d_override, k_override=args.k_override,
        epochs_override=args.epochs_override,
        batch_size_override=args.batch_size_override,
        lr_override=args.lr_override,
        seed_override=args.seed_override,
    )

if __name__ == "__main__":
    main()