#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Any

import torch

# ---------------------------------------------------------------------
# 간단 유틸
# ---------------------------------------------------------------------
def get_git_commit_or_none():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def write_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def touch_latest_symlink(base_dir: str, target_dir: str):
    """
    base_dir/<latest> → target_dir 로 심볼릭 링크 갱신 (가능할 때만).
    """
    latest = os.path.join(base_dir, "latest")
    try:
        if os.path.islink(latest) or os.path.exists(latest):
            try:
                os.remove(latest)
            except IsADirectoryError:
                import shutil
                shutil.rmtree(latest, ignore_errors=True)
        os.symlink(target_dir, latest)
    except Exception as e:
        print(f"[WARN] Could not update latest symlink: {e}")


def parse_gpu_list(arg: str | None) -> List[int]:
    if arg is None or arg == "":
        return []
    if isinstance(arg, int):
        return [arg]
    parts = [p.strip() for p in str(arg).split(",") if p.strip() != ""]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------
# 단일 실행(예전 방식) 로직
# ---------------------------------------------------------------------
def single_run(
    *,
    mode_key: str,            # "run" | "projection" | "base" | "residual"
    model: str,
    gpu_sel: str,             # "cpu" | "cuda" | "cuda:<idx>"
    plots_base: str,
    tag: str | None = None,
):
    ROOT = os.path.dirname(os.path.abspath(__file__))
    CORE_PATH = os.path.join(ROOT, "core")
    TEST_PATH = os.path.join(ROOT, "tests", model)

    # import 경로
    sys.path.insert(0, CORE_PATH)
    sys.path.insert(0, TEST_PATH)

    print(f"✅ Running with model: {model}")
    print(f"✅ Core path: {CORE_PATH}")
    print(f"✅ Test path: {TEST_PATH}")

    # GPU 선택(단일 모드)
    if gpu_sel.startswith("cuda"):
        try:
            # gpu_sel 이 "cuda" 또는 "cuda:<idx>"
            if ":" in gpu_sel:
                idx = int(gpu_sel.split(":")[1])
                torch.cuda.set_device(idx)
                sel = f"cuda:{idx}"
            else:
                sel = "cuda"
        except Exception:
            sel = "cuda"
    else:
        sel = "cpu" if not torch.cuda.is_available() else "cuda"
    print(f"✅ Selected GPU: {sel}")

    # 시간/출력 경로: plots/<model>/<mode>/<timestamp>[_tag]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_name = ts if tag is None else f"{ts}_{tag}"
    OUT_DIR = os.path.join(ROOT, plots_base, model, mode_key, ts_name)
    ensure_dir(OUT_DIR)
    print(f"✅ Output dir: {OUT_DIR}")

    mode_human = {
        "run": "Variance-Reduced Simulation (Default)",
        "projection": "P-PGDPO with Projection",
        "base": "Basic Simulation",
        "residual": "Residual Learning",
    }[mode_key]
    print(f"🚀 Mode: {mode_human}")

    # 공통 심볼 로드
    from pgdpo_base import (
        seed, epochs, batch_size, lr,
        T, m, d, k,
        CRN_SEED_EU,
    )

    # P-PGDPO 반복 설정(있으면)
    REPEATS = None
    SUBBATCH = None
    try:
        from pgdpo_with_projection import REPEATS as _R, SUBBATCH as _S
        REPEATS, SUBBATCH = int(_R), int(_S)
    except Exception:
        pass

    # 메타 기록
    meta = {
        "model": model,
        "mode": mode_key,
        "mode_human": mode_human,
        "timestamp": ts,
        "outdir": OUT_DIR,
        "core_path": CORE_PATH,
        "test_path": TEST_PATH,
        "selected_gpu": sel,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit_or_none(),
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "T": float(T),
        "m": int(m),
        "d": int(d),
        "k": int(k),
        "CRN_SEED_EU": int(CRN_SEED_EU),
        "REPEATS": REPEATS,
        "SUBBATCH": SUBBATCH,
    }
    write_json(os.path.join(OUT_DIR, "run.json"), meta)
    write_text(
        os.path.join(OUT_DIR, "run.txt"),
        (
            f"Model : {model}\n"
            f"Mode  : {mode_key} ({mode_human})\n"
            f"Time  : {ts}\n"
            f"GPU   : {sel}\n"
            f"Core  : {CORE_PATH}\n"
            f"Test  : {TEST_PATH}\n"
        ),
    )
    # latest 링크 갱신
    base_mode_dir = os.path.join(ROOT, plots_base, model, mode_key)
    touch_latest_symlink(base_mode_dir, OUT_DIR)

    # 모드별 실행
    from pgdpo_base import run_common

    if mode_key == "run":
        from pgdpo_run import train_stage1_run, print_policy_rmse_and_samples_run
        train_fn = train_stage1_run
        rmse_fn = print_policy_rmse_and_samples_run
        train_kwargs = {"outdir": OUT_DIR}
        rmse_kwargs = {"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "outdir": OUT_DIR}

    elif mode_key == "projection":
        from pgdpo_base import train_stage1_base
        from pgdpo_with_projection import print_policy_rmse_and_samples_direct
        train_fn = train_stage1_base
        rmse_fn = print_policy_rmse_and_samples_direct
        train_kwargs = {"outdir": OUT_DIR}
        rmse_kwargs = {"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "outdir": OUT_DIR}

    elif mode_key == "base":
        from pgdpo_base import train_stage1_base, print_policy_rmse_and_samples_base
        train_fn = train_stage1_base
        rmse_fn = print_policy_rmse_and_samples_base
        train_kwargs = {"outdir": OUT_DIR}
        rmse_kwargs = {"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "outdir": OUT_DIR}

    else:  # "residual"
        from pgdpo_residual import train_residual_stage1
        from pgdpo_run import print_policy_rmse_and_samples_run
        train_fn = train_residual_stage1
        rmse_fn = print_policy_rmse_and_samples_run
        train_kwargs = {"outdir": OUT_DIR}
        rmse_kwargs = {"repeats": REPEATS, "sub_batch": SUBBATCH, "seed_eval": CRN_SEED_EU, "outdir": OUT_DIR}

    # 실행
    run_common(
        train_fn=train_fn,
        rmse_fn=rmse_fn,
        seed_train=None,  # pgdpo_base 내부에서 user seed 사용
        train_kwargs=train_kwargs,
        rmse_kwargs=rmse_kwargs,
    )

    write_text(os.path.join(OUT_DIR, "_DONE"), "ok\n")




# ---------------------------------------------------------------------
# 오케스트레이터(병렬 실행) 모드
# ---------------------------------------------------------------------

def _model_supports_residual(ROOT: str, model: str) -> bool:
    """
    tests/<model>/user_pgdpo_residual.py 존재 여부 또는
    user_pgdpo_base 내 residual 관련 심볼 존재 여부로 residual 지원 감지.
    """
    import importlib
    test_dir = os.path.join(ROOT, "tests", model)
    has_residual_file = os.path.isfile(os.path.join(test_dir, "user_pgdpo_residual.py"))

    has_sim_residual = False
    try:
        sys.path.insert(0, test_dir)
        upb = importlib.import_module("user_pgdpo_base")
        has_sim_residual = any(
            hasattr(upb, name)
            for name in (
                "simulate_residual",
                "ResidualPolicy",
                "DirectPolicyResidual",
                "ResidualDirectPolicy",
            )
        )
    except Exception:
        pass
    finally:
        try:
            sys.path.remove(test_dir)
        except ValueError:
            pass

    return bool(has_residual_file or has_sim_residual)


def orchestrate_all(
    *,
    model: str,
    gpus: List[int],
    plots_base: str,
):
    """
    4개 모드(base/run/projection/residual)를 병렬 실행.
    단, 모델이 residual을 지원하지 않으면 base/run/projection 3개만 실행.
    """
    ROOT = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_dir = os.path.join(ROOT, plots_base, model, "multi", ts)
    ensure_dir(multi_dir)

    # 모델 기능 감지
    supports_residual = _model_supports_residual(ROOT, model)

    modes = ["base", "run", "projection"]
    if supports_residual:
        modes.append("residual")

    # GPU 매핑 준비
    if len(gpus) == 0:
        if torch.cuda.is_available():
            gpus = list(range(torch.cuda.device_count()))
        else:
            gpus = [-1]  # CPU

    gpu_map = [gpus[i % len(gpus)] for i in range(len(modes))]

    # 요약 메타
    summary: Dict[str, Any] = {
        "model": model,
        "timestamp": ts,
        "orchestrator_dir": multi_dir,
        "modes": [],
        "git_commit": get_git_commit_or_none(),
        "torch_version": torch.__version__,
        "detected_support": {"residual": supports_residual},
    }

    procs = []
    log_files = []
    print("🧩 Orchestrating multi-run:")
    for i, (mode_key, gpu_id) in enumerate(zip(modes, gpu_map)):
        # 자식 커맨드 구성
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            f"--{mode_key}", model,
            "--plots", plots_base,
            "--tag", f"auto-{ts}-{mode_key}"
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if gpu_id >= 0 and torch.cuda.is_available():
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            cmd += ["--gpu", "0"]    # 자식은 자신의 0번을 사용
            gpu_label = f"cuda:{gpu_id}"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            cmd += ["--gpu", "-1"]
            gpu_label = "cpu"

        # 로그 파일
        log_path = os.path.join(multi_dir, f"{i:02d}_{mode_key}_gpu{gpu_id if gpu_id>=0 else 'cpu'}.log")
        lf = open(log_path, "w", encoding="utf-8")
        log_files.append(lf)

        print(f"  • [{mode_key}] → GPU={gpu_label}  log={log_path}")
        p = subprocess.Popen(cmd, stdout=lf, stderr=lf, env=env, cwd=ROOT)
        procs.append((mode_key, gpu_id, p, log_path))

        summary["modes"].append({
            "mode": mode_key,
            "gpu": gpu_id,
            "log": log_path,
            "pid": p.pid,
            "cmd": cmd,
        })

    # 요약 먼저 기록
    write_json(os.path.join(multi_dir, "orchestrator.json"), summary)
    write_text(os.path.join(multi_dir, "README.txt"),
               "This folder contains logs for parallel runs started by --all.\n"
               "Each child writes its outputs under plots/<model>/<mode>/<timestamp>/.\n")

    # 대기
    retcodes = {}
    for mode_key, gpu_id, p, log_path in procs:
        rc = p.wait()
        retcodes[mode_key] = rc

    # 로그 닫기
    for lf in log_files:
        try:
            lf.flush(); lf.close()
        except Exception:
            pass

    # 결과 요약 갱신
    summary["return_codes"] = retcodes
    write_json(os.path.join(multi_dir, "orchestrator_result.json"), summary)

    print("✅ All runs finished.")
    for mode_key in modes:
        rc = retcodes.get(mode_key, None)
        print(f"  - {mode_key}: return_code={rc}")



# ---------------------------------------------------------------------
# 엔트리
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="PG-DPO runner")

    # 단일/병렬 모드
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", metavar="MODEL", help="Variance-reduced simulation mode")
    mode.add_argument("--projection", metavar="MODEL", help="P-PGDPO with projection mode")
    mode.add_argument("--base", metavar="MODEL", help="Basic simulation mode")
    mode.add_argument("--residual", metavar="MODEL", help="Residual learning mode")
    mode.add_argument("--all", metavar="MODEL", help="Run all supported modes concurrently for MODEL")

    parser.add_argument("--gpu", type=str, default=None, help="GPU index or CSV (e.g., 0 or 1,2,3,4,5). For --all, CSV maps to modes.")
    parser.add_argument("--plots", type=str, default="plots", help="Base plots directory")
    parser.add_argument("--tag", type=str, default=None, help="Optional tag appended to folder name")

    args = parser.parse_args()

    # 병렬(--all) 모드
    if args.all:
        MODEL = args.all
        GPU_LIST = parse_gpu_list(args.gpu)
        orchestrate_all(model=MODEL, gpus=GPU_LIST, plots_base=args.plots)
        return

    # 단일 모드
    if args.run:
        MODE = "run"
        MODEL = args.run
    elif args.projection:
        MODE = "projection"
        MODEL = args.projection
    elif args.base:
        MODE = "base"
        MODEL = args.base
    else:
        MODE = "residual"
        MODEL = args.residual

    # 단일 모드 GPU 셀렉터 구성
    # - --gpu가 "2"면 cuda:2
    # - --gpu가 "-1"면 cpu
    # - None이면 자동
    gpu_sel = "auto"
    if args.gpu is not None:
        gstr = str(args.gpu).strip()
        if gstr == "-1":
            gpu_sel = "cpu"
        else:
            try:
                idx = int(gstr)
                gpu_sel = f"cuda:{idx}"
            except Exception:
                # CSV가 들어왔을 수 있음 -> 첫 번째만 사용
                lst = parse_gpu_list(gstr)
                if len(lst) > 0:
                    gpu_sel = f"cuda:{lst[0]}"
                else:
                    gpu_sel = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        gpu_sel = "cuda" if torch.cuda.is_available() else "cpu"

    single_run(
        mode_key=MODE,
        model=MODEL,
        gpu_sel=gpu_sel,
        plots_base=args.plots,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()