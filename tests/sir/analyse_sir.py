# tests/sir/analyse_sir.py (개선된 버전)
# 역할: 학습된 SIR 정책 모델을 불러와 상태 공간 히트맵을 생성합니다.
# 사용법: python3 tests/sir/analyse_sir.py <--mode>
# 예시: python3 tests/sir/analyse_sir.py --run

import torch
import os
import sys
import argparse # ✨ [추가] 커맨드 라인 인자 처리를 위한 모듈

def analyse(mode_key: str, d_value: int, k_value: int):
    """
    지정된 모드(mode_key)의 최신 학습 결과를 분석합니다.
    """
    # --- 경로 설정 (중요) ---
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CORE_PATH = os.path.join(ROOT, "core")
    TEST_PATH = os.path.join(ROOT, "tests", "sir")
    sys.path.insert(0, CORE_PATH)
    sys.path.insert(0, TEST_PATH)

    from user_pgdpo_base import DirectPolicy, T
    from viz import save_policy_heatmap

    # 1. 학습된 모델 불러오기
    # ✨ [수정] 인자로 받은 mode_key를 사용하여 동적으로 경로 생성
    param_str = f"d{d_value}_k{k_value}"
    outdir = os.path.join(ROOT, "plots", "sir", mode_key, param_str, "latest")
    
    model_path = os.path.join(outdir, "policy.pt") 
    
    if not os.path.exists(model_path):
        print(f"오류: '{outdir}' 폴더에서 학습된 모델 파일(policy.pt)을 찾을 수 없습니다.")
        print(f"팁: 'python3 run.py sir --{mode_key}'를 먼저 실행했는지 확인해주세요.")
        return

    print(f"'{model_path}'에서 '{mode_key}' 모드의 정책 모델을 불러옵니다...")
    policy = DirectPolicy()
    policy.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    policy.eval()

    # 2. 시간대별로 정책 히트맵 생성
    times_to_plot = [0.1, T / 2, T - 0.1]
    print(f"시간대 {times_to_plot}에 대한 정책 히트맵을 생성합니다...")
    
    for t in times_to_plot:
        save_policy_heatmap(
            policy=policy, t=t,
            s_range=(0.0, 1.5), i_range=(0.0, 0.5),
            outdir=outdir,
            fname=f"sir_policy_heatmap_t_{t:.2f}.png"
        )

    print(f"분석이 완료되었습니다. 결과는 '{outdir}' 폴더에 저장되었습니다.")

# ✨ [추가] run.py와 유사한 커맨드 라인 인터페이스
def main():
    parser = argparse.ArgumentParser(
        description="Analyse a trained SIR model and generate policy heatmaps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 어떤 모드를 분석할지 선택하는 옵션 (하나만 선택 가능, 필수)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--run", action='store_true', help="Analyse the latest 'run' mode result.")
    mode_group.add_argument("--projection", action='store_true', help="Analyse the latest 'projection' mode result.")
    mode_group.add_argument("--base", action='store_true', help="Analyse the latest 'base' mode result.")
    mode_group.add_argument("--residual", action='store_true', help="Analyse the latest 'residual' mode result.")
    
    # 분석할 모델의 d, k 값 (run.py와 일치해야 함)
    parser.add_argument("-d", type=int, default=3, help="Dimension 'd' of the model to analyse.")
    parser.add_argument("-k", type=int, default=0, help="Dimension 'k' of the model to analyse (always 0 for SIR).")
    
    args = parser.parse_args()
    
    # 선택된 모드 확인
    if args.run: mode = "run"
    elif args.projection: mode = "projection"
    elif args.base: mode = "base"
    else: mode = "residual"
    
    # 분석 함수 호출
    analyse(mode_key=mode, d_value=args.d, k_value=args.k)

if __name__ == "__main__":
    main()