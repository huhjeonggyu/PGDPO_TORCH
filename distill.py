# distill.py
# 역할: 저장된 teacher_dataset.pt를 불러와 "학생" 정책을 지도학습합니다.

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- 증류 하이퍼파라미터 ---
DISTILL_EPOCHS = 100
DISTILL_BATCH_SIZE = 4096
DISTILL_LR = 1e-4

def train_distilled_policy(dataset_path: str, model_save_path: str):
    """학생 정책을 지도학습하고, 학습된 모델을 저장합니다."""
    
    print(f"[Distill] Loading teacher dataset from: {dataset_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    data = torch.load(dataset_path)
    states, u_teacher = data['states'], data['u_teacher']
    dataset = TensorDataset(states, u_teacher)
    loader = DataLoader(dataset, batch_size=DISTILL_BATCH_SIZE, shuffle=True, num_workers=4)

    # 학생 정책 초기화 (DirectPolicy 구조를 가져와야 함)
    # 이 부분을 위해 user_pgdpo_base를 import할 수 있도록 경로 설정이 필요합니다.
    # 예시: from tests.ko_nd.user_pgdpo_base import DirectPolicy
    # 이 부분은 모델마다 달라지므로, argparse로 모델 이름을 받아 동적으로 import 하는 것이 좋습니다.
    
    # 임시: ko_nd 모델을 하드코딩
    from tests.ko_nd.user_pgdpo_base import DirectPolicy
    student_policy = DirectPolicy().to(device)
    student_policy.train()

    optimizer = torch.optim.Adam(student_policy.parameters(), lr=DISTILL_LR)
    loss_fn = nn.MSELoss()
    
    print("[Distill] Starting training...")
    for epoch in range(1, DISTILL_EPOCHS + 1):
        for batch_states, batch_u_teacher in loader:
            batch_states, batch_u_teacher = batch_states.to(device), batch_u_teacher.to(device)
            
            # DirectPolicy의 forward 입력 형식에 맞게 텐서 재구성
            # (이 부분은 모델의 입력 차원에 따라 달라지므로 주의가 필요합니다)
            dim_x = 1 # X는 1차원
            states_dict = {
                'X': batch_states[:, :dim_x],
                'TmT': batch_states[:, dim_x:dim_x+1],
                'Y': batch_states[:, dim_x+1:]
            }

            optimizer.zero_grad()
            u_student = student_policy(**states_dict)
            loss = loss_fn(u_student, batch_u_teacher)
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"  - Epoch {epoch:03d}, MSE Loss: {loss.item():.8f}")

    # 학습된 모델 저장
    torch.save(student_policy.state_dict(), model_save_path)
    print(f"[Distill] Training complete. Distilled policy saved to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG-DPO Knowledge Distillation Trainer")
    parser.add_argument("teacher_dataset", type=str, help="Path to the teacher_dataset.pt file.")
    parser.add_argument("output_model", type=str, help="Path to save the trained distilled model.")
    args = parser.parse_args()
    
    # PYTHONPATH에 core와 tests/ko_nd 경로 추가 (실행 환경에 맞게 조정 필요)
    import sys
    sys.path.append('./core')
    sys.path.append('./tests/ko_nd')

    train_distilled_policy(args.teacher_dataset, args.output_model)