import os
import json
import torch
from tqdm import tqdm

###############################################################################
# 1) 한 비디오(임베딩)에서 '연속 프레임 거리'의 (합, min, max, 개수) 전부 구하기
###############################################################################
def compute_distance_stats_for_video(embeddings):
    """
    T개의 프레임 임베딩이 주어졌을 때, 연속 프레임 (i, i+1) 간의 L2 거리를
    한 루프에서 모두 구한다:
      - sum_d: 모든 연속 거리의 합
      - min_d: 그 중 최소값
      - max_d: 그 중 최대값
      - count_d: 거리 개수 (T-1)

    T < 2 면 거리 계산 불가능 -> (sum=0, min=inf, max=-inf, count=0)
    """
    T = embeddings.size(0)
    if T < 2:
        return 0.0, float('inf'), float('-inf'), 0

    sum_d = 0.0
    min_d = float('inf')
    max_d = float('-inf')
    count_d = T - 1

    for i in range(T - 1):
        dist = torch.norm(embeddings[i+1] - embeddings[i]).item()
        sum_d += dist
        if dist < min_d:
            min_d = dist
        if dist > max_d:
            max_d = dist

    return sum_d, min_d, max_d, count_d


###############################################################################
# 2) Dominant Score 계산 예시 (전역 W 사용)
###############################################################################
def calculate_dominant_scores(embeddings, W):
    """
    W 내 코사인 유사도 합을 Dominant Score로 정의
    (프레임별로 동일한 W)
    """
    device = embeddings.device
    T, dim = embeddings.shape
    norms = torch.norm(embeddings, dim=1)
    dom_scores = torch.zeros(T, device=device)

    for t in range(T):
        start = max(0, t - W)
        end = min(T, t + W + 1)
        local_emb = embeddings[start:end]
        cs = torch.matmul(embeddings[t], local_emb.T)
        denom = norms[t] * norms[start:end] + 1e-8
        cs = cs / denom
        dom_scores[t] = cs.sum()

    return dom_scores


###############################################################################
# 3) Dominant Score 기반 Greedy 샘플링
###############################################################################
def select_frames_greedy(dominant_scores, W, N):
    T = len(dominant_scores)
    idx_score_list = sorted(
        enumerate(dominant_scores.tolist()), 
        key=lambda x: x[1], 
        reverse=True
    )
    selected = []
    candidate = set(range(T))

    for idx, score in idx_score_list:
        if idx not in candidate:
            continue
        selected.append(idx)

        # 근방 W 제거
        start = max(0, idx - W)
        end = min(T, idx + W + 1)
        for i in range(start, end):
            if i in candidate:
                candidate.remove(i)

        if len(selected) >= N:
            break
    
    # 혹시 N 미달 시, 남은 후보 중 점수 높은 순 계속 추가
    if len(selected) < N:
        need = N - len(selected)
        remain_list = [(i, dominant_scores[i].item()) for i in candidate]
        remain_sorted = sorted(remain_list, key=lambda x: x[1], reverse=True)
        selected += [tup[0] for tup in remain_sorted[:need]]

    selected.sort()
    return selected


###############################################################################
# 4) 메인: 영상마다 각각 "로컬 min~max"로부터 N 결정
###############################################################################
def main():
    frame_root = "/home/user16/HT/VideoTree/videoMME_features_long"
    
    video_list = os.listdir(frame_root)

    MIN_FRAMES = 300
    MAX_FRAMES = 400

    results = []
    for vf in tqdm(video_list):
        vid_id = os.path.splitext(vf)[0]
        
        emb_path = os.path.join(frame_root, vf)
        embeddings = torch.load(emb_path, map_location='cuda')  # [T, dim]
        T = embeddings.size(0)

        # (1) 연속 프레임 거리 통계
        sum_d, local_min, local_max, count_d = compute_distance_stats_for_video(embeddings)
        if count_d <= 0:
            # 프레임 수가 1 이하인 경우 -> 그냥 전체(=0 or 1) 사용
            # 혹은 min_frames에 맞추기 등
            N = MIN_FRAMES
        else:
            avg_dist = sum_d / count_d

            # 혹시 min==max인 경우(= 모든 dist 동일) -> 분모=0 방지
            if abs(local_max - local_min) < 1e-8:
                # avg_dist도 결국 local_min=local_max와 같을 것
                # => ratio를 0이라 보고 => N=min_frames로 둘 수도 있고,
                #    아니면 그냥 중간값(0.5)라 가정할 수도 있음.
                ratio = 0.0  
            else:
                ratio = (avg_dist - local_min) / (local_max - local_min)
                # clamp (혹시 -0.1~1.1처럼 튀어나올 수도)
                ratio = max(0.0, min(1.0, ratio))
            
            # (2) N 결정(선형 보간)
            N_float = MIN_FRAMES + ratio * (MAX_FRAMES - MIN_FRAMES)
            N = round(N_float)
            if N < MIN_FRAMES:
                N = MIN_FRAMES
            elif N > MAX_FRAMES:
                N = MAX_FRAMES
        
        print(T, N, sum_d, local_min, local_max, count_d, ratio)
        # 만약 T < N 이면 그냥 전부 골라도 됨
        if T <= N:
            sampled_frames = list(range(T))
            W = None
        else:
            # (3) W 계산 (예: round(T / (2N)))
            W = int(round(T / (2.0 * N))) if N > 0 else 1
            W = max(W, 1)

            # (4) Dominant Score
            dom_scores = calculate_dominant_scores(embeddings, W)

            # (5) Greedy
            sampled_frames = select_frames_greedy(dom_scores, W, N)
        
        # 결과 저장
        results.append({
            "name": vid_id,
            "sorted_values": sampled_frames,
            'window_size': W
        })
    
    # 최종 저장
    out_path = "/home/user16/HT/Video/new/mdf_sampled_frames_localminmax_mme_min10_max50.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("처리 완료:", out_path)

if __name__ == "__main__":
    main()
