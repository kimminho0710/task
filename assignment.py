import math, re
import pandas as pd
from collections import defaultdict, Counter
from ortools.sat.python import cp_model


# 공통 함수
    # 불리언 처리
def as_yes(v) -> bool:
    return str(v).strip().lower() in {"y", "yes", "true", "1"}

    # 학생 아이디 처리(오류 해결용)
def norm_id(v) -> str:
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        try: return str(int(float(s)))
        except: pass
    digits = re.findall(r"\d+", s)
    return "".join(digits) if digits else s

    # 관계 처리(오류 해결용)
def parse_ids(cell) -> list[str]:
    if pd.isna(cell): return []
    toks = re.split(r"[,\s;/\n\r]+", str(cell).strip())
    out = []
    for t in toks:
        t = norm_id(t)
        if t: out.append(t)
    return out

    # 속성 목표치 계산함수
def targets_by_capacity(total_ones: int, class_sizes: list[int]) -> list[int]:
    if total_ones <= 0: return [0]*len(class_sizes)
    total_cap = sum(class_sizes)
    raw = [cs * total_ones / total_cap for cs in class_sizes]
    flo = [math.floor(x) for x in raw]
    rem = total_ones - sum(flo)
    order = sorted(range(len(class_sizes)), key=lambda c: raw[c]-flo[c], reverse=True)
    for k in range(rem): flo[order[k]] += 1
    return flo


# 데이터 정리 및 전처리
def load_student_data(input_csv: str):
    # 데이터 받아오기
    df = pd.read_csv(input_csv)
    for col, default in [
        ("클럽",""),("24년 학급",""),("좋은관계",""),("나쁜관계",""),
        ("Leadership",""),("Piano",""),("비등교",""),("운동선호",""),
    ]:
        if col not in df.columns: df[col] = default

    # 문자열 입력값 전처리
    df["id"] = df["id"].map(norm_id)
    ids = df["id"].tolist()
    id_to_idx = {sid:i for i,sid in enumerate(ids)}

    sex  = df["sex"].astype(str).str.lower().tolist()
    score= df["score"].fillna(0).astype(int).tolist()
    club = df["클럽"].fillna("").astype(str).tolist()
    prev = df["24년 학급"].fillna("").astype(str).tolist()

    # 속성 불리언 처리
    leader01 = [1 if as_yes(v) else 0 for v in df["Leadership"]]
    piano01  = [1 if as_yes(v) else 0 for v in df["Piano"]]
    nonatt01 = [1 if as_yes(v) else 0 for v in df["비등교"]]
    sport01  = [1 if as_yes(v) else 0 for v in df["운동선호"]]

    # 관계쌍 정리 및 저장
    good_pairs, bad_pairs = set(), set()
    unknown_good, unknown_bad = set(), set()
    for i,row in df.iterrows():
        for sid in parse_ids(row["좋은관계"]):
            if sid in id_to_idx and sid != df.at[i,"id"]:
                a,b = sorted([i, id_to_idx[sid]])
                good_pairs.add((a,b))
            else:
                unknown_good.add((df.at[i,"id"], sid))
        for sid in parse_ids(row["나쁜관계"]):
            if sid in id_to_idx and sid != df.at[i,"id"]:
                a,b = sorted([i, id_to_idx[sid]])
                bad_pairs.add((a,b))
            else:
                unknown_bad.add((df.at[i,"id"], sid))

        # 충돌시 good 우선
    bad_pairs -= good_pairs

    # 전년도 같은 반 쌍 묶기(추후 패널티 요소)
    same_prev = defaultdict(list)
    for i,lab in enumerate(prev): same_prev[lab].append(i)
    prev_pairs = set()
    for lab, members in same_prev.items():
        for a in range(len(members)):
            for b in range(a+1, len(members)):
                prev_pairs.add((members[a], members[b]))

    # 딕셔너리 묶기
    parsed = dict(
        df=df, ids=ids, id_to_idx=id_to_idx,
        sex=sex, score=score, club=club, prev=prev,
        leader01=leader01, piano01=piano01, nonatt01=nonatt01, sport01=sport01,
        good_pairs=good_pairs, bad_pairs=bad_pairs, prev_pairs=prev_pairs,
        unknown_good=unknown_good, unknown_bad=unknown_bad
    )
    return parsed


# solver 모델 입력
def solve_class_assignment(parsed, class_sizes, time_limit=60.0, num_workers=8,
                           use_prev_label_balance=True, leader_soft_backup=True):

    # 변수 지정
    df = parsed["df"]
    N = len(df)
    C = len(class_sizes)
    sex = parsed["sex"]
    score = parsed["score"]
    club = parsed["club"]
    prev = parsed["prev"]
    leader01 = parsed["leader01"]
    piano01 = parsed["piano01"]
    nonatt01 = parsed["nonatt01"]
    sport01 = parsed["sport01"]
    good_pairs = parsed["good_pairs"]
    bad_pairs = parsed["bad_pairs"]
    prev_pairs = parsed["prev_pairs"]

    # 모델 선언 및 하드 제약 입력
    model = cp_model.CpModel()
    x = {(i,c): model.NewBoolVar(f"x_{i}_{c}") for i in range(N) for c in range(C)}

        # 1) 학생=딱 1반
    for i in range(N): model.Add(sum(x[i,c] for c in range(C)) == 1)
        # 2) 반 정원
    for c in range(C): model.Add(sum(x[i,c] for i in range(N)) == class_sizes[c])
        # 3) 나쁜관계: 같은 반 금지
    for (i,j) in bad_pairs:
        for c in range(C): model.Add(x[i,c] + x[j,c] <= 1)
        # 4) 좋은관계: 같은 반 강제
    for (i,j) in good_pairs:
        for c in range(C): model.Add(x[i,c] == x[j,c])

    terms = [] # 소프트 제약 벌점 리스트

        # 5) 리더 ≥1 (소프트로 백업)
    leader_idx = [i for i,v in enumerate(leader01) if v==1]
    if leader_idx and (len(leader_idx) >= C or not leader_soft_backup):
        for c in range(C): model.Add(sum(x[i,c] for i in leader_idx) >= 1)
    elif leader_idx and leader_soft_backup:
        for c in range(C):
            cnt = model.NewIntVar(0, len(leader_idx), f"lead_cnt_{c}")
            need = 1
            deficit = model.NewIntVar(0, need, f"lead_def_{c}")
            model.Add(cnt == sum(x[i,c] for i in leader_idx))
            model.Add(deficit >= need - cnt)
            terms.append(20 * deficit)

    # 소프트 제약 공통함수 선언
    def add_balance_binary(name, arr01, weight):
        total = sum(arr01)
        if total == 0: return
        targets = targets_by_capacity(total, class_sizes)
        idx1 = [i for i,v in enumerate(arr01) if v]
        for c in range(C):
            cnt = model.NewIntVar(0, total, f"{name}_cnt_{c}")
            model.Add(cnt == sum(x[i,c] for i in idx1))
            dev = model.NewIntVar(0, total, f"{name}_dev_{c}")
            model.Add(dev >= cnt - targets[c])
            model.Add(dev >= targets[c] - cnt)
            terms.append(weight * dev)

    # 소프트 제약 입력
        # 클럽 제약
    add_balance_binary("male", [1 if s=="boy" else 0 for s in sex], weight=6)
    add_balance_binary("piano", piano01, weight=3)
    add_balance_binary("nonatt", nonatt01, weight=6)
    add_balance_binary("sport", sport01, weight=3)

    for lab in set(club):
        add_balance_binary(f"club_{lab}", [1 if lab==v else 0 for v in club], weight=3)

    if use_prev_label_balance:
        for lab in set(prev):
            add_balance_binary(f"prevlab_{lab}", [1 if lab==v else 0 for v in prev], weight=3)

        # 전년도 같은반 쌍 제약
    for (i,j) in prev_pairs:
        for c in range(C):
            y = model.NewBoolVar(f"prevpair_{i}_{j}_{c}")
            model.Add(x[i,c] + x[j,c] - 1 <= y)
            model.Add(y <= x[i,c]); model.Add(y <= x[j,c])
            terms.append(1 * y)

        # 성적 합 균형
    total_score = sum(score); target_score = total_score // C
    for c in range(C):
        s_c = model.NewIntVar(0, total_score, f"score_sum_{c}")
        model.Add(s_c == sum(score[i]*x[i,c] for i in range(N)))
        d = model.NewIntVar(0, total_score, f"score_dev_{c}")
        model.Add(d >= s_c - target_score); model.Add(d >= target_score - s_c)
        terms.append(10 * d)

    # solver 문제 해결 코드
    model.Minimize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)

    # 해 구하기 실패시
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": solver.StatusName(status), "assign": None}

    # 해 추출
    assign = [None]*N
    for i in range(N):
        for c in range(C):
            if solver.Value(x[i,c]) == 1:
                assign[i] = c; break

    return {
        "status": solver.StatusName(status),
        "objective": solver.ObjectiveValue(),
        "assign": assign,
    }


#통합 함수
def assign_classes_from_csv(input_csv, output_csv, class_sizes, time_limit=60.0, num_workers=8):
    parsed = load_student_data(input_csv)
    result = solve_class_assignment(parsed, class_sizes, time_limit=time_limit, num_workers=num_workers)
    if result["assign"] is None:
        print(f"[해 실패] Status={result['status']}")
        return {"status": result["status"], "objective": None, "class_counts": {}}

    assign = result["assign"]
    
    # 결과 저장
    df_out = parsed["df"].copy()
    df_out["배정반"] = assign
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # 최종 상태 출력
    counts = Counter(assign)
    return {"status": result["status"], "objective": result.get("objective"), "class_counts": dict(counts)}


#실행
if __name__ == "__main__":
    info = assign_classes_from_csv(
        input_csv="students.csv",
        output_csv="assigned_classes.csv",
        class_sizes=[33,33,33,33,34,34],
        time_limit=120.0,
    )
    print(info)
