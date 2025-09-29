import math, re
import pandas as pd
from collections import defaultdict, Counter
from ortools.sat.python import cp_model


# ---------- 공통 유틸 ----------
def as_yes(v) -> bool:
    return str(v).strip().lower() in {"y", "yes", "true", "1"}

def norm_id(v) -> str:
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        try: return str(int(float(s)))
        except: pass
    digits = re.findall(r"\d+", s)
    return "".join(digits) if digits else s

def parse_ids(cell) -> list[str]:
    if pd.isna(cell): return []
    toks = re.split(r"[,\s;/\n\r]+", str(cell).strip())
    out = []
    for t in toks:
        t = norm_id(t)
        if t: out.append(t)
    return out

def targets_by_capacity(total_ones: int, class_sizes: list[int]) -> list[int]:
    if total_ones <= 0: return [0]*len(class_sizes)
    total_cap = sum(class_sizes)
    raw = [cs * total_ones / total_cap for cs in class_sizes]
    flo = [math.floor(x) for x in raw]
    rem = total_ones - sum(flo)
    order = sorted(range(len(class_sizes)), key=lambda c: raw[c]-flo[c], reverse=True)
    for k in range(rem): flo[order[k]] += 1
    return flo


# ---------- 데이터 파싱 ----------
def load_student_data(input_csv: str):
    df = pd.read_csv(input_csv)
    # 필수 컬럼 보정
    for col, default in [
        ("클럽",""),("24년 학급",""),("좋은관계",""),("나쁜관계",""),
        ("Leadership",""),("Piano",""),("비등교",""),("운동선호",""),
    ]:
        if col not in df.columns: df[col] = default

    df["id"] = df["id"].map(norm_id)
    ids = df["id"].tolist()
    id_to_idx = {sid:i for i,sid in enumerate(ids)}

    sex  = df["sex"].astype(str).str.lower().tolist()
    score= df["score"].fillna(0).astype(int).tolist()
    club = df["클럽"].fillna("").astype(str).tolist()
    prev = df["24년 학급"].fillna("").astype(str).tolist()

    leader01 = [1 if as_yes(v) else 0 for v in df["Leadership"]]
    piano01  = [1 if as_yes(v) else 0 for v in df["Piano"]]
    nonatt01 = [1 if as_yes(v) else 0 for v in df["비등교"]]
    sport01  = [1 if as_yes(v) else 0 for v in df["운동선호"]]

    # 관계쌍
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

    # 전년도 같은 반 쌍(참고용 soft 평가)
    same_prev = defaultdict(list)
    for i,lab in enumerate(prev): same_prev[lab].append(i)
    prev_pairs = set()
    for lab, members in same_prev.items():
        for a in range(len(members)):
            for b in range(a+1, len(members)):
                prev_pairs.add((members[a], members[b]))

    parsed = dict(
        df=df, ids=ids, id_to_idx=id_to_idx,
        sex=sex, score=score, club=club, prev=prev,
        leader01=leader01, piano01=piano01, nonatt01=nonatt01, sport01=sport01,
        good_pairs=good_pairs, bad_pairs=bad_pairs, prev_pairs=prev_pairs,
        unknown_good=unknown_good, unknown_bad=unknown_bad
    )
    return parsed


# ---------- 모델 & 풀이 ----------
def solve_class_assignment(parsed, class_sizes, time_limit=60.0, num_workers=8,
                           use_prev_label_balance=True, leader_soft_backup=True):
    df = parsed["df"]; N = len(df); C = len(class_sizes)
    sex = parsed["sex"]; score=parsed["score"]; club=parsed["club"]; prev=parsed["prev"]
    leader01=parsed["leader01"]; piano01=parsed["piano01"]
    nonatt01=parsed["nonatt01"]; sport01=parsed["sport01"]
    good_pairs=parsed["good_pairs"]; bad_pairs=parsed["bad_pairs"]; prev_pairs=parsed["prev_pairs"]

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

    # 5) 리더 ≥1 (백업: 소프트)
    leader_idx = [i for i,v in enumerate(leader01) if v==1]
    terms = []
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

    # 소프트 균형 공통
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

    add_balance_binary("male", [1 if s=="boy" else 0 for s in sex], weight=6)
    add_balance_binary("piano", piano01, weight=4)
    add_balance_binary("nonatt", nonatt01, weight=6)
    add_balance_binary("sport", sport01, weight=3)

    for lab in set(club):
        add_balance_binary(f"club_{lab}", [1 if lab==v else 0 for v in club], weight=2)

    if use_prev_label_balance:
        for lab in set(prev):
            add_balance_binary(f"prevlab_{lab}", [1 if lab==v else 0 for v in prev], weight=2)

    # 전년도 같은반 쌍 소량 페널티
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

    model.Minimize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)

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


# ---------- 검증 함수들 ----------
def check_unique_assignment(assign):
    ok = all(a is not None for a in assign)
    msg = "OK" if ok else "미배정 학생 존재"
    return ok, {"unassigned": sum(1 for a in assign if a is None)}, msg

def check_class_sizes(assign, class_sizes):
    count = Counter(assign)
    diffs = {c: count.get(c,0) - class_sizes[c] for c in range(len(class_sizes))}
    ok = all(v==0 for v in diffs.values())
    return ok, {"counts": dict(count), "diffs": diffs}, "OK" if ok else "정원 불일치 존재"

def check_pairs(assign, ids, pairs, expect_same: bool):
    viol = []
    for (i,j) in pairs:
        same = (assign[i] == assign[j])
        if expect_same and not same:
            viol.append((ids[i], ids[j], assign[i], assign[j]))
        if (not expect_same) and same:
            viol.append((ids[i], ids[j], assign[i]))
    ok = len(viol)==0
    return ok, {"violations": viol[:20], "count": len(viol)}, "OK" if ok else f"위배 {len(viol)}건"

def check_leaders(assign, leader01, C):
    per = Counter(c for i,c in enumerate(assign) if leader01[i]==1)
    missing = [c for c in range(C) if per.get(c,0)==0]
    ok = len(missing)==0
    return ok, {"per_class": dict(per), "missing_classes": missing}, "OK" if ok else f"부족 {len(missing)}개 반"

def metric_binary_balance(assign, arr01, class_sizes, name):
    total = sum(arr01)
    targets = targets_by_capacity(total, class_sizes)
    C = len(class_sizes)
    counts = [0]*C
    for i,a in enumerate(assign):
        if arr01[i]: counts[a]+=1
    devs = [abs(counts[c]-targets[c]) for c in range(C)]
    return {
        "name": name,
        "total": total,
        "targets": targets,
        "counts": counts,
        "sum_dev": sum(devs),
        "max_dev": max(devs) if devs else 0,
    }

def metric_category_balance(assign, labels, class_sizes, name):
    labs = sorted(set(labels))
    rows = []
    for lab in labs:
        arr01 = [1 if lab==v else 0 for v in labels]
        rows.append(metric_binary_balance(assign, arr01, class_sizes, f"{name}:{lab}"))
    return rows

def metric_score(assign, scores, class_sizes):
    C = len(class_sizes)
    sums = [0]*C
    for i,a in enumerate(assign): sums[a]+=scores[i]
    target = sum(scores)//C
    devs = [abs(s-target) for s in sums]
    return {"target": target, "sum_per_class": sums, "sum_dev_total": sum(devs), "max_dev": max(devs)}

def print_validation_report(parsed, assign, class_sizes):
    ids = parsed["ids"]; C=len(class_sizes)
    print("\n===== 제약 검증 리포트 =====")
    # 하드 제약
    ok1,info1,msg1 = check_unique_assignment(assign);        print(f"[하드] 단일 배정      : {msg1}", info1)
    ok2,info2,msg2 = check_class_sizes(assign, class_sizes); print(f"[하드] 정원 일치      : {msg2}", info2["diffs"])
    ok3,info3,msg3 = check_pairs(assign, ids, parsed["bad_pairs"], expect_same=False)
    print(f"[하드] 나쁜관계 분리   : {msg3} (예시:{info3['violations'][:3]})")
    ok4,info4,msg4 = check_pairs(assign, ids, parsed["good_pairs"], expect_same=True)
    print(f"[하드] 좋은관계 동행   : {msg4} (예시:{info4['violations'][:3]})")
    ok5,info5,msg5 = check_leaders(assign, parsed["leader01"], C)
    print(f"[하드] 리더 ≥1         : {msg5} (per_class:{info5['per_class']})")

    # 파싱 경고
    if parsed["unknown_good"]:
        print(f"[경고] 좋은관계 ID 미매칭 {len(parsed['unknown_good'])}건 예:{list(parsed['unknown_good'])[:5]}")
    if parsed["unknown_bad"]:
        print(f"[경고] 나쁜관계 ID 미매칭 {len(parsed['unknown_bad'])}건 예:{list(parsed['unknown_bad'])[:5]}")

    # 소프트/지표
    bin_metrics = []
    bin_metrics.append(metric_binary_balance(assign, [1 if s=="boy" else 0 for s in parsed["sex"]], class_sizes, "male"))
    bin_metrics.append(metric_binary_balance(assign, parsed["piano01"], class_sizes, "piano"))
    bin_metrics.append(metric_binary_balance(assign, parsed["nonatt01"], class_sizes, "nonatt"))
    bin_metrics.append(metric_binary_balance(assign, parsed["sport01"], class_sizes, "sport"))
    print("\n[소프트] 이진 균형 지표(합Dev, 최대Dev):")
    for m in bin_metrics:
        print(f" - {m['name']:7s}  sum_dev={m['sum_dev']}, max_dev={m['max_dev']}  targets={m['targets']} counts={m['counts']}")

    print("\n[소프트] 클럽 균형 지표(각 라벨):")
    for row in metric_category_balance(assign, parsed["club"], class_sizes, "club"):
        print(f" - {row['name']:12s} sum_dev={row['sum_dev']}, max_dev={row['max_dev']}")

    print("\n[소프트] 전년도 라벨 균형 지표:")
    for row in metric_category_balance(assign, parsed["prev"], class_sizes, "prev"):
        print(f" - {row['name']:12s} sum_dev={row['sum_dev']}, max_dev={row['max_dev']}")

    sc = metric_score(assign, parsed["score"], class_sizes)
    print(f"\n[소프트] 성적 합 균형: target_per_class={sc['target']}, sum_dev_total={sc['sum_dev_total']}, max_dev={sc['max_dev']}")
    print(" sum_per_class:", sc["sum_per_class"])

    # 전년도 같은반 쌍 카운트(페널티 참고)
    vp = [(parsed["ids"][i], parsed["ids"][j], assign[i]) for (i,j) in parsed["prev_pairs"] if assign[i]==assign[j]]
    print(f"\n[지표] 전년도 같은반 쌍이 이번에도 동일반인 경우: {len(vp)}건 (예시:{vp[:5]})")
    print("===== 끝 =====\n")


# ---------- 통합 함수 ----------
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

    # 리포트
    print(f"[해 상태] {result['status']}, objective={result.get('objective')}")
    print_validation_report(parsed, assign, class_sizes)

    counts = Counter(assign)
    return {"status": result["status"], "objective": result.get("objective"), "class_counts": dict(counts)}


# ---------- 사용 예시 ----------
if __name__ == "__main__":
    # 예: 6개 반(33,33,33,33,34,34)
    info = assign_classes_from_csv(
        input_csv="students.csv",
        output_csv="assigned_classes.csv",
        class_sizes=[33,33,33,33,34,34],
        time_limit=120.0,
    )
    print(info)
