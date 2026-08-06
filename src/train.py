#!/usr/bin/env python
"""
Q-Selector 优化版训练 (Optimized Training)
=============================================
优化内容：
1. 新增8个电路特征（gate_density, gate_diversity等）
2. 新增13个电路-后端交互特征（cum_cnot_error, noise_budget等）
3. 特征重要性阈值 0.005
4. 倾向性得分权重 0.20（优化后）
5. 保持RF模型不变（n_estimators=100）
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.unicode_minus'] = False

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = BASE_DIR / 'data' / 'MQTBench_578_analysis_results.csv'
QASM_FOLDER = BASE_DIR / 'data' / 'QASM'
BACKEND_JSON_FOLDER = BASE_DIR / 'data' / 'backend_json'

print("=" * 70)
print("Q-Selector 优化版训练 (RF + 增强特征 + 阈值0.005 + 权重0.20)")
print("=" * 70)

# ============================================================================
# 1. 数据加载
# ============================================================================
df = pd.read_csv(DATA_CSV)
df['FOM'] = df['Hellinger_Fidelity'] * 10
print(f"数据: {len(df)} 行, {df['Circuit'].nunique()} 电路, {df['Backend'].nunique()} 后端")

# ============================================================================
# 2. 后端特征
# ============================================================================
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

backend_cache = {}
coupling_maps = {}
json_folder = Path(BACKEND_JSON_FOLDER)
conf_files, props_files = {}, {}
for f in json_folder.glob("*.json"):
    name = f.stem
    if name.startswith("conf_"):
        conf_files[name.replace("conf_", "").lower()] = f
    elif name.startswith("props_"):
        props_files[name.replace("props_", "").lower()] = f

for key in set(conf_files.keys()) & set(props_files.keys()):
    conf = load_json(conf_files[key])
    props = load_json(props_files[key])
    if conf is None:
        continue
    bk_name = conf.get('name', key)

    cm = conf.get('coupling_map', [])
    edges = set()
    for e in cm:
        u, v = e[0], e[1]
        edges.add((u, v) if u < v else (v, u))
    coupling_maps[bk_name] = edges

    feat = {}
    feat['backend_num_qubits'] = conf.get('num_qubits', 0)
    nq = feat['backend_num_qubits']
    ne = len(edges)
    max_e = nq * (nq - 1) // 2
    feat['num_edges'] = ne
    feat['connectivity_density'] = ne / max_e if max_e > 0 else 0
    feat['avg_degree'] = 2 * ne / nq if nq > 0 else 0
    feat['_coupling_set'] = edges

    for err_type, keys in [('single_qubit_errors',
                             ['s1_error_mean', 's1_error_std', 's1_error_max', 's1_error_min']),
                            ('two_qubit_errors',
                             ['cnot_error_mean', 'cnot_error_std', 'cnot_error_max', 'cnot_error_min'])]:
        vals = []
        if props and err_type in props:
            for v in props[err_type].values():
                if isinstance(v, dict):
                    for gate, e_ in v.items():
                        if isinstance(e_, (int, float)) and e_ > 0:
                            vals.append(e_)
                elif isinstance(v, (int, float)):
                    vals.append(v)
        for k, fn in zip(keys, [np.mean, np.std, np.max, np.min]):
            feat[k] = fn(vals) if vals else 0

    ro_vals = []
    if props and 'readout_errors' in props:
        rd = props['readout_errors']
        if isinstance(rd, list):
            ro_vals = [e for e in rd if isinstance(e, (int, float))]
        elif isinstance(rd, dict):
            ro_vals = [e for e in rd.values() if isinstance(e, (int, float))]
    for k, fn in zip(['ro_error_mean', 'ro_error_std', 'ro_error_max', 'ro_error_min'],
                     [np.mean, np.std, np.max, np.min]):
        feat[k] = fn(ro_vals) if ro_vals else 0

    if props and 't1_t2' in props:
        t1v, t2v = [], []
        for t in props['t1_t2']:
            if isinstance(t, list) and len(t) >= 2:
                if isinstance(t[0], (int, float)):
                    t1v.append(t[0])
                if isinstance(t[1], (int, float)):
                    t2v.append(t[1])
        if t1v:
            feat['t1_mean'] = np.mean(t1v)
            feat['t1_std'] = np.std(t1v)
        if t2v:
            feat['t2_mean'] = np.mean(t2v)
            feat['t2_std'] = np.std(t2v)

    backend_cache[bk_name] = feat

all_backends = list(backend_cache.keys())
BACKEND_FEATS = [f for f in [
    'backend_num_qubits', 'num_edges', 'connectivity_density', 'avg_degree',
    's1_error_mean', 's1_error_std', 's1_error_max', 's1_error_min',
    'cnot_error_mean', 'cnot_error_std', 'cnot_error_max', 'cnot_error_min',
    'ro_error_mean', 'ro_error_std', 'ro_error_max', 'ro_error_min',
    't1_mean', 't1_std', 't2_mean', 't2_std',
] if f in next(iter(backend_cache.values()))]
print(f"后端特征: {len(BACKEND_FEATS)} 个")

# ============================================================================
# 3. 电路特征（增强版）
# ============================================================================
def extract_circuit(qasm_path):
    try:
        circ = QuantumCircuit.from_qasm_file(str(qasm_path))
    except:
        return None
    w = circ.num_qubits
    d = circ.depth()
    tg = len(circ.data)
    gc = {}
    for inst in circ.data:
        gc[inst.operation.name] = gc.get(inst.operation.name, 0) + 1
    s1 = sum(1 for i in circ.data if len(i.qubits) == 1)
    s2 = sum(1 for i in circ.data if len(i.qubits) == 2)
    meas = gc.get('measure', 0)
    er = s2 / tg if tg > 0 else 0

    inter = {i: set() for i in range(w)}
    circ_edges = set()
    for inst in circ.data:
        if len(inst.qubits) == 2:
            q0 = circ.find_bit(inst.qubits[0]).index
            q1 = circ.find_bit(inst.qubits[1]).index
            inter[q0].add(q1)
            inter[q1].add(q0)
            circ_edges.add((q0, q1) if q0 < q1 else (q1, q0))

    ai = sum(len(v) for v in inter.values()) / w if w > 0 else 0
    pc = ai / (w - 1) if w > 1 else 0
    pl = 1 - (d / tg) if tg > 0 else 0

    dag = circuit_to_dag(circ)
    cp2q = sum(1 for n in dag.topological_op_nodes() if len(n.qargs) == 2)
    cd_ = cp2q / d if d > 0 else 0

    active = set()
    for inst in circ.data:
        for q in inst.qubits:
            active.add(circ.find_bit(q).index)
    live = len(active) / w if w > 0 else 0

    # 新增特征
    gate_density = tg / (w * d) if w > 0 and d > 0 else 0
    gate_div = len(gc) / 15
    rz = gc.get('rz', 0)
    nc_ratio = rz / tg if tg > 0 else 0
    na = len(active)
    max_e_circ = na * (na - 1) // 2 if na > 1 else 1
    cdc = len(circ_edges) / max_e_circ if max_e_circ > 0 else 0
    nm = tg - meas
    tqnm = s2 / nm if nm > 0 else 0

    dag_widths = [len(layer['graph'].op_nodes()) for layer in dag.layers()]
    max_dw = max(dag_widths) if dag_widths else 1
    pcapa = max_dw / w if w > 0 else 0
    aspect = np.log1p(d / w if w > 0 else d)
    cx_sw = gc.get('cx', 0) + gc.get('swap', 0)
    es = cx_sw / tg if tg > 0 else 0

    feats = {
        'width': w, 'depth': d, 'total_gates': tg,
        'single_qubit_gates': s1, 'two_qubit_gates': s2,
        'measure_gates': meas, 'entanglement_ratio': er,
        'program_communication': pc, 'parallelism': pl,
        'critical_depth': cd_, 'liveness': live,
        'h_gates': gc.get('h', 0), 'x_gates': gc.get('x', 0),
        'cx_gates': gc.get('cx', 0), 'rz_gates': rz,
        'gate_density': gate_density, 'gate_diversity': gate_div,
        'non_clifford_ratio': nc_ratio,
        'coupling_density_circuit': cdc,
        'twoq_nonmeasure_ratio': tqnm,
        'parallel_capacity': pcapa,
        'aspect_ratio': aspect,
        'entanglement_strength': es,
        '_circuit_edges': circ_edges,
    }
    return feats


def find_qasm(cname, folder):
    folder = Path(folder)
    exact = folder / f"{cname}.qasm"
    if exact.exists():
        return exact
    for f in folder.glob("*.qasm"):
        if cname in f.stem:
            return f
    return None


circ_cache = {}
for c in df['Circuit'].unique():
    p = find_qasm(c, QASM_FOLDER)
    if p is None:
        continue
    f = extract_circuit(p)
    if f:
        circ_cache[c] = f

CIRCUIT_FEATS_BASE = [
    'width', 'depth', 'total_gates', 'single_qubit_gates', 'two_qubit_gates',
    'measure_gates', 'entanglement_ratio', 'program_communication',
    'parallelism', 'critical_depth', 'liveness',
    'h_gates', 'x_gates', 'cx_gates', 'rz_gates',
]
CIRCUIT_FEATS_NEW = [
    'gate_density', 'gate_diversity', 'non_clifford_ratio',
    'coupling_density_circuit', 'twoq_nonmeasure_ratio',
    'parallel_capacity', 'aspect_ratio', 'entanglement_strength',
]
print(f"电路特征: {len(CIRCUIT_FEATS_BASE)}基础 + {len(CIRCUIT_FEATS_NEW)}新增 = {len(CIRCUIT_FEATS_BASE)+len(CIRCUIT_FEATS_NEW)}")

# ============================================================================
# 4. 交互特征
# ============================================================================
def interaction_feats(circuit, backend):
    cf = circ_cache[circuit]
    bf = backend_cache[backend]
    f = {}
    f['cum_cnot_error'] = cf['two_qubit_gates'] * bf.get('cnot_error_mean', 0)
    f['cum_s1_error'] = cf['single_qubit_gates'] * bf.get('s1_error_mean', 0)
    f['cum_ro_error'] = cf['width'] * bf.get('ro_error_mean', 0)
    f['depth_weighted_error'] = cf['depth'] * (bf.get('s1_error_mean', 0) + bf.get('cnot_error_mean', 0))
    f['cnot_sensitivity'] = cf['entanglement_ratio'] * bf.get('cnot_error_mean', 1)
    sr = cf['width'] / bf.get('backend_num_qubits', 1)
    f['size_mismatch'] = max(0, sr - 1)
    ce = cf.get('_circuit_edges', set())
    bc = bf.get('_coupling_set', set())
    f['coupling_match_ratio'] = sum(1 for e in ce if e in bc) / len(ce) if ce and bc else 0
    f['parallel_connect_need'] = cf['parallelism'] * bf.get('connectivity_density', 0)
    f['comm_degree'] = cf['program_communication'] * bf.get('avg_degree', 0)
    f['qubit_utilization'] = cf['liveness'] * bf.get('backend_num_qubits', 1)
    t1 = bf.get('t1_mean', 100e-6)
    f['t1_risk'] = cf['depth'] * 100e-9 / t1 if t1 > 0 else 0
    t2 = bf.get('t2_mean', 50e-6)
    f['t2_risk'] = cf['depth'] * 100e-9 / t2 if t2 > 0 else 0
    f['noise_budget'] = (
        cf['single_qubit_gates'] * bf.get('s1_error_mean', 0) +
        cf['two_qubit_gates'] * bf.get('cnot_error_mean', 0) +
        cf['width'] * bf.get('ro_error_mean', 0)
    )
    return f

INTERACTION_FEATS = [
    'cum_cnot_error', 'cum_s1_error', 'cum_ro_error', 'depth_weighted_error',
    'cnot_sensitivity', 'size_mismatch', 'coupling_match_ratio',
    'parallel_connect_need', 'comm_degree', 'qubit_utilization',
    't1_risk', 't2_risk', 'noise_budget',
]
print(f"交互特征: {len(INTERACTION_FEATS)} 个")

# ============================================================================
# 5. 数据划分
# ============================================================================
def get_alg(name):
    return name.split('_')[0] if '_' in name else 'unknown'

circuit_best = df.loc[df.groupby('Circuit')['FOM'].idxmax()][['Circuit', 'Backend']]
circuit_best = circuit_best.set_index('Circuit')['Backend'].to_dict()
all_circuits = df['Circuit'].unique()
labels = [circuit_best[c] for c in all_circuits]

from collections import Counter
lc = Counter(labels)
rare = [b for b, c in lc.items() if c == 1]
rare_circs = [c for c, b in circuit_best.items() if b in rare]

if rare_circs:
    remaining = [c for c in all_circuits if c not in rare_circs]
    rem_labels = [circuit_best[c] for c in remaining]
    target = int(len(all_circuits) * 0.70)
    train_rem = target - len(rare_circs)
    ts = 1 - (train_rem / len(remaining))
    sss = StratifiedShuffleSplit(1, test_size=ts, random_state=RANDOM_SEED)
    tr_idx, te_idx = next(sss.split(remaining, rem_labels))
    train_circuits = rare_circs + [remaining[i] for i in tr_idx]
    test_circuits = [remaining[i] for i in te_idx]
else:
    sss = StratifiedShuffleSplit(1, test_size=0.30, random_state=RANDOM_SEED)
    tr_idx, te_idx = next(sss.split(all_circuits, labels))
    train_circuits = all_circuits[tr_idx].tolist()
    test_circuits = all_circuits[te_idx].tolist()

train_df = df[df['Circuit'].isin(train_circuits)].copy()
test_df = df[df['Circuit'].isin(test_circuits)].copy()
print(f"训练: {len(train_circuits)} 电路, {len(train_df)} 样本")
print(f"测试: {len(test_circuits)} 电路, {len(test_df)} 样本")

# 倾向性得分
alg_prefs = {}
for _, r in train_df.iterrows():
    a = get_alg(r['Circuit'])
    b = r['Backend']
    fo = r['FOM']
    if a not in alg_prefs:
        alg_prefs[a] = {}
    alg_prefs[a][b] = alg_prefs[a].get(b, 0) + fo
alg_cnt = {}
for _, r in train_df.iterrows():
    a = get_alg(r['Circuit'])
    b = r['Backend']
    if a not in alg_cnt:
        alg_cnt[a] = {}
    alg_cnt[a][b] = alg_cnt[a].get(b, 0) + 1
for a in alg_prefs:
    for b in alg_prefs[a]:
        alg_prefs[a][b] /= alg_cnt[a][b]

# 算法级别后端排名（用于自适应权重）
alg_ranks = {}
for a in alg_prefs:
    ranked = sorted(alg_prefs[a].items(), key=lambda x: -x[1])
    alg_ranks[a] = [b for b, _ in ranked]

global_avg_fom = train_df['FOM'].mean()

# 基线后端
ro_bk = min(all_backends, key=lambda b: backend_cache[b].get('ro_error_mean', 1))
s1_bk = min(all_backends, key=lambda b: backend_cache[b].get('s1_error_mean', 1))
cnot_bk = min(all_backends, key=lambda b: backend_cache[b].get('cnot_error_mean', 1))
print(f"基线后端: RO={ro_bk}, S1={s1_bk}, CNOT={cnot_bk}")

# ============================================================================
# 6. 构建特征矩阵（增强版 = 基础电路 + 新增电路 + 后端 + 交互）
# ============================================================================
def build_X(circuits, df_subset):
    Xl, yl, pl = [], [], []
    for circ in circuits:
        if circ not in circ_cache:
            continue
        cf = circ_cache[circ]
        cd = df_subset[df_subset['Circuit'] == circ]
        for bk in all_backends:
            if bk not in backend_cache:
                continue
            row = cd[cd['Backend'] == bk]
            if len(row) == 0:
                continue
            comb = {}
            for f in CIRCUIT_FEATS_BASE:
                comb[f] = cf[f]
            for f in CIRCUIT_FEATS_NEW:
                comb[f] = cf[f]
            bf = backend_cache[bk]
            for f in BACKEND_FEATS:
                comb[f] = bf.get(f, 0)
            comb.update(interaction_feats(circ, bk))
            Xl.append(comb)
            yl.append(row['FOM'].iloc[0])
            pl.append((circ, bk))
    return pd.DataFrame(Xl), np.array(yl), pl


X_train, y_train, train_pairs = build_X(train_circuits, train_df)
X_test, y_test, test_pairs = build_X(test_circuits, test_df)
ALL_FEATURE_NAMES = list(X_train.columns)
print(f"特征矩阵: train {X_train.shape}, test {X_test.shape}")
print(f"总特征数: {len(ALL_FEATURE_NAMES)}")

# ============================================================================
# 7. 特征选择（阈值=0.005）
# ============================================================================
print("\n特征选择（阈值=0.005）...")
temp_rf = RandomForestRegressor(n_estimators=80, random_state=RANDOM_SEED, n_jobs=-1)
temp_rf.fit(X_train, y_train)
importance = pd.Series(temp_rf.feature_importances_, index=ALL_FEATURE_NAMES)

THRESHOLD = 0.005
selected_feats = importance[importance > THRESHOLD].index.tolist()
dropped_feats = importance[importance <= THRESHOLD].index.tolist()

print(f"  保留: {len(selected_feats)} 个特征")
print(f"  丢弃: {len(dropped_feats)} 个特征")

# 标准化
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train[selected_feats])
X_test_s = scaler.transform(X_test[selected_feats])

# ============================================================================
# 8. 训练RF模型
# ============================================================================
print("\n训练RF模型（阈值0.005，与敏感性分析一致）...")
rf_model = RandomForestRegressor(
    n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
)
rf_model.fit(X_train_s, y_train)

# 回归评估
yp_tr = rf_model.predict(X_train_s)
yp_te = rf_model.predict(X_test_s)
train_r2 = r2_score(y_train, yp_tr)
test_r2 = r2_score(y_test, yp_te)
train_mae = mean_absolute_error(y_train, yp_tr)
test_mae = mean_absolute_error(y_test, yp_te)
print(f"  Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")
print(f"  Train MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
print(f"  保留特征: {len(selected_feats)}/{len(ALL_FEATURE_NAMES)}")
print(f"  丢弃特征: {len(dropped_feats)}/{len(ALL_FEATURE_NAMES)}")

PROPENSITY_WEIGHT = 0.20

# ============================================================================
# 9. 保存模型
# ============================================================================
# 12. 保存模型
# ============================================================================
model_artifacts = {
    'model': rf_model,
    'scaler': scaler,
    'selected_features': selected_feats,
    'all_backends': all_backends,
    'train_circuits': train_circuits,
    'test_circuits': test_circuits,
    'circuit_features_cache': circ_cache,
    'backend_features_cache': backend_cache,
    'propensity_weight': PROPENSITY_WEIGHT,
    'algorithm_preferences': alg_prefs,
    'algorithm_ranks': alg_ranks,
    'global_avg_fom': global_avg_fom,
    'threshold': THRESHOLD,
    'feature_importance': importance.to_dict(),
    'lowest_ro_backend': ro_bk,
    'lowest_s1_backend': s1_bk,
    'lowest_cnot_backend': cnot_bk,
    'train_df': train_df,
    'test_df': test_df,
    'train_r2': train_r2,
    'test_r2': test_r2,
    'train_mae': train_mae,
    'test_mae': test_mae,
}

with open(OUTPUT_DIR / 'trained_model_optimized.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

# 特征列表保存
feat_df = pd.DataFrame({
    'feature': importance.index,
    'importance': importance.values,
    'threshold_0_005': ['retained' if imp > THRESHOLD else 'dropped' for imp in importance.values],
})
feat_df.to_csv(OUTPUT_DIR / 'feature_list_threshold_005.csv', index=False, encoding='utf-8-sig')

# 结果报告
with open(OUTPUT_DIR / 'optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write("Q-Selector 优化版训练报告\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"特征重要性阈值: {THRESHOLD}\n")
    f.write(f"倾向性得分权重: {PROPENSITY_WEIGHT}\n")
    f.write(f"保留特征: {len(selected_feats)}/56\n")
    f.write(f"丢弃特征: {len(dropped_feats)}/56\n\n")
    f.write(f"Regression Performance:\n")
    f.write(f"  Train R2: {train_r2:.4f}\n")
    f.write(f"  Test R2:  {test_r2:.4f}\n")
    f.write(f"  Train MAE: {train_mae:.6f}\n")
    f.write(f"  Test MAE:  {test_mae:.6f}\n\n")
    f.write(f"Configuration:\n")
    f.write(f"  Feature Threshold: {THRESHOLD}\n")
    f.write(f"  Propensity Weight: {PROPENSITY_WEIGHT}\n")
    f.write(f"  Retained Features: {len(selected_feats)}/56\n")
    f.write(f"  Dropped Features: {len(dropped_feats)}/56\n")

print(f"\n模型已保存至: {OUTPUT_DIR}")
print(f"特征列表: {OUTPUT_DIR / 'feature_list_threshold_005.csv'}")
print(f"报告: {OUTPUT_DIR / 'optimization_report.txt'}")
print("=" * 70)
