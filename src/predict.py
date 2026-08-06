"""
优化版预测脚本 (Predict Optimized)
====================================
加载 train_model_optimized.py 训练好的模型，对测试电路做预测。
用法: python predict_optimized.py

依赖:
  - results_optimized/trained_model_optimized.pkl
  - MQTBench_578/ (QASM文件)
  - backend_json/ (后端配置)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / 'results'
QASM_FOLDER = BASE_DIR / 'data' / 'QASM'
BACKEND_JSON_FOLDER = BASE_DIR / 'data' / 'backend_json'

print("=" * 70)
print("Optimized Prediction (predict_optimized.py)")
print("=" * 70)

# ============================================================================
# 1. 加载模型和工件
# ============================================================================
print("\n[1] 加载模型...")
with open(OUTPUT_DIR / 'trained_model_optimized.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

rf_model = model_artifacts['model']
scaler = model_artifacts['scaler']
selected_features = model_artifacts['selected_features']
all_backends = model_artifacts['all_backends']
train_circuits = model_artifacts['train_circuits']
test_circuits = model_artifacts['test_circuits']
circuit_features_cache = model_artifacts['circuit_features_cache']
backend_features_cache = model_artifacts['backend_features_cache']
propensity_weight = model_artifacts['propensity_weight']
algorithm_preferences = model_artifacts['algorithm_preferences']
global_avg_fom = model_artifacts['global_avg_fom']
train_df = model_artifacts['train_df']
test_df = model_artifacts['test_df']

print(f"  模型: RF (n_estimators=100)")
print(f"  特征数: {len(selected_features)}")
print(f"  倾向性权重: {propensity_weight}")
print(f"  后端数: {len(all_backends)}")
print(f"  测试电路: {len(test_circuits)}")

# ============================================================================
# 2. 加载后端特征（用于预测时计算交互特征）
# ============================================================================
print("\n[2] 加载后端配置...")
def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

# 加载耦合映射（交互特征需要）
coupling_maps = {}
json_folder = Path(BACKEND_JSON_FOLDER)
for key in backend_features_cache:
    bk_lower = key.lower().replace("fake", "").replace("v2", "")
    possible = [
        json_folder / f"conf_{bk_lower}.json",
        json_folder / f"conf_{key.lower()}.json",
    ]
    for p in possible:
        if p.exists():
            conf = load_json(p)
            if conf and 'coupling_map' in conf:
                cm = conf['coupling_map']
                edges = set()
                for e in cm:
                    u, v = e[0], e[1]
                    edges.add((u, v) if u < v else (v, u))
                coupling_maps[key] = edges
            break

# 给后端特征缓存补充耦合集
for bk_name in all_backends:
    if bk_name in backend_features_cache and bk_name in coupling_maps:
        backend_features_cache[bk_name]['_coupling_set'] = coupling_maps[bk_name]

# ============================================================================
# 3. 电路特征提取（增强版 = 基础15 + 新增8）
# ============================================================================
def extract_circuit(qasm_path):
    try:
        circ = QuantumCircuit.from_qasm_file(str(qasm_path))
    except:
        return None

    w = circ.num_qubits; d = circ.depth(); tg = len(circ.data)
    gc = {}
    for inst in circ.data: gc[inst.operation.name] = gc.get(inst.operation.name, 0) + 1
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
            inter[q0].add(q1); inter[q1].add(q0)
            circ_edges.add((q0, q1) if q0 < q1 else (q1, q0))

    ai = sum(len(v) for v in inter.values()) / w if w > 0 else 0
    pc = ai / (w - 1) if w > 1 else 0
    pl = 1 - (d / tg) if tg > 0 else 0

    dag = circuit_to_dag(circ)
    cp2q = sum(1 for n in dag.topological_op_nodes() if len(n.qargs) == 2)
    cd_ = cp2q / d if d > 0 else 0

    active = set()
    for inst in circ.data:
        for q in inst.qubits: active.add(circ.find_bit(q).index)
    live = len(active) / w if w > 0 else 0

    # ---- 8个新增特征 ----
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
    if exact.exists(): return exact
    for f in folder.glob("*.qasm"):
        if cname in f.stem: return f
    return None


# ============================================================================
# 4. 交互特征计算
# ============================================================================
def interaction_feats(circuit, backend):
    cf = circuit_features_cache[circuit]
    bf = backend_features_cache[backend]
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


# ============================================================================
# 5. 预测函数
# ============================================================================
def get_algorithm_type(circuit_name):
    parts = circuit_name.split('_')
    return parts[0] if len(parts) >= 1 else 'unknown'


def predict_with_optimized(circuit_name, backends):
    """对单个电路，预测所有后端的FOM"""
    predictions = {}
    algorithm = get_algorithm_type(circuit_name)

    if circuit_name not in circuit_features_cache:
        return {bk: 0 for bk in backends}

    for backend_name in backends:
        if backend_name not in backend_features_cache:
            predictions[backend_name] = 0
            continue

        cf = circuit_features_cache[circuit_name]
        bf = backend_features_cache[backend_name]

        combined = {}
        # 基础电路特征 (15个)
        base_circuit_feats = ['width', 'depth', 'total_gates', 'single_qubit_gates',
            'two_qubit_gates', 'measure_gates', 'entanglement_ratio',
            'program_communication', 'parallelism', 'critical_depth', 'liveness',
            'h_gates', 'x_gates', 'cx_gates', 'rz_gates']
        for f in base_circuit_feats:
            combined[f] = cf[f]
        # 新增电路特征 (8个)
        new_circuit_feats = ['gate_density', 'gate_diversity', 'non_clifford_ratio',
            'coupling_density_circuit', 'twoq_nonmeasure_ratio',
            'parallel_capacity', 'aspect_ratio', 'entanglement_strength']
        for f in new_circuit_feats:
            combined[f] = cf[f]
        # 后端特征
        for f in ['backend_num_qubits', 'num_edges', 'connectivity_density', 'avg_degree',
            's1_error_mean', 's1_error_std', 's1_error_max', 's1_error_min',
            'cnot_error_mean', 'cnot_error_std', 'cnot_error_max', 'cnot_error_min',
            'ro_error_mean', 'ro_error_std', 'ro_error_max', 'ro_error_min',
            't1_mean', 't1_std', 't2_mean', 't2_std']:
            combined[f] = bf.get(f, 0)
        # 交互特征
        combined.update(interaction_feats(circuit_name, backend_name))

        # 模型预测
        X_single = pd.DataFrame([combined])[selected_features]
        feature_scaled = scaler.transform(X_single)
        rf_pred = rf_model.predict(feature_scaled)[0]

        # 倾向性得分调整（固定权重0.15）
        preference_adjustment = 0
        if algorithm in algorithm_preferences and backend_name in algorithm_preferences[algorithm]:
            algo_avg_fom = algorithm_preferences[algorithm][backend_name]
            preference_adjustment = (algo_avg_fom - global_avg_fom) * propensity_weight

        predictions[backend_name] = rf_pred + preference_adjustment

    return predictions


# ============================================================================
# 6. 对测试电路做预测
# ============================================================================
print(f"\n[3] 提取测试电路特征...")
test_qasm_features = {}
failed_circuits = []

for circuit_name in test_circuits:
    qasm_path = find_qasm(circuit_name, QASM_FOLDER)
    if qasm_path is None:
        failed_circuits.append(circuit_name)
        continue
    features = extract_circuit(qasm_path)
    if features is not None:
        test_qasm_features[circuit_name] = features

print(f"  成功: {len(test_qasm_features)} 电路")
if failed_circuits:
    print(f"  失败: {len(failed_circuits)} 电路")

# 补充到特征缓存
for circ_name in test_qasm_features:
    if circ_name not in circuit_features_cache:
        circuit_features_cache[circ_name] = test_qasm_features[circ_name]

print(f"\n[4] 逐电路预测...")
results = []
for circuit in test_circuits:
    if circuit not in circuit_features_cache:
        continue

    circuit_data = test_df[test_df['Circuit'] == circuit]
    if len(circuit_data) == 0:
        continue

    true_best_row = circuit_data.loc[circuit_data['FOM'].idxmax()]
    true_best_backend = true_best_row['Backend']
    true_best_fom = true_best_row['FOM']

    all_preds = predict_with_optimized(circuit, all_backends)
    if not all_preds:
        continue

    # 最佳后端
    pred_best_backend = max(all_preds, key=all_preds.get)
    pred_best_fom = all_preds[pred_best_backend]
    is_correct = (pred_best_backend == true_best_backend)

    # Top-3
    sorted_preds = sorted(all_preds.items(), key=lambda x: -x[1])
    top3_backends = [bk for bk, _ in sorted_preds[:3]]
    top3_preds = [fom for _, fom in sorted_preds[:3]]

    # 实际FOM
    actual_fom_selected = circuit_data[circuit_data['Backend'] == pred_best_backend]['FOM'].iloc[0] \
        if len(circuit_data[circuit_data['Backend'] == pred_best_backend]) > 0 else 0

    # 排名
    ranking = circuit_data.sort_values('FOM', ascending=False)['Backend'].tolist()
    pred_rank = ranking.index(pred_best_backend) + 1 if pred_best_backend in ranking else len(all_backends)

    results.append({
        'circuit': circuit,
        'true_best': true_best_backend,
        'true_fom': true_best_fom,
        'pred_best': pred_best_backend,
        'pred_fom': pred_best_fom,
        'actual_fom_selected': actual_fom_selected,
        'correct': is_correct,
        'top3_backends': top3_backends,
        'top3_preds': top3_preds,
        'pred_rank': pred_rank,
    })

print(f"  完成 {len(results)} 个电路")

correct = sum(1 for r in results if r['correct'])
total = len(results)
accuracy = correct / total if total > 0 else 0
top3_correct = sum(1 for r in results if r['true_best'] in r['top3_backends'])
top3_accuracy = top3_correct / total if total > 0 else 0

print(f"\n  Top-1准确率: {accuracy:.4f} ({correct}/{total})")
print(f"  Top-3准确率: {top3_accuracy:.4f} ({top3_correct}/{total})")

# ============================================================================
# 7. 保存结果
# ============================================================================
print(f"\n[5] 保存预测结果...")

results_df = pd.DataFrame([{
    'circuit': r['circuit'],
    'true_best_backend': r['true_best'],
    'true_fom': r['true_fom'],
    'pred_best_backend': r['pred_best'],
    'pred_fom': r['pred_fom'],
    'actual_fom_selected': r['actual_fom_selected'],
    'correct': r['correct'],
    'pred_rank': r['pred_rank'],
    'top3_backends': str(r['top3_backends']),
    'top3_preds': str(r['top3_preds']),
} for r in results])

results_df.to_csv(OUTPUT_DIR / 'predictions_optimized.csv', index=False, encoding='utf-8-sig')
print(f"  预测结果: {OUTPUT_DIR / 'predictions_optimized.csv'}")

# 打印前15个
print(f"\n预测结果 (前15):")
print("-" * 100)
print(f"{'Circuit':<45} {'True Best':<22} {'Pred Best':<22} {'Correct'}")
print("-" * 100)
for r in results[:15]:
    status = "Yes" if r['correct'] else "No"
    print(f"{r['circuit']:<45} {r['true_best']:<22} {r['pred_best']:<22} {status}")

# 错误分析
errors = [r for r in results if not r['correct']]
if errors:
    print(f"\n错误分析 ({len(errors)} 个误判):")
    for r in errors[:10]:
        print(f"\n  Circuit: {r['circuit'][:50]}")
        print(f"    True best: {r['true_best']} (FOM={r['true_fom']:.4f})")
        print(f"    Pred best: {r['pred_best']} (FOM={r['pred_fom']:.4f})")
        circuit_data = test_df[test_df['Circuit'] == r['circuit']]
        ranking = circuit_data.sort_values('FOM', ascending=False)['Backend'].tolist()
        if r['pred_best'] in ranking:
            print(f"    实际排名: #{ranking.index(r['pred_best']) + 1}")

# 预测汇总文本
with open(OUTPUT_DIR / 'prediction_summary.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("Prediction Summary Report (Optimized)\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  Total test circuits: {total}\n")
    f.write(f"  Best backend selection accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Correct predictions: {correct}\n")
    f.write(f"  Top-3 accuracy: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)\n")
    f.write(f"  Top-3 correct: {top3_correct}\n\n")
    f.write("CIRCUIT PREDICTIONS:\n")
    f.write("-" * 90 + "\n")
    f.write(f"{'Circuit':<50} {'True Best':<22} {'Pred Best':<22} {'Correct'}\n")
    f.write("-" * 90 + "\n")
    for r in results:
        status = "Yes" if r['correct'] else "No"
        f.write(f"{r['circuit']:<50} {r['true_best']:<22} {r['pred_best']:<22} {status}\n")
    if errors:
        f.write("\n\nMISCLASSIFIED CIRCUITS:\n")
        for r in errors:
            f.write(f"\n  Circuit: {r['circuit']}\n")
            f.write(f"    True best: {r['true_best']} (FOM={r['true_fom']:.4f})\n")
            f.write(f"    Pred best: {r['pred_best']} (FOM={r['pred_fom']:.4f})\n")

print(f"  预测汇总: {OUTPUT_DIR / 'prediction_summary.txt'}")
print("\n" + "=" * 70)
print("Prediction completed!")
print("=" * 70)