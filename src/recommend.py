#!/usr/bin/env python
"""
Q-Selector: Quantum Backend Recommender
=========================================
对未知的量子线路 QASM 文件/文件夹，直接加载已训练好的 PKL 模型推荐最佳处理器，
无需与任何最优处理器比较（适用于真实场景中的未知线路）。

支持：
  - 单个 QASM 文件推荐
  - 整个 QASM 文件夹批量推荐

结果自动保存到 results/your_work/<timestamp>/ 目录下，每次运行独立存档：
  - best_recommendation.csv     每个电路的最佳推荐（核心结果）
  - full_ranking.csv            所有电路 × 所有后端的完整排名
  - full_ranking.json           结构化 JSON（含错误率详情）

Usage:
    python src/recommend.py <qasm_file_or_dir>
    python src/recommend.py data/QASM/
    python src/recommend.py my_circuit.qasm --top 10
    python src/recommend.py my_circuit.qasm --no-save
"""

import sys
import json as json_mod
import pickle
import warnings
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

QASM_EXTENSIONS = {'.qasm', '.QASM'}


# ============================================================================
# 工具函数
# ============================================================================

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json_mod.load(f)
    except Exception:
        return None


def get_algorithm_type(circuit_name):
    """从电路文件名推断算法类型，用于倾向性得分修正（可选）。"""
    parts = Path(circuit_name).stem.split('_')
    return parts[0] if len(parts) >= 1 else 'unknown'


def collect_qasm_files(path):
    """收集路径下的所有 QASM 文件。如果是单个文件直接返回，文件夹则递归查找。"""
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix in QASM_EXTENSIONS else []
    elif p.is_dir():
        return sorted([f for f in p.rglob('*') if f.suffix in QASM_EXTENSIONS])
    return []


# ============================================================================
# 电路特征提取（与 train.py / predict.py 完全一致）
# ============================================================================

def extract_circuit_features(qasm_path):
    """
    从 QASM 文件中提取所有电路特征（23 维：15 基础 + 8 新增）。

    Parameters
    ----------
    qasm_path : str or Path
        QASM 文件路径。

    Returns
    -------
    dict
        包含电路特征的字典，提取失败返回 None。
    """
    try:
        circ = QuantumCircuit.from_qasm_file(str(qasm_path))
    except Exception as e:
        print(f"  [错误] 无法加载 QASM 文件: {e}")
        return None

    w = circ.num_qubits          # 宽度（量子比特数）
    d = circ.depth()             # 深度
    tg = len(circ.data)          # 总门数

    # 门计数器
    gc = {}
    for inst in circ.data:
        gc[inst.operation.name] = gc.get(inst.operation.name, 0) + 1

    s1 = sum(1 for i in circ.data if len(i.qubits) == 1)   # 单量子比特门
    s2 = sum(1 for i in circ.data if len(i.qubits) == 2)   # 两量子比特门
    meas = gc.get('measure', 0)
    er = s2 / tg if tg > 0 else 0                           # 纠缠比

    # 电路连通性分析
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
    pc = ai / (w - 1) if w > 1 else 0                       # 程序通信度
    pl = 1 - (d / tg) if tg > 0 else 0                      # 并行度

    dag = circuit_to_dag(circ)
    cp2q = sum(1 for n in dag.topological_op_nodes() if len(n.qargs) == 2)
    cd_ = cp2q / d if d > 0 else 0                          # 关键深度

    active = set()
    for inst in circ.data:
        for q in inst.qubits:
            active.add(circ.find_bit(q).index)
    live = len(active) / w if w > 0 else 0                  # 活跃度

    # ---- 8 个新增特征 ----
    gate_density = tg / (w * d) if w > 0 and d > 0 else 0
    gate_div = len(gc) / 15                                  # 门多样性
    rz = gc.get('rz', 0)
    nc_ratio = rz / tg if tg > 0 else 0                      # 非 Clifford 比
    na = len(active)
    max_e_circ = na * (na - 1) // 2 if na > 1 else 1
    cdc = len(circ_edges) / max_e_circ if max_e_circ > 0 else 0  # 电路耦合密度
    nm = tg - meas
    tqnm = s2 / nm if nm > 0 else 0                          # 非测量双门比

    dag_widths = [len(layer['graph'].op_nodes()) for layer in dag.layers()]
    max_dw = max(dag_widths) if dag_widths else 1
    pcapa = max_dw / w if w > 0 else 0                       # 并行容量
    aspect = np.log1p(d / w if w > 0 else d)                 # 长宽比
    cx_sw = gc.get('cx', 0) + gc.get('swap', 0)
    es = cx_sw / tg if tg > 0 else 0                         # 纠缠强度

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


# ============================================================================
# 交互特征计算（与 train.py / predict.py 完全一致）
# ============================================================================

def interaction_feats(cf, bf):
    """计算电路与后端之间的 13 个交互特征。"""
    f = {}
    f['cum_cnot_error'] = cf['two_qubit_gates'] * bf.get('cnot_error_mean', 0)
    f['cum_s1_error'] = cf['single_qubit_gates'] * bf.get('s1_error_mean', 0)
    f['cum_ro_error'] = cf['width'] * bf.get('ro_error_mean', 0)
    f['depth_weighted_error'] = cf['depth'] * (
        bf.get('s1_error_mean', 0) + bf.get('cnot_error_mean', 0))
    f['cnot_sensitivity'] = cf['entanglement_ratio'] * bf.get('cnot_error_mean', 1)
    sr = cf['width'] / bf.get('backend_num_qubits', 1)
    f['size_mismatch'] = max(0, sr - 1)
    ce = cf.get('_circuit_edges', set())
    bc = bf.get('_coupling_set', set())
    f['coupling_match_ratio'] = (
        sum(1 for e in ce if e in bc) / len(ce) if ce and bc else 0)
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
# 后端耦合映射加载（交互特征需要 _coupling_set）
# ============================================================================

def load_coupling_maps(backend_json_folder, all_backends):
    """从 conf_*.json 加载所有后端的耦合映射集。"""
    jfolder = Path(backend_json_folder)
    maps = {}
    for bk in all_backends:
        key = bk.lower().replace("fake", "").replace("v2", "")
        candidates = [
            jfolder / f"conf_{key}.json",
            jfolder / f"conf_{bk.lower()}.json",
        ]
        for p in candidates:
            if p.exists():
                conf = load_json(p)
                if conf and 'coupling_map' in conf:
                    edges = set()
                    for e in conf['coupling_map']:
                        u, v = e[0], e[1]
                        edges.add((u, v) if u < v else (v, u))
                    maps[bk] = edges
                break
    return maps


# ============================================================================
# 模型加载（仅一次）
# ============================================================================

def load_model(model_path, backend_json_folder):
    """
    加载训练好的模型工件和后端耦合映射。

    Returns
    -------
    dict
        包含模型、scaler、特征列表、后端缓存等所有工件的字典。
    """
    print(f"● 第1步: 加载模型")
    print(f"  模型文件: {model_path}")
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)

    # 注入耦合映射到后端特征缓存
    print(f"\n● 第2步: 加载后端耦合映射")
    all_backends = artifacts['all_backends']
    coupling_maps = load_coupling_maps(backend_json_folder, all_backends)
    bk_cache = artifacts['backend_features_cache']
    for bk in all_backends:
        if bk in bk_cache and bk in coupling_maps:
            bk_cache[bk]['_coupling_set'] = coupling_maps[bk]
    print(f"  候选后端数: {len(all_backends)}")
    print(f"  已加载 {len(coupling_maps)} 个后端的耦合映射")
    print(f"  保留特征数: {len(artifacts['selected_features'])}")

    artifacts['_coupling_maps'] = coupling_maps
    return artifacts


# ============================================================================
# 单电路预测
# ============================================================================

def predict_circuit(qasm_path, artifacts):
    """
    对单个 QASM 电路预测所有后端的 FOM。

    Parameters
    ----------
    qasm_path : Path
        QASM 文件路径。
    artifacts : dict
        由 load_model() 返回的模型工件字典。

    Returns
    -------
    dict | None
        包含 'circuit_name', 'features', 'rankings' 的字典，失败返回 None。
    """
    rf_model = artifacts['model']
    scaler = artifacts['scaler']
    selected_features = artifacts['selected_features']
    all_backends = artifacts['all_backends']
    backend_features_cache = artifacts['backend_features_cache']
    propensity_weight = artifacts.get('propensity_weight', 0.20)
    algorithm_preferences = artifacts.get('algorithm_preferences', {})
    global_avg_fom = artifacts.get('global_avg_fom', 0)

    # 特征名列
    base_circuit_feats = [
        'width', 'depth', 'total_gates', 'single_qubit_gates',
        'two_qubit_gates', 'measure_gates', 'entanglement_ratio',
        'program_communication', 'parallelism', 'critical_depth', 'liveness',
        'h_gates', 'x_gates', 'cx_gates', 'rz_gates',
    ]
    new_circuit_feats = [
        'gate_density', 'gate_diversity', 'non_clifford_ratio',
        'coupling_density_circuit', 'twoq_nonmeasure_ratio',
        'parallel_capacity', 'aspect_ratio', 'entanglement_strength',
    ]
    backend_feat_names = [
        'backend_num_qubits', 'num_edges', 'connectivity_density', 'avg_degree',
        's1_error_mean', 's1_error_std', 's1_error_max', 's1_error_min',
        'cnot_error_mean', 'cnot_error_std', 'cnot_error_max', 'cnot_error_min',
        'ro_error_mean', 'ro_error_std', 'ro_error_max', 'ro_error_min',
        't1_mean', 't1_std', 't2_mean', 't2_std',
    ]

    # 提取特征
    cf = extract_circuit_features(qasm_path)
    if cf is None:
        return None

    circuit_name = qasm_path.stem
    algorithm = get_algorithm_type(circuit_name)
    use_propensity = algorithm in algorithm_preferences

    rankings = []
    for bk_name in all_backends:
        if bk_name not in backend_features_cache:
            continue

        bf = backend_features_cache[bk_name]

        combined = {}
        for f in base_circuit_feats:
            combined[f] = cf[f]
        for f in new_circuit_feats:
            combined[f] = cf[f]
        for f in backend_feat_names:
            combined[f] = bf.get(f, 0)
        combined.update(interaction_feats(cf, bf))

        X_df = pd.DataFrame([combined])[selected_features]
        X_scaled = scaler.transform(X_df)
        pred_fom = rf_model.predict(X_scaled)[0]

        if use_propensity and bk_name in algorithm_preferences[algorithm]:
            algo_avg = algorithm_preferences[algorithm][bk_name]
            pred_fom += (algo_avg - global_avg_fom) * propensity_weight

        rankings.append({
            'backend': bk_name,
            'predicted_fom': pred_fom,
            'num_qubits': bf.get('backend_num_qubits', 0),
            'cnot_error': bf.get('cnot_error_mean', 0),
            'ro_error': bf.get('ro_error_mean', 0),
        })

    rankings.sort(key=lambda x: -x['predicted_fom'])

    return {
        'circuit_name': circuit_name,
        'qasm_filename': qasm_path.name,
        'features': {
            'width': cf['width'],
            'depth': cf['depth'],
            'total_gates': cf['total_gates'],
            'single_qubit_gates': cf['single_qubit_gates'],
            'two_qubit_gates': cf['two_qubit_gates'],
        },
        'rankings': rankings,
        'best_backend': rankings[0]['backend'],
        'best_fom': rankings[0]['predicted_fom'],
    }


# ============================================================================
# 结果展示
# ============================================================================

def print_circuit_result(result, top_k=5):
    """友好打印单个电路的推荐结果。"""
    cn = result['circuit_name']
    feat = result['features']
    rankings = result['rankings']

    print(f"\n  ── {cn} ──")
    print(f"     量子比特: {feat['width']}, 深度: {feat['depth']}, "
          f"总门数: {feat['total_gates']}")
    print(f"     >> Top-1 推荐: {result['best_backend']}  "
          f"(Pred FOM = {result['best_fom']:.4f})")

    print(f"     {'Rank':<6} {'Backend':<30} {'Pred FOM':<14} {'Qubits':<8} {'CNOT Err':<12}")
    print(f"     {'-'*68}")
    for i, r in enumerate(rankings[:top_k]):
        print(f"     {i+1:<6} {r['backend']:<30} {r['predicted_fom']:<14.4f} "
              f"{r['num_qubits']:<8} {r['cnot_error']:<12.6f}")
    if top_k < len(rankings):
        print(f"     ... 共 {len(rankings)} 个后端")


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Q-Selector: 对未知量子线路推荐最佳处理器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python src/recommend.py data/QASM/ae_indep_qiskit_10.qasm
  python src/recommend.py data/QASM/                    # 批量推荐整个文件夹
  python src/recommend.py my_circuit.qasm --top 10
  python src/recommend.py my_circuit.qasm --no-save     # 仅终端输出
        """,
    )
    parser.add_argument('input', type=str,
                        help='QASM 文件或包含 QASM 文件的文件夹路径')
    parser.add_argument('--model', type=str, default=None,
                        help='训练好的模型 PKL 路径（默认: results/trained_model_optimized.pkl）')
    parser.add_argument('--backend-json', type=str, default=None,
                        help='后端 JSON 配置文件夹路径（默认: data/backend_json）')
    parser.add_argument('--top', type=int, default=5,
                        help='显示前 N 个推荐（默认: 5）')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存结果文件，仅终端输出')

    args = parser.parse_args()

    # ---- 解析路径 ----
    base_dir = Path(__file__).resolve().parent.parent
    model_path = Path(args.model) if args.model else base_dir / 'results' / 'trained_model_optimized.pkl'
    bk_json_dir = Path(args.backend_json) if args.backend_json else base_dir / 'data' / 'backend_json'

    if not model_path.exists():
        print(f"[错误] 模型文件不存在: {model_path}")
        print("请先运行 python src/train.py 训练模型。")
        sys.exit(1)

    if not bk_json_dir.exists():
        print(f"[错误] 后端配置文件夹不存在: {bk_json_dir}")
        sys.exit(1)

    # ---- 收集 QASM 文件 ----
    qasm_files = collect_qasm_files(args.input)
    if not qasm_files:
        print(f"[错误] 未找到 QASM 文件: {args.input}")
        sys.exit(1)

    input_path = Path(args.input)
    src_desc = f"文件夹 ({len(qasm_files)} 个 QASM 文件)" if input_path.is_dir() else qasm_files[0].name
    print(f"输入: {src_desc}\n")

    # ---- 加载模型（仅一次） ----
    artifacts = load_model(model_path, bk_json_dir)

    rf_model = artifacts['model']
    print(f"  随机森林模型: n_estimators={rf_model.n_estimators}")

    # ---- 逐电路预测 ----
    print(f"\n● 第3步: 特征提取与 FOM 预测")
    all_results = []
    failed = []
    n_total = len(qasm_files)

    for idx, qf in enumerate(qasm_files, 1):
        print(f"\r  正在处理 [{idx}/{n_total}] {qf.name}...", end='', flush=True)
        result = predict_circuit(qf, artifacts)
        if result is not None:
            all_results.append(result)
        else:
            failed.append(qf.name)

    print()
    print(f"\n  成功: {len(all_results)} 个电路", end='')
    if failed:
        print(f", 失败: {len(failed)} 个 ({', '.join(failed)})")
    else:
        print()

    if not all_results:
        print("[错误] 所有电路处理失败。")
        sys.exit(1)

    # ---- 保存结果到 your_work/<timestamp>/ ----
    if not args.no_save:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = base_dir / 'results' / 'your_work' / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) best_recommendation.csv — 每个电路的最佳推荐
        best_rows = []
        for r in all_results:
            best = r['rankings'][0]
            best_rows.append({
                'circuit': r['circuit_name'],
                'qasm_file': r['qasm_filename'],
                'width': r['features']['width'],
                'depth': r['features']['depth'],
                'total_gates': r['features']['total_gates'],
                'recommended_backend': best['backend'],
                'predicted_fom': round(best['predicted_fom'], 6),
                'num_qubits': best['num_qubits'],
            })
        best_df = pd.DataFrame(best_rows)
        best_path = out_dir / 'best_recommendation.csv'
        best_df.to_csv(best_path, index=False, encoding='utf-8-sig')
        print(f"\n   [保存] 最佳推荐 -> {best_path}")

        # 2) full_ranking.csv — 所有电路 × 所有后端（展开为长表）
        rank_rows = []
        for r in all_results:
            for i, bk in enumerate(r['rankings']):
                rank_rows.append({
                    'circuit': r['circuit_name'],
                    'qasm_file': r['qasm_filename'],
                    'rank': i + 1,
                    'backend': bk['backend'],
                    'predicted_fom': round(bk['predicted_fom'], 6),
                    'num_qubits': bk['num_qubits'],
                    'cnot_error': round(bk['cnot_error'], 8),
                    'ro_error': round(bk['ro_error'], 8),
                })
        rank_df = pd.DataFrame(rank_rows)
        rank_path = out_dir / 'full_ranking.csv'
        rank_df.to_csv(rank_path, index=False, encoding='utf-8-sig')
        print(f"   [保存] 完整排名 -> {rank_path}  ({len(rank_rows)} 行)")

        # 3) full_ranking.json — 结构化 JSON
        json_output = []
        for r in all_results:
            json_output.append({
                'circuit': r['circuit_name'],
                'qasm_file': r['qasm_filename'],
                'circuit_features': r['features'],
                'total_backends': len(r['rankings']),
                'top1_recommendation': {
                    'backend': r['rankings'][0]['backend'],
                    'predicted_fom': round(r['rankings'][0]['predicted_fom'], 6),
                    'num_qubits': r['rankings'][0]['num_qubits'],
                    'cnot_error': round(r['rankings'][0]['cnot_error'], 8),
                    'ro_error': round(r['rankings'][0]['ro_error'], 8),
                },
                'all_rankings': [
                    {
                        'rank': i + 1,
                        'backend': bk['backend'],
                        'predicted_fom': round(bk['predicted_fom'], 6),
                        'num_qubits': bk['num_qubits'],
                        'cnot_error': round(bk['cnot_error'], 8),
                        'ro_error': round(bk['ro_error'], 8),
                    }
                    for i, bk in enumerate(r['rankings'])
                ],
            })
        json_path = out_dir / 'full_ranking.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json_mod.dump({
                'timestamp': ts,
                'total_circuits': len(all_results),
                'total_backends': len(artifacts['all_backends']),
                'results': json_output,
            }, f, indent=2, ensure_ascii=False)
        print(f"   [保存] 完整排名 -> {json_path}")

    # ---- 终端输出 ----
    print(f"\n{'='*80}")
    print("   Q-Selector 处理器推荐结果")
    print(f"{'='*80}")

    if len(all_results) == 1:
        print_circuit_result(all_results[0], top_k=args.top)
    else:
        print(f"\n   共 {len(all_results)} 个电路:")
        print(f"   {'Circuit':<45} {'Top-1 Backend':<28} {'Pred FOM':<12}")
        print(f"   {'-'*85}")
        for r in all_results:
            print(f"   {r['circuit_name']:<45} {r['best_backend']:<28} {r['best_fom']:<12.4f}")

    if not args.no_save:
        print(f"\n   结果已保存至: {out_dir}")

    print(f"\n推荐完成！")


if __name__ == '__main__':
    main()
