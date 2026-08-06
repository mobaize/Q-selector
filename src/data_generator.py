"""
data_generator.py - 量子电路数据生成器
负责从QASM文件获取数据，进行噪声模拟，并生成CSV文件
"""

import os
import csv
import pandas as pd
import numpy as np
import re
from math import log2
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime.fake_provider import (
    FakeAlmadenV2, FakeBoeblingenV2, FakeBrisbane, FakeBrooklynV2,
    FakeCairoV2, FakeCasablancaV2, FakeGeneva, FakeHanoiV2,
    FakeJohannesburgV2, FakeKolkataV2, FakeKyoto, FakeManhattanV2,
    FakeMontrealV2, FakeNairobiV2, FakeParisV2, FakeQuebec,
    FakeRochesterV2, FakeSherbrooke, FakeSingaporeV2, FakeSydneyV2,
    FakeTorontoV2, FakeValenciaV2, FakeWashingtonV2, FakeYorktownV2
)
from qiskit.quantum_info import hellinger_fidelity

# ==================== 辅助函数 ====================

def calculate_polarization_multiple_targets(counts: Dict[str, int], 
                                           correct_states: Optional[List[str]] = None) -> float:
    """
    计算测量结果的极化程度（支持多个正确状态）
    """
    if not counts:
        return 0.0
    
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # 获取量子比特数（线路宽度w）
    n_qubits = len(next(iter(counts.keys())))
    
    # 确定正确状态列表
    if correct_states is None:
        correct_states = ['0' * n_qubits]
    elif isinstance(correct_states, str):
        correct_states = [correct_states]
    
    # 调整状态长度
    adjusted_states = []
    for state in correct_states:
        if len(state) < n_qubits:
            state = state + '0' * (n_qubits - len(state))
        elif len(state) > n_qubits:
            state = state[:n_qubits]
        adjusted_states.append(state)
    
    # 计算成功率S（命中任意一个正确结果的概率）
    success_count = 0
    for state in adjusted_states:
        success_count += counts.get(state, 0)
    
    success_rate = success_count / total_shots
    
    # 计算分母中的项：1/2^w
    denominator_term = 1.0 / (2 ** n_qubits)
    
    # 计算极化程度P = (S - 1/2^w) / (1 - 1/2^w)
    if denominator_term == 1.0:  # 避免除以0
        return 0.0
    
    polarization = (success_rate - denominator_term) / (1.0 - denominator_term)
    
    # 限制极化程度在合理范围内（理论上应该在-1到1之间）
    polarization = max(min(polarization, 1.0), -1.0)
    
    return polarization

# ==================== 常量定义 ====================

class AnalysisMode(Enum):
    """分析模式枚举"""
    SINGLE_BACKEND = 'single_backend'
    MULTI_BACKEND = 'multi_backend'

# ==================== 配置类 ====================

@dataclass
class DataGeneratorConfig:
    """数据生成配置"""
    qasm_dir: str
    output_dir: str = "results"
    unified_csv: Optional[str] = None  # 改为可选，自动生成
    shots: int = 2000
    backends: List[str] = field(default_factory=lambda: ["FakeWashingtonV2"])
    algorithm: Optional[str] = None
    algorithms: Optional[List[str]] = None
    incremental_update: bool = True
    optimization_level: int = 1  # 添加编译优化级别
    
    def __post_init__(self):
        """初始化后自动生成统一CSV文件名"""
        if self.unified_csv is None:
            # 从qasm_dir提取目录名作为文件名前缀
            qasm_dir_name = os.path.basename(os.path.normpath(self.qasm_dir))
            self.unified_csv = f"{qasm_dir_name}_analysis_results_O{self.optimization_level}.csv"

# ==================== 数据生成器类 ====================

class QuantumCircuitDataGenerator:
    """量子电路数据生成器"""
    
    BACKEND_MAP = {
        "FakeAlmadenV2": FakeAlmadenV2,
        "FakeBoeblingenV2": FakeBoeblingenV2,
        "FakeBrisbane": FakeBrisbane,
        "FakeBrooklynV2": FakeBrooklynV2,
        "FakeCairoV2": FakeCairoV2,
        "FakeCasablancaV2": FakeCasablancaV2,
        "FakeGeneva": FakeGeneva,
        "FakeHanoiV2": FakeHanoiV2,
        "FakeJohannesburgV2": FakeJohannesburgV2,
        "FakeKolkataV2": FakeKolkataV2,
        "FakeKyoto": FakeKyoto,
        "FakeManhattanV2": FakeManhattanV2,
        "FakeMontrealV2": FakeMontrealV2,
        "FakeNairobiV2": FakeNairobiV2,
        "FakeParisV2": FakeParisV2,
        "FakeQuebec": FakeQuebec,
        "FakeRochesterV2": FakeRochesterV2,
        "FakeSherbrooke": FakeSherbrooke,
        "FakeSingaporeV2": FakeSingaporeV2,
        "FakeSydneyV2": FakeSydneyV2,
        "FakeTorontoV2": FakeTorontoV2,
        "FakeValenciaV2": FakeValenciaV2,
        "FakeWashingtonV2": FakeWashingtonV2,
        "FakeYorktownV2": FakeYorktownV2,
        "ideal": None
    }
    
    def __init__(self, config: DataGeneratorConfig):
        self.config = config
        self.processed_circuits = set()
        
        # 确保输出目录存在
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 统一CSV文件路径 - 改为在output_dir下
        # 检查unified_csv是否是绝对路径，如果不是，则使用output_dir
        if not os.path.isabs(self.config.unified_csv):
            self.unified_csv_path = os.path.join(self.config.output_dir, self.config.unified_csv)
        else:
            self.unified_csv_path = self.config.unified_csv
        
        # 加载已处理的数据
        self.load_processed_circuits()

    def load_processed_circuits(self) -> Set[Tuple[str, str]]:
        """加载已处理的电路"""
        processed = set()
        if os.path.exists(self.unified_csv_path):
            try:
                df = pd.read_csv(self.unified_csv_path)
                if not df.empty and 'Circuit' in df.columns and 'Backend' in df.columns:
                    for _, row in df.iterrows():
                        processed.add((str(row['Circuit']), str(row['Backend'])))
                    print(f"📊 已加载 {len(processed)} 个已处理的电路+后端组合")
            except Exception as e:
                print(f"⚠️ 读取统一CSV文件失败: {e}")
        return processed
    
    def is_processed(self, circuit_name: str, backend_name: str) -> bool:
        """检查电路是否已处理"""
        return (circuit_name, backend_name) in self.processed_circuits
    
    def calculate_error_rates(self, backend_name: str):
        """计算后端的平均错误率"""
        try:
            if backend_name == "ideal":
                return 0.0, 0.0, 0.0
            
            if backend_name not in self.BACKEND_MAP:
                backend_name = "FakeWashingtonV2"
            
            backend_class = self.BACKEND_MAP[backend_name]
            backend = backend_class()
            
            single_errors = []
            two_errors = []
            readout_errors = []
            
            # 从 properties 获取错误率
            if hasattr(backend, 'properties') and backend.properties() is not None:
                props = backend.properties()
                
                # 读出错误率
                for qubit in range(backend.num_qubits):
                    try:
                        readout_err = props.readout_error(qubit)
                        if readout_err is not None:
                            readout_errors.append(readout_err)
                    except:
                        pass
                
                # 单比特门错误率
                for gate in props.gates:
                    if gate.gate in ['sx', 'x', 'id', 'u1', 'u2', 'u3'] and len(gate.qubits) == 1:
                        for param in gate.parameters:
                            if param.name == 'gate_error' and param.value < 1.0:  # 过滤损坏的门
                                single_errors.append(param.value)
                
                # 两比特门错误率
                two_qubit_gates = ['cx', 'cz', 'ecr', 'cnot']
                for gate in props.gates:
                    if gate.gate in two_qubit_gates and len(gate.qubits) == 2:
                        for param in gate.parameters:
                            if param.name == 'gate_error' and param.value < 1.0:  # 过滤损坏的门
                                two_errors.append(param.value)
            
            # 从 target 获取错误率（作为补充）
            if hasattr(backend, 'target') and backend.target is not None:
                target = backend.target
                
                # 单比特门错误率
                for gate in ['x', 'sx', 'rz', 'id']:
                    if gate in target:
                        for inst in target[gate].values():
                            if hasattr(inst, 'error') and inst.error is not None and inst.error < 1.0:
                                single_errors.append(inst.error)
                
                # 两比特门错误率
                for gate in ['cx', 'cz', 'ecr']:
                    if gate in target:
                        for inst in target[gate].values():
                            if hasattr(inst, 'error') and inst.error is not None and inst.error < 1.0:
                                two_errors.append(inst.error)
            
            # 计算平均值
            avg_single_qubit_error = sum(single_errors) / len(single_errors) if single_errors else 0.0
            avg_two_qubit_error = sum(two_errors) / len(two_errors) if two_errors else 0.0
            avg_readout_error = sum(readout_errors) / len(readout_errors) if readout_errors else 0.0
            
            return avg_single_qubit_error, avg_two_qubit_error, avg_readout_error
        except Exception as e:
            print(f"Error calculating error rates for {backend_name}: {e}")
            return 0.0, 0.0, 0.0
    
    def save_to_unified_csv(self, result: Dict[str, Any]):
        """保存结果到统一CSV文件"""
        # 计算错误率
        single_qubit_error, two_qubit_error, readout_error = self.calculate_error_rates(result["backend"])
        
        row = {
            "Circuit": result["circuit_name"],
            "Algorithm": result["algorithm_name"],
            "Backend": result["backend"],
            "Width": result["width"],
            "Depth": result["depth"],
            "Total_Gates": result["total_gates"],
            "Single_Qubit_Gates": result["single_qubit_gates"],
            "Two_Qubit_Gates": result["two_qubit_gates"],
            "Measure_Gates": result["measure_gates"],
            "Optimization_Level": result.get("optimization_level", self.config.optimization_level),
            "Single_Qubit_Error_Rate": f"{single_qubit_error:.6f}",
            "Two_Qubit_Error_Rate": f"{two_qubit_error:.6f}",
            "Readout_Error_Rate": f"{readout_error:.6f}",
            "Hellinger_Fidelity": f"{result['hellinger_fidelity']:.6f}",
            "Polarization": f"{result['polarization']:.6f}",
            "Ideal_Counts_Full": result.get("ideal_counts_full", ""),
            "Noisy_Counts_Full": result.get("noisy_counts_full", ""),
            "Shots": self.config.shots,
            "Timestamp": result["timestamp"]
        }
        
        if os.path.exists(self.unified_csv_path):
            try:
                df_existing = pd.read_csv(self.unified_csv_path)
                
                # 检查是否需要添加新列
                new_columns = [col for col in row.keys() if col not in df_existing.columns]
                if new_columns:
                    for col in new_columns:
                        df_existing[col] = ""
                    print(f"🔄 添加新列: {', '.join(new_columns)}")
                
                mask = (df_existing['Circuit'] == row['Circuit']) & (df_existing['Backend'] == row['Backend'])
                if mask.any():
                    for key, value in row.items():
                        if key in df_existing.columns:
                            df_existing.loc[mask, key] = value
                    print(f"🔄 更新记录: {row['Circuit']} @ {row['Backend']}")
                else:
                    df_existing = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
                    print(f"✅ 添加新记录: {row['Circuit']} @ {row['Backend']}")
                df_existing.to_csv(self.unified_csv_path, index=False)
            except Exception as e:
                print(f"⚠️ 读取CSV文件失败，重新创建: {e}")
                self._append_to_csv(row, write_header=True)
        else:
            self._append_to_csv(row, write_header=True)
    
    def _append_to_csv(self, row: Dict, write_header: bool):
        """直接追加到CSV文件"""
        mode = 'a' if not write_header else 'w'
        with open(self.unified_csv_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    
    # ==================== 核心分析函数 ====================
    
    @staticmethod
    def extract_algorithm_name(circuit_name: str) -> str:
        """从电路名称提取算法名称"""
        return circuit_name.split('_')[0] if '_' in circuit_name else circuit_name

    def create_noise_model(self, backend_name: str) -> Optional[NoiseModel]:
        """根据后端名称创建噪声模型"""
        if backend_name == "ideal":
            return None
        
        if backend_name not in self.BACKEND_MAP:
            backend_name = "FakeWashingtonV2"
        
        try:
            backend_class = self.BACKEND_MAP[backend_name]
            fake_backend = backend_class()
            return NoiseModel.from_backend(fake_backend)
        except Exception:
            return None
    
    def run_noisy_simulation(self, qc: QuantumCircuit, backend_name: str) -> Tuple[Dict[str, int], Dict[str, int], float, float]:
        """使用噪声模型进行模拟"""
        noise_model = self.create_noise_model(backend_name)
        shots = self.config.shots
        
        try:
            # 1. 理想模拟（无噪声）
            simulator_ideal = AerSimulator()
            # 使用优化级别进行转译
            qc_transpiled_ideal = transpile(qc, simulator_ideal, optimization_level=self.config.optimization_level)
            result_ideal = simulator_ideal.run(qc_transpiled_ideal, shots=shots).result()
            counts_ideal = result_ideal.get_counts()
            
            # 2. 带噪声模拟
            if noise_model is not None:
                simulator_noisy = AerSimulator(noise_model=noise_model)
                # 使用相同的优化级别进行转译
                qc_transpiled_noisy = transpile(qc, simulator_noisy, optimization_level=self.config.optimization_level)
                result_noisy = simulator_noisy.run(qc_transpiled_noisy, shots=shots).result()
                counts_noisy = result_noisy.get_counts()
                # 使用转译后的电路统计门信息
                qc_transpiled = qc_transpiled_noisy
            else:
                # 无噪声情况，噪声计数等于理想计数
                counts_noisy = counts_ideal
                qc_transpiled = qc_transpiled_ideal
            
            # 3. 计算海林格保真度（基于计数分布）
            fidelity = hellinger_fidelity(counts_ideal, counts_noisy)
            
            # 4. 计算极化程度
            polarization = calculate_polarization_multiple_targets(counts_noisy, correct_states=counts_ideal)
            
            return counts_ideal, counts_noisy, fidelity, polarization, qc_transpiled
        except Exception as e:
            # 捕获所有模拟错误，包括Qiskit转译器的内部错误
            raise RuntimeError(f"Simulation error: {str(e)}")
    
    def analyze_single_circuit(self, qasm_file: str, backend_name: str) -> Dict[str, Any]:
        """分析单个QASM文件"""
        try:
            with open(qasm_file, 'r') as f:
                qasm_str = f.read()
            
            qc = QuantumCircuit.from_qasm_str(qasm_str)
            
            # 检查是否有测量操作
            if not any(op.name == 'measure' for op in qc.data):
                return {"error": "No measurement gates"}
            
            try:
                counts_ideal, counts_noisy, fidelity, polarization, qc_transpiled = self.run_noisy_simulation(qc, backend_name)
            except Exception as e:
                # 捕获Qiskit转译器的内部错误
                return {"error": f"Simulation error: {str(e)}"}
            
            # 计算编译后的门统计
            gate_counts = {}
            total_gates = 0
            single_gates = 0
            two_gates = 0
            measure_gates = 0
            
            for instr in qc_transpiled.data:
                gate_name = instr.operation.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
                
                if gate_name == 'measure':
                    measure_gates += 1
                elif len(instr.qubits) == 1:
                    single_gates += 1
                elif len(instr.qubits) == 2:
                    two_gates += 1
                
                total_gates += 1
            
            circuit_name = os.path.splitext(os.path.basename(qasm_file))[0]
            algorithm_name = self.extract_algorithm_name(circuit_name)
            
            # 转换计数为字符串格式
            def counts_to_string(counts):
                return '; '.join([f"{state}:{count}" for state, count in counts.items()])
            
            ideal_counts_full = counts_to_string(counts_ideal)
            noisy_counts_full = counts_to_string(counts_noisy)
            
            return {
                "circuit_name": circuit_name,
                "algorithm_name": algorithm_name,
                "width": qc_transpiled.num_qubits,  # 使用编译后的宽度
                "depth": qc_transpiled.depth(),  # 使用编译后的深度
                "total_gates": total_gates,  # 使用编译后的门统计
                "single_qubit_gates": single_gates,
                "two_qubit_gates": two_gates,
                "measure_gates": measure_gates,
                "backend": backend_name,
                "hellinger_fidelity": fidelity,
                "polarization": polarization,
                "ideal_counts": counts_ideal,
                "noisy_counts": counts_noisy,
                "ideal_counts_full": ideal_counts_full,
                "noisy_counts_full": noisy_counts_full,
                "optimization_level": self.config.optimization_level,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_single_backend(self, backend_name: str) -> Tuple[int, int]:
        """分析单个后端的所有电路"""
        print(f"\n🔬 分析后端: {backend_name}")
        
        qasm_files = [f for f in os.listdir(self.config.qasm_dir) if f.endswith(".qasm")]
        qasm_files.sort()
        
        print(f"发现 {len(qasm_files)} 个QASM文件")
        print(f"测量次数: {self.config.shots} 次/模拟")
        print(f"编译优化级别: O{self.config.optimization_level}")
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        for idx, filename in enumerate(qasm_files, 1):
            circuit_name = os.path.splitext(filename)[0]
            
            if self.config.incremental_update and self.is_processed(circuit_name, backend_name):
                print(f"[{idx}/{len(qasm_files)}] 跳过: {filename} @ {backend_name}")
                skipped_count += 1
                continue
            
            print(f"[{idx}/{len(qasm_files)}] 处理: {filename} @ {backend_name}")
            
            filepath = os.path.join(self.config.qasm_dir, filename)
            result = self.analyze_single_circuit(filepath, backend_name)
            
            if "error" in result:
                error_count += 1
                print(f"    ❌ 错误: {result['error']}")
            else:
                success_count += 1
                self.save_to_unified_csv(result)
                self.processed_circuits.add((result["circuit_name"], result["backend"]))
                print(f"    ✅ 成功: Fidelity={result['hellinger_fidelity']:.4f}, Depth={result['depth']}")
        
        print(f"✅ 完成: 成功={success_count}, 错误={error_count}, 跳过={skipped_count}")
        return success_count, error_count
    
    def run(self) -> Dict[str, Any]:
        """运行数据生成流程"""
        print("="*60)
        print("量子电路数据生成器")
        print("="*60)
        print(f"📊 数据存储: {self.unified_csv_path}")
        print(f"📂 QASM目录: {self.config.qasm_dir}")
        print(f"🔧 编译优化级别: O{self.config.optimization_level}")
        print("="*60)
        
        results = {
            'config': self.config.__dict__,
            'unified_csv': self.unified_csv_path,
            'analysis_results': {}
        }
        
        # 检查现有数据
        if os.path.exists(self.unified_csv_path):
            df = pd.read_csv(self.unified_csv_path)
            print(f"📊 现有数据: {len(df)} 条记录")
            print(f"  后端: {sorted(df['Backend'].unique())}")
            print(f"  算法: {sorted(df['Algorithm'].unique())}")
            
            # 加载现有记录到processed_circuits
            for _, row in df.iterrows():
                self.processed_circuits.add((row['Circuit'], row['Backend']))
            print(f"🔄 加载了 {len(self.processed_circuits)} 条已处理记录")
        
        # 执行分析
        for backend in self.config.backends:
            success, error = self.analyze_single_backend(backend)
            results['analysis_results'][backend] = {'success': success, 'error': error}
        
        # 显示最终数据统计
        if os.path.exists(self.unified_csv_path):
            df = pd.read_csv(self.unified_csv_path)
            print(f"\n📊 最终数据统计:")
            print(f"  总记录数: {len(df)}")
            print(f"  后端数量: {len(df['Backend'].unique())}")
            print(f"  算法数量: {len(df['Algorithm'].unique())}")
            print(f"  量子比特范围: {df['Width'].min()} - {df['Width'].max()}")
        
        print("\n" + "="*60)
        print("✅ 数据生成完成!")
        print("="*60)
        
        return results

# ==================== 快捷函数 ====================

def quick_generate_data(qasm_dir: str, 
                      backends: Optional[List[str]] = None,
                      shots: int = 2000,
                      incremental_update: bool = True,
                      optimization_level: int = 1) -> Dict[str, Any]:
    """
    简化的快速数据生成函数
    
    参数:
        qasm_dir: QASM文件目录
        backends: 后端列表
        shots: 测量次数
        incremental_update: 是否增量更新
        optimization_level: 编译优化级别 (0, 1, 2, 3)
    """
    if not backends:
        backends = ["FakeWashingtonV2"]
    
    config = DataGeneratorConfig(
        qasm_dir=qasm_dir,
        backends=backends,
        shots=shots,
        incremental_update=incremental_update,
        optimization_level=optimization_level
    )
    
    generator = QuantumCircuitDataGenerator(config)
    return generator.run()

# ==================== 主程序入口 ====================

if __name__ == "__main__":
    # 生成-O1数据集
    print("="*60)
    print("生成 MQTBench 数据集 - O1 优化级别")
    print("="*60)
    
    # 设置路径
    qasm_dir = "MQTBench"  # QASM文件目录
    output_dir = "results"  # 输出目录
    csv_filename = "MQTBench_578_O1.csv"
    
    # 检查目录是否存在
    if not os.path.exists(qasm_dir):
        print(f"❌ 错误: QASM目录 '{qasm_dir}' 不存在")
        print(f"请确保当前目录下有 MQTBench 文件夹，包含578个QASM文件")
        exit(1)
    
    # 统计QASM文件数量
    qasm_files = [f for f in os.listdir(qasm_dir) if f.endswith('.qasm')]
    print(f"📁 发现 {len(qasm_files)} 个QASM文件")
    
    if len(qasm_files) == 0:
        print("❌ 错误: 未找到任何QASM文件")
        exit(1)
    
    # 创建配置
    config = DataGeneratorConfig(
        qasm_dir=qasm_dir,
        output_dir=output_dir,
        unified_csv=csv_filename,
        shots=2000,
        backends=["FakeWashingtonV2"],
        incremental_update=True,
        optimization_level=1  # O1优化
    )
    
    # 创建生成器并运行
    generator = QuantumCircuitDataGenerator(config)
    results = generator.run()
    
    print(f"\n🎉 完成! 结果保存在: {os.path.join(output_dir, csv_filename)}")