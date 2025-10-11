#!/bin/bash
# 运行所有baselines并收集Fidelity-统计数据

echo "========================================="
echo "Running all baselines with Fidelity- metrics"
echo "========================================="

# 设置工作目录
cd /Users/anthonyche/Desktop/Research/GroundingGEXP

# 1. ApxChase
echo ""
echo "===== Running ApxChase ====="
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['exp_name'] = 'apxchase_mutag'
config['save_dir'] = 'results/mutag_apxchase'
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
"
python src/Run_Experiment.py --run_all 2>&1 | tee apxchase_fidelity_run.log

# 2. ExhaustChase
echo ""
echo "===== Running ExhaustChase ====="
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['exp_name'] = 'exhaustchase_mutag'
config['save_dir'] = 'results/mutag_exhaustchase'
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
"
python src/Run_Experiment.py --run_all 2>&1 | tee exhaustchase_fidelity_run.log

# 3. HeuChase
echo ""
echo "===== Running HeuChase ====="
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
config['exp_name'] = 'heuchase_mutag'
config['save_dir'] = 'results/mutag_heuchase'
with open('config.yaml', 'w') as f:
    yaml.dump(config, f)
"
python src/Run_Experiment.py --run_all 2>&1 | tee heuchase_fidelity_run.log

# 提取所有方法的Fidelity-统计
echo ""
echo "========================================="
echo "Fidelity- Summary Comparison"
echo "========================================="
echo ""
echo "ApxChase:"
grep "Overall Average Fidelity-" apxchase_fidelity_run.log
echo ""
echo "ExhaustChase:"
grep "Overall Average Fidelity-" exhaustchase_fidelity_run.log
echo ""
echo "HeuChase:"
grep "Overall Average Fidelity-" heuchase_fidelity_run.log
