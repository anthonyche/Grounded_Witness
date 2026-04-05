# DBLP主链最终验证结果

## 当前结论

- `dblp_smoke` 已完整跑完。
- `dblp_small_regression` 已完成；`Exh` 在 600s 超时下完成了 19/30 个 workload 的评估，其余方法完整跑完。
- sampled author workload 的规模分布没有极端偏斜，但存在少量较大的 2-hop 子图。
- 当前 regression 使用的 20 条 resolved constraints 中，`c` 至少命中过一次的约束数为 20/20。

## smoke 方法级结果

| method | status | num_targets | avg_num_witnesses | avg_hit_constraint_count | avg_active_constraint_count | avg_covered_constraint_count | coverage_global | coverage_normalized | conciseness | fidelity_minus | runtime_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HeuC | completed | 3 | 2.0000 | 18.6667 | 2.0000 | 2.0000 | 0.1000 | 0.3333 | 0.2069 | 0.0091 | 6.7768 |
| GEX | completed | 3 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8239 | 0.3740 | 5.6510 |
| ApxC | completed | 3 | 2.0000 | 18.6667 | 2.6667 | 2.6667 | 0.1333 | 0.3333 | 0.2711 | 0.2031 | 7.7605 |
| PGX | completed | 3 |  | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.8239 | 0.4747 | 5.3915 |
| Exh | timeout | 2 | 0.0000 | 18.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 100.0468 |

## small regression 方法级结果

| method | status | num_targets | avg_num_witnesses | avg_hit_constraint_count | avg_active_constraint_count | avg_covered_constraint_count | coverage_global | coverage_normalized | conciseness | fidelity_minus | runtime_total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HeuC | completed | 30 | 2.2000 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.2187 | 0.0890 | 26.3686 |
| ApxC | completed | 30 | 2.2000 | 15.6000 | 1.9333 | 1.9333 | 0.0967 | 0.3667 | 0.0187 | 0.0117 | 25.6671 |
| Exh | timeout | 19 | 2.5263 | 15.5789 | 2.4211 | 2.4211 | 0.1211 | 0.4211 | 0.0187 | 0.0311 | 600.0455 |

## sampled workload 规模分布

- 样本数: 30
- 子图节点数: min=5 , p50=19.5, p90=50.1, max=79
- 子图边数: min=8 , p50=44.0, p90=120.6, max=210
- 节点规模最大值 / 中位数 ≈ 4.05

## consequent 可匹配性

- resolved constraint 数量: 20
- `c` 至少命中过一次的约束数: 20/20
- 前 10 条 consequent 命中情况：

| constraint_name | consequent_workload_hit_count | consequent_match_count |
| --- | --- | --- |
| mined_dblp_aat_2_19_3_c1_p1_t_1 | 30 | 474 |
| mined_dblp_aat_2_19_3_c2_p2_t_1 | 30 | 474 |
| mined_dblp_aat_0_19_3_c1_p1_t_1 | 30 | 474 |
| mined_dblp_aat_0_19_3_c2_p2_t_1 | 30 | 474 |
| mined_dblp_aat_1_19_3_c1_p1_t_1 | 30 | 474 |
| mined_dblp_aat_1_19_3_c2_p2_t_1 | 30 | 474 |
| mined_dblp_aat_1_19_2_c1_p1_t_1 | 30 | 474 |
| mined_dblp_aat_1_19_2_c2_p2_t_1 | 30 | 474 |
| mined_dblp_aat_2_18_3_c1_p1_t_1 | 19 | 68 |
| mined_dblp_aat_2_18_3_c2_p2_t_1 | 19 | 68 |

## 文件

- 主结论: `/Users/anthonyche/Desktop/Research/GroundingGEXP/DBLP主链最终验证结果.md`
- 方法汇总: `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_mainline_validation_summary.csv`
- sampled workload 规模明细: `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_small_regression_sample_sizes.csv`
- consequent 可匹配性明细: `/Users/anthonyche/Desktop/Research/GroundingGEXP/outputs/csv/dblp_small_regression_consequent_matchability.csv`
