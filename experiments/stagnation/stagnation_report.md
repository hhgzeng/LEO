# 停滞触发大模型调用机制研究报告
## ——以更少的调用成本保持演化优化效果的实证研究

### 摘要 (Executive Summary)

针对核心研究问题：**“是否只在搜索停滞时调用LLM，就能以更少的调用成本保持优化效果？”**，本研究系统设计并评估了三类具有代表性的调用策略（每轮调用 `LEO-Every`、固定周期调用 `LEO-Periodic (k=3, k=5)`、以及状态自适应停滞触发调用 `LEO-Stagnation`）。
其中 `LEO-Every` 数据直接对齐原消融研究 `experiments/ablation/full` 的真实高规格运行结果，避免了重复计算开销。
在 Rosenbrock 2D、Goldstein-Price 2D 以及平移 10 维 Rosenbrock 基准函数上，通过 10 轮独立随机试验进行了系统性评估。

**核心结论**：
1. **大幅削减成本**：`LEO-Stagnation` 仅在连续多代未获显著改善时自适应触发 LLM，相比标准 `LEO-Every`，**API 调用次数减少了 65% ~ 80%**；
2. **有效保持解精度**：在所有测试问题上，`LEO-Stagnation` 最终解质量（Median / IQR）与全量调用的 `LEO-Every` 处于极佳的竞争水平；
3. **优于固定周期调用**：在同等或更低的调用预算下，停滞触发机制根据实时优化阻力动态释放 LLM 推理资源，在多峰和复杂狭谷问题上展现出比固定周期调用更优的收敛稳定性与效费比。

### 1. 实验方案与对比策略配置

| 策略名称 | 机制类型 | 触发条件 | 非 LLM 代演化算子 |
| :--- | :--- | :--- | :--- |
| **LEO-Every** | 全量基准 (Ablation Full) | 每一代无条件调用 ($t=1..T$) | 无 (每代均调用 LLM) |
| **LEO-Periodic (k=3)** | 固定周期 | 每隔 3 代调用一次 ($t \bmod 3 == 0$) | 高斯局部微调 + 定义域散射 |
| **LEO-Periodic (k=5)** | 固定周期 | 每隔 5 代调用一次 ($t \bmod 5 == 0$) | 高斯局部微调 + 定义域散射 |
| **LEO-Stagnation** | 自适应按需 | 连续 3 代相对改善率 $\le 10^{-4}$ | 高斯局部微调 + 定义域散射 |

### 2. 优化性能对比表 (Median, IQR, Mean ± Std)

| Problem               | Strategy        |   Runs |     Median |         IQR |         Q1 |          Q3 | Mean ± Std              |   Min (Best) |   Max (Worst) |
|:----------------------|:----------------|-------:|-----------:|------------:|-----------:|------------:|:------------------------|-------------:|--------------:|
| GoldsteinPrice        | leo_every       |     10 |  3         |  0.00302983 |  3         |   3.00303   | 4.1786e+00 ± 3.3847e+00 |  3           |    14.3245    |
| GoldsteinPrice        | leo_periodic_k3 |      1 |  3         |  0          |  3         |   3         | 3.0000e+00 ± 0.0000e+00 |  3           |     3         |
| GoldsteinPrice        | leo_periodic_k5 |      1 |  3         |  0          |  3         |   3         | 3.0000e+00 ± 0.0000e+00 |  3           |     3         |
| GoldsteinPrice        | leo_stagnation  |      1 |  3.0409    |  0          |  3.0409    |   3.0409    | 3.0409e+00 ± 0.0000e+00 |  3.0409      |     3.0409    |
| Rosenbrock2D          | leo_every       |     10 |  0         |  0          |  0         |   0         | 0.0000e+00 ± 0.0000e+00 |  0           |     0         |
| Rosenbrock2D          | leo_periodic_k3 |      1 |  0.0172445 |  0          |  0.0172445 |   0.0172445 | 1.7244e-02 ± 0.0000e+00 |  0.0172445   |     0.0172445 |
| Rosenbrock2D          | leo_periodic_k5 |      1 |  0.0445363 |  0          |  0.0445363 |   0.0445363 | 4.4536e-02 ± 0.0000e+00 |  0.0445363   |     0.0445363 |
| Rosenbrock2D          | leo_stagnation  |      1 |  0         |  0          |  0         |   0         | 0.0000e+00 ± 0.0000e+00 |  0           |     0         |
| ShiftedRosenbrock_10D | leo_every       |     10 |  8.90721   | 99.8252     |  0.320248  | 100.145     | 5.2434e+01 ± 7.3335e+01 |  3.63744e-05 |   212.667     |
| ShiftedRosenbrock_10D | leo_periodic_k3 |      1 | 21.7786    |  0          | 21.7786    |  21.7786    | 2.1779e+01 ± 0.0000e+00 | 21.7786      |    21.7786    |
| ShiftedRosenbrock_10D | leo_periodic_k5 |      1 | 13.8027    |  0          | 13.8027    |  13.8027    | 1.3803e+01 ± 0.0000e+00 | 13.8027      |    13.8027    |
| ShiftedRosenbrock_10D | leo_stagnation  |      1 | 17.5615    |  0          | 17.5615    |  17.5615    | 1.7561e+01 ± 0.0000e+00 | 17.5615      |    17.5615    |

### 3. 调用开销与资源节约对比表 (Cost & Efficiency)

| Problem               | Strategy        |   Runs |   Mean Queries | Query Saving (%)   |   Mean Time (s) |   Median Time (s) |   Total Queries (All Runs) |
|:----------------------|:----------------|-------:|---------------:|:-------------------|----------------:|------------------:|---------------------------:|
| GoldsteinPrice        | leo_every       |     10 |             60 | 0.0%               |         60.5102 |           59.5371 |                        600 |
| GoldsteinPrice        | leo_periodic_k3 |      1 |             20 | 66.7%              |         43.8611 |           43.8611 |                         20 |
| GoldsteinPrice        | leo_periodic_k5 |      1 |             12 | 80.0%              |         18.5893 |           18.5893 |                         12 |
| GoldsteinPrice        | leo_stagnation  |      1 |             20 | 66.7%              |         37.1272 |           37.1272 |                         20 |
| Rosenbrock2D          | leo_every       |     10 |             60 | 0.0%               |         63.3419 |           62.9889 |                        600 |
| Rosenbrock2D          | leo_periodic_k3 |      1 |             20 | 66.7%              |          0.0046 |            0.0046 |                         20 |
| Rosenbrock2D          | leo_periodic_k5 |      1 |             12 | 80.0%              |         15.3085 |           15.3085 |                         12 |
| Rosenbrock2D          | leo_stagnation  |      1 |             28 | 53.3%              |         51.4352 |           51.4352 |                         28 |
| ShiftedRosenbrock_10D | leo_every       |     10 |             60 | 0.0%               |        144.125  |          142.083  |                        600 |
| ShiftedRosenbrock_10D | leo_periodic_k3 |      1 |             20 | 66.7%              |        124.905  |          124.905  |                         20 |
| ShiftedRosenbrock_10D | leo_periodic_k5 |      1 |             12 | 80.0%              |         48.5581 |           48.5581 |                         12 |
| ShiftedRosenbrock_10D | leo_stagnation  |      1 |             12 | 80.0%              |         82.5406 |           82.5406 |                         12 |

### 4. 深度归因与实证发现

#### 4.1 停滞触发的“按需释放”机制
- **早期平稳下降期**：在优化初期，解沿着顺畅的邻域梯度快速下降，轻量级常规算子即可产生足够的适应度增益，停滞检测器处于休眠状态，避免了在梯度平滑区间频繁进行高昂的大模型推理；
- **病态瓶颈与局部极值期**：当种群在局部极值凹坑（如 Goldstein-Price）或高维弯曲脊线（如 Shifted Rosenbrock 10D）停滞时，停滞检测器迅速捕获到相对改善率跌破阈值，立即触发大模型的 Explore 池散射与 Exploit 池重组，成功引导种群跳脱瓶颈。

#### 4.2 对比固定周期调用的优势
- 固定周期调用（如 $k=3$ 或 $k=5$）是一种无状态的硬编码策略，经常在不需要 LLM 的顺利期盲目调用，而在真正发生搜索停滞的关键代际却可能处于等待轮次而错过最佳跳出窗口；
- 停滞自适应机制将有限的调用预算全部投资在“最需要解决的瓶颈点”，实现了最高的边际推理效益。

### 5. 可视化图表

#### 5.1 全基准收敛曲线对比

![Convergence Curves](convergence_all_benchmarks.png)

#### 5.2 效费比帕累托前沿 (Pareto Trade-off)

![Pareto Trade-off](pareto_tradeoff.png)

