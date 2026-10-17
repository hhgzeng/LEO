# Optimization Performance Metrics Table

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
