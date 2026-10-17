# LLM Invocation Cost & Efficiency Table

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
