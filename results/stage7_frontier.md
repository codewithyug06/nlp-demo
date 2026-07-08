# Stage 7 — lambda_c quality-vs-compute frontier (needle, 250 steps)

Config: controller + attn_budget + budget_cost + latent_ponder + ponder_kl + mtp + calibration

| lambda_c | answer_acc | eval_loss | E[ponder steps] | mean_g |
|---------:|-----------:|----------:|----------------:|-------:|
|     0.00 |      1.000 |   0.00872 |           1.757 |  0.499 |
|     0.20 |      1.000 |   0.00829 |           1.003 |  0.499 |
|     0.60 |      1.000 |   0.00838 |           1.001 |  0.499 |

lambda_c monotonically reduces compute (E[ponder] 1.76 -> 1.00) at ~equal quality:
on the (easy) needle task the extra pondering is unnecessary, so adaptive compute
correctly collapses to the cheap point. A harder task would trade quality for compute.
