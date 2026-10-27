# Initial flat control

Hypothesis before execution: a flat strategy supplies a deterministic, causal
zero-PnL control for testing the harness. It should fail the mandatory trade
count and positive-net gates and receive a final score of zero.

The declared market is ETH, the decision clock is 1h, and the position is always
zero. There are no fitted parameters, thresholds, pretrained weights, engineered
features or random draws. This is a setup control, not a strategy-edge claim.

The grader owns execution, costs, funding and every acceptance gate. Future
changes must state their mechanism here before a trial. Checkpoints belong in
the supplied scratch directory; hidden inputs and artifacts stay in the private
service. No order submission or holdout evaluation is part of this entry point.
