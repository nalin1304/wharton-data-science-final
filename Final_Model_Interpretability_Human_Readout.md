# Human Readout: How The Model Makes Calls

The final prediction blends several model opinions. Elo supplies baseline team strength, while feature-driven models add matchup context.

## Blend Weights
- Elo: 0.691
- SGD: 0.248
- LogReg: 0.057
- XGB: 0.003
- XGB2: 0.002

## Most Influential Home-Leaning Factors
- PDO (shooting + save luck proxy): stronger home edge usually helps (importance 14.7%, stability High).
- win percentage: stronger home edge usually helps (importance 9.5%, stability High).
- rolling xGF (20 games): stronger home edge usually helps (importance 7.9%, stability Low).
- xG trend x rest interaction: stronger home edge usually helps (importance 6.4%, stability Low).
- even-strength xG share: stronger home edge usually helps (importance 3.7%, stability Low).

## Most Influential Away-Leaning Factors
- close-game win percentage: stronger away edge usually hurts home win odds (importance 10.8%, stability High).
- pregame Elo rating: stronger away edge usually hurts home win odds (importance 8.5%, stability Medium).
- shelter index: stronger away edge usually hurts home win odds (importance 5.7%, stability High).
- recent form (last 5 games): stronger away edge usually hurts home win odds (importance 5.5%, stability High).
- rolling xGA (20 games): stronger away edge usually hurts home win odds (importance 5.2%, stability Low).

## What To Trust Most
- Trust holdout log loss/AUC for performance.
- Trust this readout for directionality and communication.
