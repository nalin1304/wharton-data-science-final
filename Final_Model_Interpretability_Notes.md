# Final Model Interpretability Notes

## Blend Structure
Selected blend uses a weighted average of model probabilities.
- Elo: 0.530, - SGD: 0.470, - LogReg: 0.000, - XGB: 0.000, - XGB2: 0.000

## Strongest Signals Supporting Home Teams
- PDO (shooting + save luck proxy) (std coef 0.314)
- rolling xGF (20 games) (std coef 0.252)
- win percentage (std coef 0.199)
- even-strength xG share (std coef 0.079)
- pythagorean win expectation (std coef 0.058)

## Strongest Signals Supporting Away Teams
- close-game win percentage (std coef -0.229)
- rolling xGA (20 games) (std coef -0.189)
- pregame Elo rating (std coef -0.177)
- shelter index (std coef -0.121)
- recent form (last 5 games) (std coef -0.116)

## Base Model Context (Holdout LL)
- Blend_Selected_By_OOF: LL=0.669612, Acc=0.5786
- Elo: LL=0.669759, Acc=0.5821
- SGD: LL=0.672006, Acc=0.5964
- XGB: LL=0.673780, Acc=0.5786
- XGB2: LL=0.676152, Acc=0.6000
