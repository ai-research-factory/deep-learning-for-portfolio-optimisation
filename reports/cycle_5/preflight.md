# Preflight Check — Cycle 5 (Phase 5: Hyperparameter Optimization with Optuna)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-27 (今日2026-03-29以前) |
| Train期間 | 2021-04-01 〜 各ウィンドウのtrain終了日 (expanding) |
| Validation期間 | N/A (Optuna objective = avg net Sharpe across WF windows) |
| Test期間 | 各ウィンドウのtest開始日 〜 test終了日 |
| 重複なし確認 | Yes (WalkForwardValidator enforces gap=1, no overlap) |
| 未来日付なし確認 | Yes (data end date 2026-03-27 < today 2026-03-29) |

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes
  - `prepare_sequences()` uses data[t-lookback:t] for features, target is data[t]
- Scaler / Imputer は train データのみで fit しているか？ → Yes
  - `scaler.fit(X_train_flat)` in `_run_windows()`, line 109 of evaluation.py
- Centered rolling window を使用していないか？ → Yes (not used)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | 10 US ETFs | SPY,QQQ,IWM,EFA,EEM,TLT,LQD,GLD,VNQ,DBC (10 ETFs) | Yes |
| ルックバック期間 | Not specified (default 60) | 60 days (to be optimized) | Yes |
| リバランス頻度 | Daily | Daily | Yes |
| 特徴量 | Historical returns | Historical returns | Yes |
| コストモデル | Not specified | 10bps fee + 5bps slippage | N/A |
| 学習率 | Not specified | 1e-4 (to be optimized) | N/A |
| LSTM hidden size | Not specified | 64 (to be optimized) | N/A |
| LSTM layers | Not specified | 2 (to be optimized) | N/A |

## 4. Phase 5 特記事項

- Optuna最適化の目的関数: ウォークフォワード全窓の平均ネットSharpe比
- 探索空間は論文デフォルト近傍に制約:
  - lookback: [30, 45, 60, 90, 120] (論文デフォルト60の近傍)
  - lr: [5e-5, 1e-4, 5e-4, 1e-3] (論文デフォルト1e-4の近傍)
  - hidden_size: [32, 64, 128] (論文デフォルト64の近傍)
  - num_layers: [1, 2, 3] (論文デフォルト2の近傍)
  - epochs: [30, 50, 100] (論文デフォルト50の近傍)
- テストデータでのハイパーパラメータ調整は行わない (WF OOS Sharpeが目的関数)
