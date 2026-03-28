# Preflight チェック — Cycle 3 (Phase 3: Walk-Forward Evaluation)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2023-12-12 (API返却の最終日、今日2026-03-28以前) |
| Train期間 | 2021-03-29 〜 各foldのtrain_end (ローリングウィンドウ) |
| Validation期間 | N/A (walk-forwardではtrain/testのみ) |
| Test期間 | 各foldのtest_start 〜 test_end (ローリングウィンドウ) |
| 重複なし確認 | Yes — WalkForwardValidatorのgap=1でtrain/test間に重複なし |
| 未来日付なし確認 | Yes — API end_dateは2023-12-12、未来データなし |

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes
  - ルックバックウィンドウはt-lookback〜t-1のリターンデータを使用
  - center=False のrolling windowのみ使用
- Scaler / Imputer は train データのみで fit しているか？ → Yes
  - 各walk-forward foldでtrain期間のみでscaler.fit()
- Centered rolling window を使用していないか？ → Yes (使用していない)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | 主要ETFs | SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, VNQ, DBC (10銘柄) | Yes |
| ルックバック期間 | 60日 (推定、論文未明示) | 60日 | Yes |
| リバランス頻度 | 日次 | 日次 | Yes |
| 特徴量 | 過去リターン系列 | 過去60日間の日次リターン | Yes |
| コストモデル | 論文未明示 | fee=10bps, slippage=5bps | N/A (論文未明示) |
| モデル | LSTM + Softmax出力 | LSTM + Softmax出力 | Yes |
| 損失関数 | 負のSharpe比 | 負のSharpe比 | Yes |
| Walk-forward分割数 | 論文未明示 | n_splits=5 | N/A (Phase3指示に従う) |

**preflight.md 作成完了 — 実装に進む。**
