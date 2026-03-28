# Preflight Check — Cycle 4 (Phase 4: Transaction Cost Model)

## 1. データ境界表

| 項目 | 値 |
|---|---|
| データ取得終了日 | 2026-03-27 (今日 2026-03-28 以前) |
| Train期間 | 2021-04-01 〜 各ウィンドウで可変 (expanding) |
| Validation期間 | N/A (walk-forward OOS) |
| Test期間 | 各ウィンドウで可変 〜 2026-03-27 |
| 重複なし確認 | Yes |
| 未来日付なし確認 | Yes |

## 2. Feature timestamp 契約

- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → **Yes**
  - `prepare_sequences()` は `data[t-lookback:t]` を特徴量、`data[t]` をターゲットとする
- Scaler / Imputer は train データのみで fit しているか？ → **Yes**
  - `evaluation.py` で `scaler.fit(X_train_flat)` のみ実行
- Centered rolling window を使用していないか？ → **Yes** (使用していない)

## 3. Paper spec 差分表

| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | 10 major US ETFs | SPY,QQQ,IWM,EFA,EEM,TLT,LQD,GLD,VNQ,DBC (10 ETFs) | Yes |
| ルックバック期間 | 未指定 (60日を使用) | 60日 | Yes |
| リバランス頻度 | 日次 | 日次 | Yes |
| 特徴量 | 日次リターン | 日次リターン | Yes |
| コストモデル | 未指定 | 10bps fee + 5bps slippage (Phase 4: 感度分析追加) | Yes |
| 損失関数 | 負のSharpe比 | 負のSharpe比 | Yes |

## 4. Phase 4 固有チェック

- Phase 3の結果を基盤として使用: gross Sharpe 0.9899, net Sharpe 0.8945 (15bps総コスト)
- コスト感度分析の範囲: 0bps, 5bps, 10bps, 15bps, 20bps, 30bps (論文近傍)
- グロスとネットの両方のパフォーマンスを報告する
