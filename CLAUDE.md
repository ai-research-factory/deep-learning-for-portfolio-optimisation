# Deep Learning for Portfolio Optimisation

## Project ID
proj_e5603456

## Taxonomy
ReinforcementLearning, Other

## Current Cycle
2

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
Traditional portfolio optimization, such as Markowitz's Mean-Variance Optimization, is a two-step process: first, forecast future asset returns and covariances, and second, use these forecasts as inputs to an optimization solver to find optimal portfolio weights. This process is highly sensitive to estimation errors in the first step, which often leads to unstable and poorly performing portfolios. This paper proposes an end-to-end deep learning framework that bypasses the explicit forecasting step. The model directly ingests historical market data and outputs portfolio weights, with the neural network's parameters being trained to directly maximize the portfolio's Sharpe ratio on the training data. This allows the model to learn complex, non-linear patterns from the data to construct a dynamically rebalanced portfolio.

### Datasets
yfinance API for a universe of US-listed ETFs. Example universe: ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'HYG', 'GLD', 'DBC']

### Targets
The primary optimization target is the portfolio's Sharpe Ratio. The model's direct output is a vector of portfolio weights for the next rebalancing period.

### Model
The model is a deep neural network (likely an LSTM or CNN) that takes a lookback window of historical market data (e.g., 60 days of price returns) for a universe of assets as input. The final layer of the network is a softmax layer, which outputs a vector of positive portfolio weights that sum to one (enforcing a long-only, fully invested strategy). The crucial component is the custom loss function, which is the negative of the portfolio's Sharpe ratio, made differentiable to allow for gradient-based optimization. The network is trained end-to-end to learn a policy that maps market history directly to portfolio allocations that maximize future Sharpe ratio.

### Training
The model is trained using a walk-forward validation scheme. For each fold, the model is trained on a rolling window of historical data (e.g., 5 years) to maximize the in-sample Sharpe ratio. The trained model is then used to generate portfolio weights for the subsequent out-of-sample period (e.g., 1 year). The model is then retrained on an updated rolling window that includes the previous test period. This process is repeated across the entire dataset. The optimizer is typically Adam.

### Evaluation
The primary evaluation metric is the out-of-sample Sharpe Ratio, aggregated across all walk-forward splits. Other key metrics include Annualized Return, Annualized Volatility, Maximum Drawdown, and Portfolio Turnover. Performance is evaluated on both a gross basis and a net basis (after accounting for transaction costs). The model's performance will be compared against several baselines: an equal-weight (1/N) portfolio, a market-cap weighted benchmark (e.g., SPY), and potentially a traditional Mean-Variance Optimization (MVO) strategy.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## Preflight チェック（実装開始前に必ず実施）

**Phase の実装コードを書く前に**、以下のチェックを実施し結果を `reports/cycle_2/preflight.md` に保存すること。

### 1. データ境界表
以下の表を埋めて、未来データ混入がないことを確認:

```markdown
| 項目 | 値 |
|---|---|
| データ取得終了日 | YYYY-MM-DD (今日以前であること) |
| Train期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Validation期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| Test期間 | YYYY-MM-DD 〜 YYYY-MM-DD |
| 重複なし確認 | Yes / No |
| 未来日付なし確認 | Yes / No |
```

### 2. Feature timestamp 契約
- 全ての特徴量は時刻 t の予測に t-1 以前のデータのみを使用しているか？ → Yes / No
- Scaler / Imputer は train データのみで fit しているか？ → Yes / No
- Centered rolling window を使用していないか？ → Yes / No (使用していたら修正)

### 3. Paper spec 差分表
論文の主要パラメータと現在の実装を比較:

```markdown
| パラメータ | 論文の値 | 現在の実装 | 一致? |
|---|---|---|---|
| ユニバース | (論文の記述) | (実装の値) | Yes/No |
| ルックバック期間 | (論文の記述) | (実装の値) | Yes/No |
| リバランス頻度 | (論文の記述) | (実装の値) | Yes/No |
| 特徴量 | (論文の記述) | (実装の値) | Yes/No |
| コストモデル | (論文の記述) | (実装の値) | Yes/No |
```

**preflight.md が作成されるまで、Phase の実装コードに進まないこと。**

## ★ 今回のタスク (Cycle 2)


### Phase 2: データパイプライン構築 [Track ]

**Track**:  (A=論文再現 / B=近傍改善 / C=独自探索)
**ゴール**: yfinanceを使用してETFデータを取得し、モデルの入力形式に前処理するパイプラインを構築する。

**具体的な作業指示**:
1. `src/data_pipeline.py`を作成する。
2. `download_etf_data`関数を実装し、ティッカーリスト `['SPY', 'TLT', 'GLD', 'QQQ', 'EFA']` と期間 `2010-01-01` から `2023-12-31` の日足データをyfinanceで取得し、`data/raw/etf_prices.csv`に保存する。
3. `preprocess_data`関数を実装し、`etf_prices.csv`を読み込み、日次リターンを計算し、`data/processed/etf_returns.csv`に保存する。
4. `create_rolling_windows`関数を実装する。この関数はリターンデータを入力とし、ルックバック期間（60日）のデータ `X` と、次の1日のリターン `y` のペアを生成する。`X`の形状は `(num_samples, 60, num_assets)`、`y`の形状は `(num_samples, num_assets)` となる。
5. `src/cli.py`に `download-data` と `preprocess-data` のサブコマンドを追加する。

**期待される出力ファイル**:
- src/data_pipeline.py
- src/cli.py
- data/processed/etf_returns.csv

**受入基準 (これを全て満たすまで完了としない)**:
- `python -m src.cli download-data`が成功し、`data/raw/etf_prices.csv`が生成される。
- `python -m src.cli preprocess-data`が成功し、`data/processed/etf_returns.csv`が生成される。
- 生成されたデータにNaNが含まれていないことを確認する。




## データ問題でスタックした場合の脱出ルール

レビューで3サイクル連続「データ関連の問題」が指摘されている場合:
1. **データの完全性を追求しすぎない** — 利用可能なデータでモデル実装に進む
2. **合成データでのプロトタイプを許可** — 実データが不足する部分は合成データで代替し、モデルの基本動作を確認
3. **データの制約を open_questions.md に記録して先に進む**
4. 目標は「論文の手法が動くこと」であり、「論文と同じデータを揃えること」ではない







## 全体Phase計画 (参考)

✓ Phase 1: コアモデルとSharpe損失関数の実装 — PyTorchでポートフォリオ最適化モデルと微分可能なSharpe比損失関数を実装し、合成データで動作確認する。
→ Phase 2: データパイプライン構築 — yfinanceを使用してETFデータを取得し、モデルの入力形式に前処理するパイプラインを構築する。
  Phase 3: 基本学習・評価ループの実装 — 単一の時系列分割（80/20）でモデルを学習させ、テストセットでのパフォーマンスを評価する。
  Phase 4: ウォークフォワード検証フレームワークの実装 — 厳密なバックテストのためにウォークフォワード検証を実装し、1/Nポートフォリオと比較する。
  Phase 5: 取引コストモデルの導入 — 取引コストを考慮したネットパフォーマンスを計算し、コストがパフォーマンスに与える影響を評価する。
  Phase 6: ハイパーパラメータ最適化 — Optunaを使い、最初の訓練期間でモデルの主要なハイパーパラメータを最適化する。
  Phase 7: 最適化済みモデルでのロバスト性検証 — 最適化されたハイパーパラメータを使用して、より多くの分割数でウォークフォワード検証を行い、モデルの安定性を評価する。
  Phase 8: 特徴量エンジニアリングの探求 — テクニカル指標などの追加特徴量を導入し、モデルのパフォーマンスが向上するかを検証する。
  Phase 9: ポートフォリオウェイトの挙動分析 — モデルが生成したポートフォリオウェイトを可視化・分析し、その投資行動を理解する。
  Phase 10: 代替モデルアーキテクチャの比較 — LSTMの代わりにシンプルなCNNベースのモデルを実装し、パフォーマンスを比較評価する。
  Phase 11: 最終テクニカルレポート生成 — 全フェーズの結果を統合し、再現・改善・探求の成果をまとめた包括的な技術レポートを生成する。
  Phase 12: エグゼクティブサマリーとコード整備 — 非技術者向けの要約を作成し、テストカバレッジを向上させ、プロジェクトを完成させる。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 再現モードのルール（論文忠実度の維持）

このプロジェクトは**論文再現**が目的。パフォーマンス改善より論文忠実度を優先すること。

### パラメータ探索の制約
- **論文で既定されたパラメータをまず実装し、そのまま評価すること**
- パラメータ最適化を行う場合、**論文既定パラメータの近傍のみ**を探索（例: 論文が12ヶ月なら [6, 9, 12, 15, 18] ヶ月）
- 論文と大きく異なるパラメータ（例: 月次論文に対して日次10営業日）で良い結果が出ても、それは「論文再現」ではなく「独自探索」
- 独自探索で得た結果は `customMetrics` に `label: "implementation-improvement"` として記録し、論文再現結果と明確に分離

### データ条件の忠実度
- 論文のデータ頻度（日次/月次/tick）にできるだけ合わせる
- ユニバース規模が論文より大幅に小さい場合、その制約を `docs/open_questions.md` に明記
- リバランス頻度・加重方法も論文に合わせる



## 禁止事項

### データ・特徴量の禁止パターン（具体的）
- `scaler.fit(full_data)` してから split → **禁止**。`scaler.fit(train_data)` のみ
- `df.rolling(window=N, center=True)` → **禁止**。`center=False` (デフォルト) を使用
- データの `end_date` が今日以降 → **禁止**。`end_date` を明示的に過去に設定
- `merge` で未来のタイムスタンプを持つ行が特徴量に混入 → **禁止**
- ラベル生成後に特徴量を合わせる（ラベルの存在を前提に特徴量を選択）→ **禁止**

### 評価・報告の禁止パターン
- コストなしのgross PnLだけで判断しない
- テストセットでハイパーパラメータを調整しない
- 時系列データにランダムなtrain/test splitを使わない
- README に metrics.json と異なる数値を手書きしない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**
- **論文が既定するパラメータから大幅に逸脱した探索を「再現」として報告しない**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/preflight.md` — Preflight チェック結果（必須、実装前に作成）
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ（Single Source of Truth）
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。

### レポート生成ルール（重要: 数値の一貫性）
- **`metrics.json` が全ての数値の唯一のソース (Single Source of Truth)**
- README や technical_findings に書く数値は **必ず metrics.json から引用** すること
- **手打ちの数値は禁止**。metrics.json に含まれない数値を README に書かない
- technical_findings.md で数値に言及する場合も metrics.json の値を参照
- README.md の Results セクションは metrics.json を読み込んで生成すること

### テスト必須
- `tests/test_data_integrity.py` のテストを実装状況に応じて有効化すること
- 新しいデータ処理や特徴量生成を追加したら、対応する leakage テストも追加
- `pytest tests/` が全パスしない場合、サイクルを完了としない

### その他の出力
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
