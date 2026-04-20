# 🏅 olympoly

**Predictive Modeling meets Olympic Prediction Markets.**

`olympoly` is an open-source data analysis tool designed to identify discrepancies between historical Olympic performance and real-time sentiment on decentralized prediction markets like **PolyMarket**.

---

## 📊 Project Overview

The core objective of `olympoly` is to determine if historical data can "out-predict" public sentiment. By leveraging over a century of Olympic datasets and modern machine learning baselines, the tool flags instances where the market's implied probability (the odds) deviates significantly from statistical reality.

### Key Features
- **Time Series Analysis:** Analyze trends regarding athlete participation, medal counts, and sport popularity.
- **Simulation:** Simulates Olympics sport betting.
- **Machine Learning:** Test different models on Olympic data.

---

## ⚙️ Installation

To set up the environment and explore the analysis, clone the repository and install the package:

```bash
git clone https://github.com/caleb-adhikari/olympoly.git
cd olympoly
python -m pip install -e ".[dev]"
```

## Requirements

`olympoly` requires:

`python` >= 3.13

`datasets` >= 4.8.4

`numpy` >= 2.0

`pandas` >=2.2

`seaborn` >=0.13

'scikit-learn' >= 1.8.0

Optional requirements:

`pytest`

`jupyter`

`matplotlib`

`ruff`
