# ProfitScout: High Gamma Trading System

ProfitScout is an autonomous machine learning system designed to identify stocks poised for **explosive, short-term price moves** (High Gamma). Unlike traditional "trend following" systems, it ignores slow movers and specifically targets volatility velocity.

**Target:** `(Next_Day_Close - Close) > 0.5 * ATR(14)`  
**Philosophy:** "Buy Volatility, Sell Velocity."

---

## 🎯 The "Sniper" Methodology

The core innovation of ProfitScout is its **Dynamic Thresholding** system. Most models choke on market noise because they force a prediction for every stock, every day. ProfitScout operates differently:

1.  **Strict Filtering:** The model is trained to recognize the "perfect setup."
2.  **The "Top 1%" Rule:** During training, we calculate the probability threshold required to be in the **Top 100** highest-confidence historical trades.
    *   *Current Threshold:* **0.814** (81.4% Confidence)
3.  **Actionable Precision:**
    *   **Baseline Precision (Random Guess):** ~12%
    *   **ProfitScout Precision (at Threshold):** **~36%**
    *   *Implication:* When the model pulls the trigger, the odds of a massive (>0.5 ATR) move trip by **3x**.

### The "Daily Picks"
Every evening, the system scans the entire market (~1,500 tickers) and delivers a ranked list of the **Top 20** setups (10 Calls, 10 Puts).
*   **Tier 1 (Sniper):** Probability > 0.814. These are the "Fat Pitches."
*   **Tier 2 (Watchlist):** Top ranked but < 0.814. Good setups, but require manual confirmation.

---

## 🧠 The Brain: XGBoost & Features

The model does not care about "company fundamentals." It only cares about **Price Structure** and **Velocity**.

### Key Drivers (Feature Importance)
1.  **Close Location Value (CLV):** Where did the price close relative to the day's High/Low?
    *   *Signal:* A close near the High suggests institutional buying into the close.
2.  **Distance from Low:** How far has it already moved?
3.  **Momentum:** ROC (Rate of Change) 1, 3, 5 days.
4.  **Volatility Compression:** Bollinger Band Width & ATR Ratio.

*Note: The model has "learned" that a strong close (CLV > 0.8) combined with rising short-term momentum (ROC) is the single best predictor of a gap-up or continuation.*

---

## ⚙️ Automated Workflow

The system runs autonomously on Google Cloud Vertex AI.

### 1. Daily Inference (Mon-Fri @ 5:00 PM EST)
*   **Trigger:** Market Close.
*   **Process:**
    1.  Loads fresh OHLCV data from BigQuery.
    2.  Engineers features for the *latest* trading day.
    3.  Applies the "Golden" model and thresholds.
    4.  Saves the **Top 20** picks to BigQuery: `profit_scout.daily_predictions`.

### 2. Weekly Retraining (Sun @ 9:00 AM EST)
*   **Trigger:** Weekly Reset.
*   **Process:**
    1.  Retrains the model on the absolute latest data (capturing new market regimes).
    2.  Recalculates the "Sniper Threshold" based on recent performance.
    3.  **Promotes** the new model to the production environment automatically.

---

## 📊 Results & Performance

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Target** | `> 0.5 * ATR` | Predicting massive, tradable volatility. |
| **PR-AUC** | `0.22` | Area Under the Precision-Recall Curve (Global). |
| **Precision @ 100** | **0.36** | **Win rate for the Top 100 predictions.** |
| **Risk/Reward** | **High** | Winners typically pay 3:1 or 4:1 due to the magnitude of the move. |

*Status: Fully Operational. Inference and Training schedules are active.*