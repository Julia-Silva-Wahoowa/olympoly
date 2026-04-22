# Olympoly CLI Guide

Command-line interface to analyze historical Olympic data, calculate medal probabilities, and compare inefficiencies against live prediction market odds (Polymarket).

## Usage

`python main.py <command> [options]`

## Core Commands

### 1. Data Exploration & Basics

- `python main.py explore`: Load the dataset and print structural information.
- `python main.py demographics`: Fetch demographic data using the Census API.

### 2. Performance & Trends

- `python main.py performance [--country <name>]`: Plot country efficiency (medals per athlete). If `--country` is provided, plots that country's trend over time.
- `python main.py timeline [--by_gender] [--entity <Team/Sport>]`: Plot participation and medal trends over time.

### 3. Historical Probability Calculations

- `python main.py hist-prob country-medal-prob --country <name>`: Get baseline medal probability.
- `python main.py hist-prob country-gold-prob --country <name>`: Get baseline gold medal probability.
- `python main.py hist-prob sport-medal-prob --sport <name>`: Get medal probability for a given sport.
- `python main.py hist-prob rolling-performance --country <name> [--window <int>]`: Get rolling performance averages.

### 4. Advanced Probability & Smoothing

- `python main.py prob medal-prob [--group_cols ...] [--medal_type <type>]`: Generate baseline grouped probabilities.
- `python main.py prob weighted-medal-prob [--decay <float>]`: Calculate recency-weighted probabilities.
- `python main.py prob country-event-prob`: Compute highly smoothed, recency-weighted posterior probabilities per country-event using Empirical Bayes.

### 5. Monte Carlo Simulation

- `python main.py simulate`: Run a Monte Carlo simulation for overall expected medals based on historical data.

### 6. Live Prediction Market Data

- `python main.py market --event "<query>"`: Fetch current odds for a specific event from Polymarket.
- `python main.py load-market --filepath <path.csv>`: Load and validate market data from a local CSV.

### 7. Regression Models

- `python main.py regression [--model <logistic/rf>] [--action <train/results/feature-importance/sample-tree>]`: Train machine learning models using historical country strength and athlete experience.

### 8. Comparing Market vs. Model

- `python main.py compare --event "<query>"`: Scrape Polymarket for an event, run the local regression model, and output the absolute edge/disagreement between the two.

### 9. Betting Simulations

- `python main.py bet-sim market-strategy --market_data_path <csv> --model_data_path <csv> [--edge_threshold <float>]`: Simulate bankroll returns betting purely on edge threshold differences.
- `python main.py bet-sim edge-strategy --market_data_path <csv> --model_data_path <csv>`: Check the raw directional accuracy of the model vs the market.

### 10. Visualization Tools

- `python main.py viz-market price-over-time --market_data_path <csv> --event <name>`: View odds changes over time.
- `python main.py viz-market market-vs-model-plot --compare_data_path <csv>`: Bar chart comparing probabilities.
- `python main.py viz-market top-edges --compare_data_path <csv> [--n <int>]`: Horizontal bar chart of the largest mispricings available.
- `python main.py viz-market prob-distribution --compare_data_path <csv>`: Scatter plot mapping market probability against model probability.
