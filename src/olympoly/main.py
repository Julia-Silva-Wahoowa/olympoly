import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from olympoly import explore_data, performance, timeline, load_data
from olympoly.monte_carlo_simulation import main as mc_main
from olympoly.olympics_betting import (
    market_data, historical_model, probability, simulation, visualize_markets, market_vs_model
)
from olympoly.olympics_betting.regression_model import model as reg_model, random_forests


def main():
    parser = argparse.ArgumentParser(
        description="Olympics Analysis CLI - Explore, analyze, simulate, and visualize Olympic data."
    )
    subparsers = parser.add_subparsers(dest="command", help="Feature to use")

    subparsers.add_parser(
        "explore", help="Show basic info about the Olympics dataset")

    perf_parser = subparsers.add_parser(
        "performance", help="Analyze country efficiency")
    perf_parser.add_argument("--country", type=str,
                             help="Country name for efficiency trends")
    perf_parser.add_argument(
        "--season", type=str, default="Summer", help="Filter by Season (Summer/Winter)")

    time_parser = subparsers.add_parser(
        "timeline", help="Analyze participation and medal trends")
    time_parser.add_argument(
        "--by_gender", action="store_true", help="Show participation by gender")
    time_parser.add_argument("--entity", type=str,
                             default="Team", help="Entity for medal trends")
    time_parser.add_argument("--season", type=str,
                             default="Summer", help="Filter by Season")

    subparsers.add_parser(
        "simulate", help="Run Monte Carlo simulation for medal predictions")

    market_parser = subparsers.add_parser(
        "market", help="Show latest prediction market prices")
    market_parser.add_argument(
        "--event", type=str, required=True, help="Event name to fetch market data for")

    compare_parser = subparsers.add_parser(
        "compare", help="Compare model and market probabilities")
    compare_parser.add_argument(
        "--event", type=str, required=True, help="Event name to compare")

    reg_parser = subparsers.add_parser(
        "regression", help="Train, evaluate, and visualize regression models")
    reg_parser.add_argument(
        "--model", choices=["logistic", "rf"], default="logistic", help="Model type")
    reg_parser.add_argument("--action", choices=["train", "results", "feature-importance",
                            "sample-tree"], default="train", help="Action to perform")
    reg_parser.add_argument("--results_path", type=str,
                            help="Path to save/load model results (CSV)")

    subparsers.add_parser(
        "demographics", help="Load demographic data using Census API")

    hist_prob_parser = subparsers.add_parser(
        "hist-prob", help="Analyze historical medal probabilities")
    hist_prob_subparsers = hist_prob_parser.add_subparsers(
        dest="hist_prob_command", help="Historical probability command")

    medal_prob_parser = hist_prob_subparsers.add_parser(
        "country-medal-prob", help="Country medal probability")
    medal_prob_parser.add_argument(
        "--country", type=str, required=True, help="Country name")

    gold_prob_parser = hist_prob_subparsers.add_parser(
        "country-gold-prob", help="Country gold medal probability")
    gold_prob_parser.add_argument(
        "--country", type=str, required=True, help="Country name")

    sport_medal_prob_parser = hist_prob_subparsers.add_parser(
        "sport-medal-prob", help="Sport medal probability")
    sport_medal_prob_parser.add_argument(
        "--sport", type=str, required=True, help="Sport name")

    rolling_perf_parser = hist_prob_subparsers.add_parser(
        "rolling-performance", help="Rolling country performance trend")
    rolling_perf_parser.add_argument(
        "--country", type=str, required=True, help="Country name")
    rolling_perf_parser.add_argument(
        "--window", type=int, default=2, help="Rolling window size")

    load_market_parser = subparsers.add_parser(
        "load-market", help="Load market data from a local CSV file")
    load_market_parser.add_argument(
        "--filepath", type=str, required=True, help="Path to the market data CSV file")

    prob_parser = subparsers.add_parser(
        "prob", help="Estimate various probabilities from Olympic data")
    prob_subparsers = prob_parser.add_subparsers(
        dest="prob_command", help="Probability command")

    medal_prob_cmd_parser = prob_subparsers.add_parser(
        "medal-prob", help="Medal probability")
    medal_prob_cmd_parser.add_argument(
        "--group_cols", nargs='+', default=['NOC', 'Event'], help="Columns to group by")
    medal_prob_cmd_parser.add_argument(
        "--medal_type", type=str, default="Gold", help="Medal type")

    prob_subparsers.add_parser(
        "athlete-medal-prob", help="Athlete medal probability")

    weighted_medal_prob_parser = prob_subparsers.add_parser(
        "weighted-medal-prob", help="Weighted medal probability")
    weighted_medal_prob_parser.add_argument(
        "--group_cols", nargs='+', default=['NOC', 'Event'], help="Columns to group by")
    weighted_medal_prob_parser.add_argument(
        "--current_year", type=int, default=2026, help="Current year for decay calculation")
    weighted_medal_prob_parser.add_argument(
        "--decay", type=float, default=0.9, help="Decay rate")

    country_event_prob_parser = prob_subparsers.add_parser(
        "country-event-prob", help="Country-event smoothed probabilities")
    country_event_prob_parser.add_argument(
        "--min_effective_n", type=float, default=8.0)
    country_event_prob_parser.add_argument(
        "--min_year", type=int, default=2000)
    country_event_prob_parser.add_argument(
        "--require_recent_event", action="store_true")
    country_event_prob_parser.add_argument(
        "--recent_event_cutoff_year", type=int, default=2000)
    country_event_prob_parser.add_argument(
        "--keep_defunct_nocs", action="store_true")
    country_event_prob_parser.add_argument(
        "--prior_strength_k", type=float, default=8.0)
    country_event_prob_parser.add_argument(
        "--fallback_laplace", action="store_true")
    country_event_prob_parser.add_argument(
        "--laplace_alpha", type=float, default=1.0)
    country_event_prob_parser.add_argument(
        "--laplace_beta", type=float, default=1.0)
    country_event_prob_parser.add_argument(
        "--half_life_years", type=float, default=12.0)

    betting_sim_parser = subparsers.add_parser(
        "bet-sim", help="Run betting simulations")
    betting_sim_subparsers = betting_sim_parser.add_subparsers(
        dest="bet_sim_command", help="Betting simulation command")

    market_strategy_parser = betting_sim_subparsers.add_parser(
        "market-strategy", help="Simulate market betting strategy")
    market_strategy_parser.add_argument(
        "--market_data_path", type=str, required=True)
    market_strategy_parser.add_argument(
        "--model_data_path", type=str, required=True)
    market_strategy_parser.add_argument(
        "--model_col", type=str, default="model_prob")
    market_strategy_parser.add_argument(
        "--market_col", type=str, default="price")
    market_strategy_parser.add_argument(
        "--edge_threshold", type=float, default=0.05)
    market_strategy_parser.add_argument("--bet_size", type=float, default=1.0)
    market_strategy_parser.add_argument(
        "--bankroll", type=float, default=100.0)
    market_strategy_parser.add_argument("--seed", type=int, default=42)

    edge_strategy_parser = betting_sim_subparsers.add_parser(
        "edge-strategy", help="Simulate edge strategy")
    edge_strategy_parser.add_argument(
        "--market_data_path", type=str, required=True)
    edge_strategy_parser.add_argument(
        "--model_data_path", type=str, required=True)
    edge_strategy_parser.add_argument("--threshold", type=float, default=0.1)

    viz_market_parser = subparsers.add_parser(
        "viz-market", help="Visualize prediction market data")
    viz_market_subparsers = viz_market_parser.add_subparsers(
        dest="viz_market_command", help="Visualization command")

    plot_price_over_time_parser = viz_market_subparsers.add_parser(
        "price-over-time", help="Plot price over time")
    plot_price_over_time_parser.add_argument(
        "--market_data_path", type=str, required=True)
    plot_price_over_time_parser.add_argument(
        "--event", type=str, required=True)
    plot_price_over_time_parser.add_argument(
        "--use_normalized", action="store_true")

    plot_market_comparison_parser = viz_market_subparsers.add_parser(
        "market-comparison", help="Compare prices across markets")
    plot_market_comparison_parser.add_argument(
        "--market_data_path", type=str, required=True)
    plot_market_comparison_parser.add_argument(
        "--event", type=str, required=True)

    plot_latest_snapshot_parser = viz_market_subparsers.add_parser(
        "latest-snapshot", help="Bar chart of latest market probabilities")
    plot_latest_snapshot_parser.add_argument(
        "--market_data_path", type=str, required=True)
    plot_latest_snapshot_parser.add_argument(
        "--event", type=str, required=True)

    plot_market_vs_model_parser = viz_market_subparsers.add_parser(
        "market-vs-model-plot", help="Compare market vs model probabilities")
    plot_market_vs_model_parser.add_argument(
        "--compare_data_path", type=str, required=True)

    plot_top_edges_parser = viz_market_subparsers.add_parser(
        "top-edges", help="Plot top N biggest mispricings")
    plot_top_edges_parser.add_argument(
        "--compare_data_path", type=str, required=True)
    plot_top_edges_parser.add_argument("--n", type=int, default=10)

    plot_prob_dist_parser = viz_market_subparsers.add_parser(
        "prob-distribution", help="Scatter plot of market vs model probabilities")
    plot_prob_dist_parser.add_argument(
        "--compare_data_path", type=str, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_usage()
        sys.exit(1)

    if args.command == "explore":
        explore_data.explore()

    elif args.command == "performance":
        df = load_data.get_cleaned_data(season=args.season)
        if args.country:
            performance.efficiency_trends(df, args.country, plot=True)
        else:
            performance.country_efficiency(df, plot=True)

    elif args.command == "timeline":
        df = load_data.get_cleaned_data(season=args.season)
        timeline.participation_trends(df, by_gender=args.by_gender, plot=True)
        timeline.medal_trends(df, entity=args.entity, plot=True)
        timeline.sport_popularity(df, plot=True)

    elif args.command == "simulate":
        if "simulate" in sys.argv:
            sys.argv.remove("simulate")
        mc_main()

    elif args.command == "market":
        df = market_data.get_current_odds_for_event(args.event)
        print(df if not df.empty else "No market data found.")

    elif args.command == "compare":
        df_market = market_data.get_current_odds_for_event(args.event)
        df = load_data.get_cleaned_data()
        model, X_test, y_test = reg_model.train_model(df)
        df_model = X_test.copy()
        df_model["event"] = args.event
        df_model["model_prob"] = model.predict_proba(X_test)[:, 1]
        result = market_vs_model.compare_market_vs_model(df_market, df_model)
        print(result)

    elif args.command == "regression":
        df = load_data.get_cleaned_data()
        if args.model == "logistic":
            model, X_test, y_test = reg_model.train_model(df)
            print("Trained logistic regression. Test size:", len(X_test))
            if args.action == "results":
                preds = model.predict_proba(X_test)[:, 1]
                results_df = X_test.copy()
                results_df['pred_prob'] = preds
                results_df['actual'] = y_test.values
                print(results_df.sort_values(
                    'pred_prob', ascending=False).head(15))
                if args.results_path:
                    results_df.to_csv(args.results_path, index=False)
        else:
            model, X_test, y_test, probs = random_forests.train_rf_model(df)
            if args.action == "results":
                results = random_forests.get_results(X_test, y_test, probs)
                print(results.head(15))
            elif args.action == "feature-importance":
                random_forests.plot_feature_importance(model)
            elif args.action == "sample-tree":
                random_forests.plot_sample_tree(model)

    elif args.command == "demographics":
        df_demographics = load_data.load_demographic_data()
        print(
            df_demographics if not df_demographics.empty else "No demographic data found.")

    elif args.command == "hist-prob":
        df = load_data.get_cleaned_data()
        if args.hist_prob_command == "country-medal-prob":
            print(historical_model.country_medal_probability(df, args.country))
        elif args.hist_prob_command == "country-gold-prob":
            print(historical_model.country_gold_probability(df, args.country))
        elif args.hist_prob_command == "sport-medal-prob":
            print(historical_model.sport_medal_probability(df, args.sport))
        elif args.hist_prob_command == "rolling-performance":
            print(historical_model.rolling_country_performance(
                df, args.country, args.window))

    elif args.command == "load-market":
        print(market_data.load_market_data(args.filepath))

    elif args.command == "prob":
        df = load_data.get_cleaned_data()
        if args.prob_command == "medal-prob":
            print(probability.medal_probability(
                df, args.group_cols, args.medal_type))
        elif args.prob_command == "athlete-medal-prob":
            print(probability.athlete_medal_probability(df))
        elif args.prob_command == "weighted-medal-prob":
            print(probability.weighted_medal_probability(
                df, args.group_cols, args.current_year, args.decay))
        elif args.prob_command == "country-event-prob":
            print(probability.compute_country_event_probabilities(
                df, min_effective_n=args.min_effective_n, min_year=args.min_year,
                require_recent_event=args.require_recent_event, recent_event_cutoff_year=args.recent_event_cutoff_year,
                drop_defunct_nocs=not args.keep_defunct_nocs, prior_strength_k=args.prior_strength_k,
                fallback_laplace=args.fallback_laplace, laplace_alpha=args.laplace_alpha,
                laplace_beta=args.laplace_beta, half_life_years=args.half_life_years
            ))

    elif args.command == "bet-sim":
        df_market = pd.read_csv(args.market_data_path)
        df_model = pd.read_csv(args.model_data_path)
        merged_df = pd.merge(df_market, df_model, on='event',
                             how='inner', suffixes=('_market', '_model'))

        if args.bet_sim_command == "market-strategy":
            results_df, summary = simulation.simulate_market_strategy(
                merged_df, model_col=args.model_col, market_col=args.market_col,
                edge_threshold=args.edge_threshold, bet_size=args.bet_size, bankroll=args.bankroll, seed=args.seed
            )
            print(results_df)
            print(summary)
        elif args.bet_sim_command == "edge-strategy":
            print(simulation.simulate_edge_strategy(merged_df, args.threshold))

    elif args.command == "viz-market":
        if args.viz_market_command == "price-over-time":
            visualize_markets.plot_price_over_time(pd.read_csv(
                args.market_data_path), args.event, args.use_normalized)
        elif args.viz_market_command == "market-comparison":
            visualize_markets.plot_market_comparison(
                pd.read_csv(args.market_data_path), args.event)
        elif args.viz_market_command == "latest-snapshot":
            visualize_markets.plot_latest_snapshot(
                pd.read_csv(args.market_data_path), args.event)
        elif args.viz_market_command == "market-vs-model-plot":
            visualize_markets.plot_market_vs_model(
                pd.read_csv(args.compare_data_path))
        elif args.viz_market_command == "top-edges":
            visualize_markets.plot_top_edges(
                pd.read_csv(args.compare_data_path), args.n)
        elif args.viz_market_command == "prob-distribution":
            visualize_markets.plot_probability_distribution(
                pd.read_csv(args.compare_data_path))


if __name__ == "__main__":
    main()
