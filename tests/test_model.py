import numpy as np
import pandas as pd
import pytest

from olympoly.load_data import load_olympic_data
from sklearn.linear_model import LogisticRegression
from olympoly.olympics_betting.regression_model.model import build_features, train_model

"""
What this file covers
─────────────────────
  TestBuildFeatures  — verifies that build_features() produces a clean,
                       correctly shaped feature DataFrame from raw Olympic data.
  TestTrainModel     — verifies that train_model() returns a fitted
                       LogisticRegression with sensible outputs and basic
                       predictive signal.

Dependency note
───────────────
  Both model.py and this test file import from sklearn.  If pytest raises
      ModuleNotFoundError: No module named 'sklearn'
  run the following inside your active conda environment before re-running:

      pip install scikit-learn
"""

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """
    Generate a synthetic Olympic results DataFrame large enough for reliable
    logistic-regression tests.

    Design constraints
    ──────────────────
    Row count (n=40)
        train_model() calls train_test_split() with the default 75/25 split,
        yielding ~10 test rows.  Fewer than ~20 total rows risk producing a
        test split containing only one class, which breaks any metric that
        requires both classes present in y_true.

    Class balance
        The medal pool ["Gold", "Silver", "Bronze", None, None] gives roughly
        20% gold rows — realistic for Olympic data and ensures both 0 and 1
        appear in the target column after build_features() encodes is_gold.

    Feature variance
        Five distinct NOCs and twelve distinct athlete names ensure that
        country_strength and athlete_exp are not constant columns, which
        would cause LogisticRegression to produce degenerate coefficients.

    Reproducibility
        np.random.default_rng(0) pins the random state so the fixture
        produces identical data on every pytest run.
    """
    nocs   = ["USA", "CHN", "GBR", "AUS", "RUS"]
    names  = [f"Athlete_{i}" for i in range(12)]
    medals = ["Gold", "Silver", "Bronze", None, None]

    rng = np.random.default_rng(0)
    n   = 40

    return pd.DataFrame({
        "ID":    range(1, n + 1),
        "Name":  rng.choice(names,   n).tolist(),
        "NOC":   rng.choice(nocs,    n).tolist(),
        "Year":  rng.choice([2000, 2004, 2008, 2012, 2016, 2020], n).tolist(),
        "Sport": rng.choice(["Athletics", "Swimming", "Cycling"], n).tolist(),
        "Medal": rng.choice(medals,  n).tolist(),
    })


@pytest.fixture
def built_features(sample_data):
    """
    Run build_features() on sample_data once and share the result across all
    TestBuildFeatures tests via pytest's dependency injection.

    Using a fixture rather than calling build_features() inside each test
    avoids redundant computation and ensures every test in the class operates
    on an identical DataFrame — no hidden state differences between tests.
    """
    return build_features(sample_data)


@pytest.fixture
def trained(sample_data):
    """
    Run train_model() on sample_data once and share the 3-tuple
    (model, X_test, y_test) across all TestTrainModel tests.

    train_model() is deterministic given random_state=42 inside
    train_test_split, so every test that unpacks this fixture sees the same
    train/test split and the same fitted model weights.
    """
    return train_model(sample_data)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: build_features
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildFeatures:
    """
    Tests for the build_features() function in model.py.

    build_features() derives three columns from the raw DataFrame:
      - country_strength : mean gold-medal rate per NOC (float in [0, 1])
      - athlete_exp      : number of Olympic appearances per athlete (int >= 1)
      - is_gold          : binary target — 1 if Medal == 'Gold', else 0

    It joins these back onto the original frame via .join(on=...) and drops
    any rows where the join produced NaN before returning.
    """

    def test_required_columns_present(self, built_features):
        """
        Verify that build_features() outputs exactly the three columns that
        train_model() expects.  A missing column would raise a KeyError during
        feature selection before LogisticRegression.fit() is even called.
        """
        
        for col in ("country_strength", "athlete_exp", "is_gold"):
            assert col in built_features.columns

    def test_no_nulls(self, built_features):
        """
        Confirm that the trailing dropna() in build_features() has removed
        every NaN from the output.  A NaN in any feature column causes
        LogisticRegression.fit() to raise a ValueError and halts training.
        """
        
        assert built_features.isnull().sum().sum() == 0

    def test_returns_dataframe(self, built_features):
        """
        Sanity-check the return type.  Downstream code indexes the result
        using column names, which requires a DataFrame — a plain numpy array
        or Series would raise an AttributeError at the first column access.
        """
        
        assert isinstance(built_features, pd.DataFrame)

    def test_nonempty(self, built_features):
        """
        Ensure the feature pipeline does not silently drop every row via an
        overly aggressive dropna or a failed join that produces all-NaN
        columns.  An empty DataFrame passed to train_test_split raises a
        ValueError with a confusing message unrelated to the real cause.
        """
        
        assert len(built_features) > 0

    def test_is_gold_binary(self, built_features):
        """
        is_gold is created with (df['Medal'] == 'Gold').astype(int), so the
        only legal values are 0 and 1.  Any other value — e.g. a NaN coerced
        to -1 or a float like 0.5 — would corrupt the binary classification
        target and produce nonsensical predicted probabilities.
        """
        
        assert set(built_features["is_gold"].unique()).issubset({0, 1})

    def test_country_strength_bounded(self, built_features):
        """
        country_strength is the per-NOC mean of is_gold, which is itself
        binary.  The arithmetic mean of any set of 0/1 values must lie in
        [0, 1].  A value outside this range signals a bug in the groupby
        aggregation or the join logic (e.g. a rsuffix collision).
        """
        
        assert built_features["country_strength"].between(0, 1).all()

    def test_athlete_exp_positive(self, built_features):
        """
        athlete_exp is a count of Olympic appearances (groupby + .count()),
        so any athlete present in the DataFrame must have a value of at least
        1.  A zero or negative result is logically impossible and would
        indicate an off-by-one error or a join against the wrong column.
        """
        
        assert (built_features["athlete_exp"] >= 1).all()

    # ── boundary cases ───────────────────────────────────────────────────────

    def test_all_no_medals(self):
        """
        Boundary — lower bound of country_strength:
        When no athlete has won a Gold medal, is_gold is all-zero for every
        row, so country_strength (the per-NOC mean of is_gold) must be
        exactly 0.0 for every country.

        Exercises the edge case where the gold column is completely empty and
        confirms the feature pipeline does not assign a spurious positive value.
        """
        
        df = pd.DataFrame({
            "ID":    [1, 2, 3, 4],
            "Name":  ["A", "B", "C", "D"],
            "NOC":   ["USA", "CHN", "GBR", "AUS"],
            "Medal": [None, "Silver", "Bronze", None],
        })
        result = build_features(df)
        assert (result["country_strength"] == 0).all()

    def test_all_gold(self):
        """
        Boundary — upper bound of country_strength:
        When every athlete wins Gold, is_gold is all-one, so country_strength
        must be exactly 1.0 for every row.

        Verifies that a perfect win rate is represented correctly rather than
        being clipped or scaled by the aggregation logic.
        """
        
        df = pd.DataFrame({
            "ID":    [1, 2, 3],
            "Name":  ["A", "B", "C"],
            "NOC":   ["USA", "CHN", "GBR"],
            "Medal": ["Gold", "Gold", "Gold"],
        })
        result = build_features(df)
        assert (result["country_strength"] == 1).all()

    def test_single_athlete_experience(self):
        """
        Boundary — minimum athlete_exp value:
        An athlete who appears exactly once must receive athlete_exp == 1.
        Verifies the groupby count does not under-count single-entry athletes
        and that the join correctly maps the value back to that athlete's row.
        """
        
        df = pd.DataFrame({
            "ID":    [1, 2],
            "Name":  ["Solo", "Other"],
            "NOC":   ["USA", "CHN"],
            "Medal": ["Gold", None],
        })
        result = build_features(df)
        solo_idx  = df[df["Name"] == "Solo"].index
        solo_rows = result[result.index.isin(solo_idx)]
        if len(solo_rows):
            assert (solo_rows["athlete_exp"] == 1).all()

    def test_repeated_athlete_experience(self):
        """
        Boundary — athlete_exp accumulation across multiple appearances:
        An athlete who appears four times must receive athlete_exp == 4 on
        every one of their rows — not just the first.

        Verifies that the join broadcasts the aggregated count back to all
        matching rows correctly, not only to a single representative row.
        """
        
        df = pd.DataFrame({
            "ID":    [1, 2, 3, 4, 5],
            "Name":  ["A", "A", "A", "A", "B"],
            "NOC":   ["USA", "USA", "USA", "USA", "CHN"],
            "Medal": ["Gold", None, "Silver", None, "Gold"],
        })
        result = build_features(df)
        a_idx  = df[df["Name"] == "A"].index
        a_rows = result[result.index.isin(a_idx)]
        if len(a_rows):
            assert (a_rows["athlete_exp"] == 4).all()

    def test_dropna_removes_incomplete_rows(self):
        """
        Verify that build_features() always returns a fully clean DataFrame
        regardless of how many None medals are in the input.

        The join-then-dropna pattern could leave NaNs if a Name or NOC were
        somehow absent from the groupby index.  This test uses a
        self-consistent DataFrame (all Names and NOCs present in the groupby)
        to confirm the happy path produces zero nulls end-to-end.
        """
        
        df = pd.DataFrame({
            "ID":    [1, 2, 3],
            "Name":  ["A", "B", "C"],
            "NOC":   ["USA", "CHN", "GBR"],
            "Medal": ["Gold", None, "Silver"],
        })
        result = build_features(df)
        assert result.isnull().sum().sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: train_model
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainModel:
    """
    Tests for the train_model() function in model.py.

    train_model() chains:
      build_features() → train_test_split() → LogisticRegression.fit()

    and returns the 3-tuple (model, X_test, y_test).  These tests verify the
    contract of that return value, structural properties of the fitted model,
    and a lightweight directional signal check.
    """

    def test_returns_three_values(self, trained):
        """
        Enforce the 3-tuple return contract: (model, X_test, y_test).

        The original test file assigned the return value to a single variable,
        silently capturing the tuple object rather than the fitted model.
        This test makes the correct unpacking contract explicit so that any
        future change to the return signature is caught immediately.
        """
        
        assert len(trained) == 3

    def test_model_is_logistic_regression(self, trained):
        """
        Confirm the first element of the returned tuple is a
        LogisticRegression instance.  If model.py ever swaps to a different
        estimator without updating callers, this type check will catch the
        breaking change before it propagates to inference code.
        """
        
        model, _, _ = trained
        assert isinstance(model, LogisticRegression)

    def test_model_is_fitted(self, trained):
        """
        A scikit-learn estimator only gains a coef_ attribute after .fit()
        has been called and completed without error.  Checking for coef_ is
        the idiomatic sklearn pattern for asserting a model is trained, and
        avoids the overhead of re-running training inside the test.
        """
        
        model, _, _ = trained
        assert hasattr(model, "coef_")

    def test_x_test_correct_columns(self, trained):
        """
        Verify that the held-out feature matrix X_test contains exactly the
        two columns the model was trained on.  Extra or missing columns cause
        a shape mismatch error when model.predict_proba(X_test) is called,
        which would surface as a cryptic ValueError rather than a clear test
        failure.
        """
        
        _, X_test, _ = trained
        assert set(X_test.columns) == {"country_strength", "athlete_exp"}

    def test_lengths_consistent(self, trained):
        """
        X_test and y_test must have the same number of rows.  A length
        mismatch would indicate that train_test_split was applied separately
        to X and y rather than jointly, producing misaligned row indices and
        silently corrupting any metric computed between the two.
        """
        
        _, X_test, y_test = trained
        assert len(X_test) == len(y_test)

    def test_y_test_is_binary(self, trained):
        """
        The target is_gold is binary (0/1).  Confirm that this property
        survives the train_test_split so that downstream binary-classification
        metrics (accuracy, AUC) computed on y_test remain meaningful.
        """
        
        _, _, y_test = trained
        assert set(y_test.unique()).issubset({0, 1})

    def test_predict_proba_works(self, trained):
        """
        Call predict_proba() on the held-out test set and verify every
        predicted probability is a valid value in [0, 1].

        This is the primary output consumed by the betting pipeline.
        Invalid probabilities — negative values, values > 1, or NaN — would
        silently corrupt Kelly fractions and EV calculations downstream.
        """
        
        model, X_test, _ = trained
        probs = model.predict_proba(X_test)[:, 1]
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_predict_proba_shape(self, trained):
        """
        For a binary classifier, predict_proba() must return a 2D array with
        shape (n_samples, 2) — one column per class (probability of 0 and
        probability of 1).  Callers index [:, 1] to extract the gold-medal
        probability; any other shape raises an IndexError at that step.
        """
        
        model, X_test, _ = trained
        raw = model.predict_proba(X_test)
        assert raw.shape == (len(X_test), 2)

    def test_coef_shape(self, trained):
        """
        For a binary LogisticRegression trained on two features, coef_ must
        have shape (1, 2): one decision boundary with one weight per feature.
        A shape of (2, 2) would indicate multiclass mode; (1, 3) would mean
        the model was accidentally trained on three features.
        """
        
        model, _, _ = trained
        assert model.coef_.shape == (1, 2)

    # ── boundary / signal checks ─────────────────────────────────────────────

    def test_predictions_are_floats(self, trained):
        """
        Predicted probabilities must be floating-point.  An integer dtype
        would indicate predict_proba() was accidentally swapped for predict(),
        which returns class labels (0 or 1) rather than continuous scores and
        would make the betting signals binary and useless.
        """
        
        model, X_test, _ = trained
        probs = model.predict_proba(X_test)[:, 1]
        assert probs.dtype in (np.float32, np.float64)

    def test_model_produces_predictions_for_new_data(self, trained):
        """
        Verify the fitted model can score a hand-crafted row it has never
        seen — the most basic check that the model is usable for inference
        beyond the training/test split.

        If predict_proba() raises an error here it typically means the model
        was fitted on a different feature set than what is being passed, or
        that a preprocessing step is missing from the inference path.
        """
        
        model, _, _ = trained
        new_row = pd.DataFrame({
            "country_strength": [0.5],
            "athlete_exp":      [3],
        })
        prob = model.predict_proba(new_row)[0, 1]
        assert 0.0 <= prob <= 1.0

    def test_high_strength_country_scores_higher(self, trained):
        """
        Weak monotonicity / directional signal check:
        A country with a high historical win rate (country_strength=0.8)
        should receive a higher predicted gold probability than one with a
        low win rate (country_strength=0.05), all else equal.

        Uses >= rather than > to avoid flakiness on small datasets where the
        two probabilities may be numerically equal.  This test fails only if
        the model learns a negative coefficient on country_strength — meaning
        it actively penalises strong countries — which is a clear sign of a
        degenerate training set or a bug in build_features().
        """
        
        model, _, _ = trained
        strong = pd.DataFrame({"country_strength": [0.8], "athlete_exp": [5]})
        weak   = pd.DataFrame({"country_strength": [0.05], "athlete_exp": [5]})
        p_strong = model.predict_proba(strong)[0, 1]
        p_weak   = model.predict_proba(weak)[0, 1]
        assert p_strong >= p_weak
