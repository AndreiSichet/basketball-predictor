/**
 * The two market groups that describe the game as a whole: the seven
 * full-game models, and the six Q1 / first-half ones.
 *
 * Purely presentational. Both results are fetched together by BrowseView
 * before this view is ever reached, so there is nothing to load here and no
 * loading state to render.
 */
function GamePredictionsTab({ prediction, quarterHalf }) {
  return (
    <div>
      <section className="market-group">
        <h2>Full game</h2>

        {prediction ? (
          <>
            {prediction.stale && (
              <p className="freshness">
                <span className="stale-badge">STALE</span>
                Computed from data as of {prediction.dataAsOf}
              </p>
            )}

            <dl className="prediction-grid">
              <Metric
                label="Home win probability"
                value={`${(prediction.homeWinProbability * 100).toFixed(1)}%`}
              />
              <Metric label="Home margin" value={prediction.homeMargin.toFixed(1)} />
              <Metric label="Total points" value={prediction.totalPoints.toFixed(1)} />
              <Metric
                label="Rebound margin"
                value={prediction.reboundMargin.toFixed(1)}
              />
              <Metric
                label="Total rebounds"
                value={prediction.totalRebounds.toFixed(1)}
              />
              <Metric
                label="Assist margin"
                value={prediction.assistMargin.toFixed(1)}
              />
              <Metric
                label="Total assists"
                value={prediction.totalAssists.toFixed(1)}
              />
            </dl>
          </>
        ) : (
          <p className="status">No prediction yet</p>
        )}
      </section>

      <section className="market-group">
        <h2>First quarter and first half</h2>

        {quarterHalf ? (
          <dl className="prediction-grid">
            <PeriodWinnerMetric
              label="Q1 home win probability"
              probability={quarterHalf.q1WinnerProbability}
              confidence={quarterHalf.q1WinnerConfidence}
              interpretation={quarterHalf.q1WinnerInterpretation}
            />
            <Metric label="Q1 home margin" value={quarterHalf.q1Spread.toFixed(1)} />
            <Metric label="Q1 total points" value={quarterHalf.q1Total.toFixed(1)} />

            <PeriodWinnerMetric
              label="1H home win probability"
              probability={quarterHalf.half1WinnerProbability}
              confidence={quarterHalf.half1WinnerConfidence}
              interpretation={quarterHalf.half1WinnerInterpretation}
            />
            <Metric
              label="1H home margin"
              value={quarterHalf.half1Spread.toFixed(1)}
            />
            <Metric
              label="1H total points"
              value={quarterHalf.half1Total.toFixed(1)}
            />
          </dl>
        ) : (
          <p className="status">No quarter or half prediction yet</p>
        )}
      </section>
    </div>
  );
}

/**
 * A winner probability, shown with both of its qualifiers.
 *
 * NEITHER IS DECORATION. The confidence tag exists because q1_winner scores
 * 0.5796 against a 0.5184 always-home baseline - genuinely better than
 * chance, but close enough to it that rendering it identically to the
 * others would misrepresent it. The interpretation string exists because
 * these two models were trained only on periods that had a winner, so the
 * number is P(home leads | not tied) and is not the same quantity as the
 * full-game win probability sitting a few rows above it.
 *
 * Hiding either one would leave a plausible-looking percentage with no way
 * to know how much to trust it - the same reason the stale badge is a badge
 * and not a console warning.
 */
function PeriodWinnerMetric({ label, probability, confidence, interpretation }) {
  return (
    <div className="metric-row">
      <dt>
        {label}
        {confidence && (
          <span className={`confidence-tag confidence-${confidence}`}>
            {confidence} confidence
          </span>
        )}
      </dt>
      <dd>
        {`${(probability * 100).toFixed(1)}%`}
        {interpretation && (
          <span className="interpretation">{interpretation}</span>
        )}
      </dd>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric-row">
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

export default GamePredictionsTab;
