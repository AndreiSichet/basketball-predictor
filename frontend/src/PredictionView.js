/**
 * Detail view: every model's output for one matchup.
 *
 * Takes the GameSummaryDto the POST already returned rather than fetching
 * anything - the prediction was made to get here.
 */
function PredictionView({ result, onBack }) {
  const prediction = result.latestPrediction;

  return (
    <div>
      <button className="back-button" onClick={onBack}>
        ← Back
      </button>

      <header className="intro">
        <h1>
          {result.awayTeamAbbreviation} @ {result.homeTeamAbbreviation}
        </h1>
        <p className="game-date">{result.gameDate}</p>
      </header>

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

export default PredictionView;
