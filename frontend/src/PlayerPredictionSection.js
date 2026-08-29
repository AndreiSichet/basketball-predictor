/**
 * One selected player's five prop predictions.
 *
 * Display only - it receives an already-fetched player line and renders it.
 *
 * modelUsed IS SHOWN, once, beside the player's name rather than repeated
 * against each of the five stats. The API returns exactly one value per
 * player, because the hybrid's routing decision is made from that player's
 * rolling-window completeness and therefore applies to his whole line;
 * printing the same word five times would suggest it could differ per stat,
 * which it cannot. Shown rather than hidden either way: which half of the
 * hybrid answered is a real property of the number, the same transparency
 * standard the API itself already enforces by returning the field at all.
 */
function PlayerPredictionSection({ player }) {
  return (
    <section className="player-section" aria-label={`${player.playerName} predictions`}>
      <h3>
        {player.playerName}
        <span className={`model-tag model-${player.modelUsed}`}>
          {player.modelUsed} model
        </span>
      </h3>

      <dl className="prediction-grid">
        <Stat label="Points" value={player.predictedPoints} />
        <Stat label="Rebounds" value={player.predictedRebounds} />
        <Stat label="Assists" value={player.predictedAssists} />
        <Stat label="Threes made" value={player.predictedThreesMade} />
        <Stat label="Points + rebounds + assists" value={player.predictedPra} />
      </dl>
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="metric-row">
      <dt>{label}</dt>
      <dd>{value.toFixed(1)}</dd>
    </div>
  );
}

export default PlayerPredictionSection;
