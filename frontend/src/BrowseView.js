import { useState } from 'react';
import {
  createPrediction,
  createQuarterHalfPrediction,
  getHealth,
  getSchedule,
} from './api';
import { latestPredictableDate } from './dates';

const SCHEDULE_DAYS_AHEAD = 14;

/** Stable identity for a fixture, which has no id until it is predicted. */
function fixtureKey(game) {
  return `${game.homeTeamId}-${game.awayTeamId}-${game.gameDate}`;
}

/**
 * Landing view: what the app is, and the real NBA schedule to pick from.
 *
 * The schedule happily returns fixtures weeks out, but only games within
 * MAX_DAYS_AHEAD of the dataset can actually be predicted. Rather than let
 * someone click into a wall of identical 400s, unreachable fixtures keep a
 * visible but disabled button with the reason on it - the same instinct as
 * the stale badge, which says so rather than hiding.
 */
function BrowseView({ games, onGamesLoaded, onSelect }) {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [fetched, setFetched] = useState(games.length > 0);
  const [pendingKey, setPendingKey] = useState(null);
  const [predictError, setPredictError] = useState(null);

  const cutoff = latestPredictableDate(health?.dataAsOf);

  async function handleFetch() {
    setLoading(true);
    setError(null);
    setPredictError(null);

    try {
      // Freshness comes back alongside the schedule because one is
      // useless without the other: a fixture list with no cutoff cannot
      // say which of its entries are reachable.
      const [schedule, freshness] = await Promise.all([
        getSchedule(SCHEDULE_DAYS_AHEAD),
        getHealth(),
      ]);
      setHealth(freshness);
      onGamesLoaded(schedule);
      setFetched(true);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleShowPredictions(game) {
    setPendingKey(fixtureKey(game));
    setPredictError(null);

    const payload = {
      homeTeamId: game.homeTeamId,
      awayTeamId: game.awayTeamId,
      gameDate: game.gameDate,
    };

    try {
      // Both together, not staggered. They are the two halves of the
      // default tab, they are both small, and the backend resolves them to
      // the same Game row - so a sequential pair would only mean the
      // detail view painting with half its content missing. Player props
      // are deliberately NOT here; they are heavier and are fetched only
      // if that tab is opened.
      const [gameResult, quarterHalfResult] = await Promise.all([
        createPrediction(payload),
        createQuarterHalfPrediction(payload),
      ]);

      // The request travels with the result so the detail view can fetch
      // player props later without reconstructing team ids from
      // abbreviations.
      onSelect({ game: gameResult, quarterHalf: quarterHalfResult, request: payload });
    } catch (failure) {
      // The disabled state should make this unreachable. If it is ever
      // reached anyway, the backend's real reason is what shows.
      setPredictError(failure.message);
    } finally {
      setPendingKey(null);
    }
  }

  return (
    <div>
      <header className="intro">
        <h1>Basketball Predictor</h1>
        <p>
          Seven self-trained models over eleven seasons of NBA game data,
          predicting the winner, the margin, and the totals for points,
          rebounds and assists.
        </p>
        <p className="intro-note">
          Predictions are built from each team's form, rest and Elo going
          into a game, so only the next game after the latest results can be
          scored — fixtures further out are listed but not predictable.
        </p>
      </header>

      <button className="fetch-button" onClick={handleFetch} disabled={loading}>
        {loading ? 'Fetching…' : 'Fetch future games'}
      </button>

      {health && (
        <p className="freshness">
          Data as of {health.dataAsOf}
          {health.stale && <span className="stale-badge">STALE</span>}
        </p>
      )}

      {error && <p className="status error">Could not load games: {error}</p>}
      {predictError && <p className="status error">{predictError}</p>}

      {fetched && !loading && !error && games.length === 0 && (
        <p className="status">
          No upcoming games scheduled in the next {SCHEDULE_DAYS_AHEAD} days.
        </p>
      )}

      {games.length > 0 && (
        <ul className="fixture-list">
          {games.map((game) => (
            <FixtureCard
              key={fixtureKey(game)}
              game={game}
              predictable={cutoff !== null && game.gameDate <= cutoff}
              pending={pendingKey === fixtureKey(game)}
              onShowPredictions={() => handleShowPredictions(game)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function FixtureCard({ game, predictable, pending, onShowPredictions }) {
  return (
    <li className="fixture">
      <div className="fixture-teams">
        <span className="matchup">
          {game.awayTeamName} ({game.awayTeamAbbr}) @ {game.homeTeamName} (
          {game.homeTeamAbbr})
        </span>
        <span className="game-date">{game.gameDate}</span>
      </div>

      <div className="fixture-action">
        {predictable ? (
          <button onClick={onShowPredictions} disabled={pending}>
            {pending ? 'Predicting…' : 'Show predictions'}
          </button>
        ) : (
          <>
            <button disabled title="Predictions available for the next game only">
              Show predictions
            </button>
            <span className="unavailable-reason">
              predictions available for the next game only
            </span>
          </>
        )}
      </div>
    </li>
  );
}

export default BrowseView;
