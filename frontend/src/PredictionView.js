import { useState } from 'react';
import GamePredictionsTab from './GamePredictionsTab';
import PlayerPredictionsTab from './PlayerPredictionsTab';
import { getPlayerPropPredictions } from './api';

/**
 * Detail view for one matchup, split into two dashboards.
 *
 * BACK IS NOT A TAB, and is kept visually and structurally apart from the
 * two. Leaving the detail view and switching which markets you are looking
 * at are different kinds of navigation, and putting them in one row would
 * invite the first to be clicked by accident.
 *
 * TWO FETCH STRATEGIES, on purpose:
 *
 *   Full-game and quarter/half arrive together, already fetched by
 *   BrowseView before this component mounts. Both are small, both are on
 *   the default tab, and the backend guarantees they share one Game row -
 *   so staggering them would buy nothing and show a loading gap on the
 *   view's first paint.
 *
 *   Player props are fetched on the FIRST click of their tab and then
 *   cached here. Twenty players by five stats is heavier than the other two
 *   calls combined, and a session may never open that tab at all. Paying
 *   for it up front would slow down every visit to serve some of them.
 *
 * Both panels stay MOUNTED once rendered, hidden rather than unmounted, so
 * switching tabs loses neither the roster fetch nor which players are
 * currently expanded.
 */
function PredictionView({ result, onBack }) {
  const [activeTab, setActiveTab] = useState('game');
  const [playerProps, setPlayerProps] = useState(null);
  const [loadingPlayers, setLoadingPlayers] = useState(false);
  const [playerError, setPlayerError] = useState(null);

  const game = result.game;
  const quarterHalf = result.quarterHalf?.prediction ?? null;

  async function showPlayerTab() {
    setActiveTab('player');

    // Already fetched, or already in flight: the cache is the whole reason
    // a second click costs nothing.
    if (playerProps || loadingPlayers) {
      return;
    }

    setLoadingPlayers(true);
    setPlayerError(null);

    try {
      const board = await getPlayerPropPredictions({
        homeTeamId: result.request.homeTeamId,
        awayTeamId: result.request.awayTeamId,
        gameDate: result.request.gameDate,
      });
      setPlayerProps(board);
    } catch (failure) {
      setPlayerError(failure.message);
    } finally {
      setLoadingPlayers(false);
    }
  }

  return (
    <div>
      <button className="back-button" onClick={onBack}>
        ← Back
      </button>

      <header className="intro">
        <h1>
          {game.awayTeamAbbreviation} @ {game.homeTeamAbbreviation}
        </h1>
        <p className="game-date">{game.gameDate}</p>
      </header>

      <div className="dashboard-tabs" role="tablist" aria-label="Prediction dashboards">
        <button
          role="tab"
          id="tab-game"
          aria-selected={activeTab === 'game'}
          aria-controls="panel-game"
          className={activeTab === 'game' ? 'tab active' : 'tab'}
          onClick={() => setActiveTab('game')}
        >
          Game Predictions
        </button>
        <button
          role="tab"
          id="tab-player"
          aria-selected={activeTab === 'player'}
          aria-controls="panel-player"
          className={activeTab === 'player' ? 'tab active' : 'tab'}
          onClick={showPlayerTab}
        >
          Player Predictions
        </button>
      </div>

      <div
        role="tabpanel"
        id="panel-game"
        aria-labelledby="tab-game"
        hidden={activeTab !== 'game'}
      >
        <GamePredictionsTab
          prediction={game.latestPrediction}
          quarterHalf={quarterHalf}
        />
      </div>

      <div
        role="tabpanel"
        id="panel-player"
        aria-labelledby="tab-player"
        hidden={activeTab !== 'player'}
      >
        {loadingPlayers && <p className="status">Loading player predictions…</p>}
        {playerError && (
          <p className="status error">
            Could not load player predictions: {playerError}
          </p>
        )}
        {playerProps && (
          <PlayerPredictionsTab
            homeTeam={playerProps.homeTeam}
            awayTeam={playerProps.awayTeam}
          />
        )}
      </div>
    </div>
  );
}

export default PredictionView;
