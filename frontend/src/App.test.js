import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import App from './App';
import {
  createPrediction,
  createQuarterHalfPrediction,
  getHealth,
  getPlayerPropPredictions,
  getSchedule,
} from './api';

jest.mock('./api');

const HEALTH = {
  status: 'ok',
  // Per-family, matching what the service actually returns. Kept accurate
  // even though no component reads it: a fixture that lies about the API
  // is how a test suite stays green through a real breakage.
  modelsLoaded: { team: 7, quarter_half: 6, player_props: 10 },
  dataAsOf: '2026-04-12',
  daysBehind: 131,
  stale: true,
};

// Within the cutoff (dataAsOf + 1), so predictable.
const NEXT_GAME = {
  homeTeamId: 1610612737,
  homeTeamAbbr: 'ATL',
  homeTeamName: 'Atlanta Hawks',
  awayTeamId: 1610612738,
  awayTeamAbbr: 'BOS',
  awayTeamName: 'Boston Celtics',
  gameDate: '2026-04-13',
};

// A real fixture from the 2026-27 schedule: months out, not predictable.
const FAR_GAME = {
  homeTeamId: 1610612765,
  homeTeamAbbr: 'DET',
  homeTeamName: 'Detroit Pistons',
  awayTeamId: 1610612738,
  awayTeamAbbr: 'BOS',
  awayTeamName: 'Boston Celtics',
  gameDate: '2026-10-20',
};

const PREDICTION_RESULT = {
  id: 4,
  homeTeamAbbreviation: 'ATL',
  awayTeamAbbreviation: 'BOS',
  gameDate: '2026-04-13',
  played: false,
  latestPrediction: {
    homeWinProbability: 0.4489843547344208,
    homeMargin: -1.0504975318908691,
    totalPoints: 229.5902557373047,
    reboundMargin: -0.34379392862319946,
    totalRebounds: 88.76042938232422,
    assistMargin: 2.2516348361968994,
    totalAssists: 51.080528259277344,
    dataAsOf: '2026-04-12',
    stale: true,
    predictedAt: '2026-08-21T14:08:57.382864Z',
  },
};

const QUARTER_HALF_RESULT = {
  gameId: 4,
  homeTeamAbbreviation: 'ATL',
  awayTeamAbbreviation: 'BOS',
  gameDate: '2026-04-13',
  prediction: {
    q1Spread: -1.124112908717173,
    q1Total: 59.26116643514509,
    q1WinnerProbability: 0.4469873035961624,
    q1WinnerConfidence: 'low',
    q1WinnerInterpretation: 'P(home leads | not tied)',
    half1Spread: -1.1725223199629748,
    half1Total: 117.88749888276107,
    half1WinnerProbability: 0.45369307064976155,
    half1WinnerConfidence: 'medium',
    half1WinnerInterpretation: 'P(home leads | not tied)',
    dataAsOf: '2026-04-12',
    stale: true,
    daysBehind: 139,
    predictedAt: '2026-08-29T10:00:00Z',
  },
};

function line(playerId, playerName, modelUsed) {
  return {
    playerId,
    playerName,
    predictedPoints: 10,
    predictedRebounds: 4,
    predictedAssists: 3,
    predictedThreesMade: 1,
    predictedPra: 17,
    modelUsed,
  };
}

const PLAYER_PROPS_RESULT = {
  gameId: 4,
  gameDate: '2026-04-13',
  homeTeam: {
    teamId: 1610612737,
    teamAbbreviation: 'ATL',
    availabilityKnown: false,
    availabilityNote:
      '21 players (21 via linear, 0 via xgb).  AVAILABILITY UNKNOWN - no injury report was available, so nobody has been excluded. This is not a clean bill of health.',
    players: [
      line(1, 'Home One', 'linear'),
      line(2, 'Home Two', 'linear'),
      line(3, 'Home Three', 'xgb'),
    ],
  },
  awayTeam: {
    teamId: 1610612738,
    teamAbbreviation: 'BOS',
    availabilityKnown: false,
    availabilityNote: '17 players (17 via linear, 0 via xgb).  AVAILABILITY UNKNOWN - no injury report was available.',
    players: [
      line(11, 'Away One', 'linear'),
      line(12, 'Away Two', 'linear'),
      line(13, 'Away Three', 'linear'),
      line(14, 'Away Four', 'linear'),
      line(15, 'Away Five', 'xgb'),
    ],
  },
};

/** Arms the API mocks and presses "Fetch future games". */
async function fetchSchedule(fixtures) {
  getSchedule.mockResolvedValue(fixtures);
  getHealth.mockResolvedValue(HEALTH);

  render(<App />);
  fireEvent.click(screen.getByRole('button', { name: 'Fetch future games' }));
  await waitFor(() => expect(getSchedule).toHaveBeenCalled());
}

/** The card for one fixture, found by its matchup text. */
function cardFor(text) {
  return screen
    .getAllByRole('listitem')
    .find((item) => within(item).queryByText(new RegExp(text)));
}

describe('browse view', () => {
  test('describes the app before anything is fetched', () => {
    render(<App />);

    expect(screen.getByText('Modellarium')).toBeInTheDocument();
    expect(screen.getByText(/A personal sportsbook/)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: 'Fetch future games' })
    ).toBeInTheDocument();
  });

  test('lists the fixtures the schedule returns', async () => {
    await fetchSchedule([NEXT_GAME, FAR_GAME]);

    expect(
      await screen.findByText(/Boston Celtics \(BOS\) @ Atlanta Hawks \(ATL\)/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Boston Celtics \(BOS\) @ Detroit Pistons \(DET\)/)
    ).toBeInTheDocument();
    expect(screen.getByText(/Data as of 2026-04-12/)).toBeInTheDocument();
    expect(screen.getByText('STALE')).toBeInTheDocument();
  });

  test('an empty schedule says so rather than showing nothing', async () => {
    await fetchSchedule([]);

    expect(
      await screen.findByText(/No upcoming games scheduled in the next 14 days/)
    ).toBeInTheDocument();
  });

  test('surfaces a failed schedule fetch', async () => {
    getSchedule.mockRejectedValue(
      new Error('Could not reach the NBA schedule API for season 2026-27')
    );
    getHealth.mockResolvedValue(HEALTH);

    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'Fetch future games' }));

    expect(
      await screen.findByText(/Could not reach the NBA schedule API/)
    ).toBeInTheDocument();
  });

  test('only fixtures within MAX_DAYS_AHEAD are actionable', async () => {
    await fetchSchedule([NEXT_GAME, FAR_GAME]);
    await screen.findByText(/Atlanta Hawks \(ATL\)/);

    expect(
      within(cardFor('Atlanta Hawks')).getByRole('button', {
        name: 'Show predictions',
      })
    ).toBeEnabled();

    const farCard = cardFor('Detroit Pistons');
    expect(
      within(farCard).getByRole('button', { name: 'Show predictions' })
    ).toBeDisabled();
    expect(
      within(farCard).getByText('predictions available for the next game only')
    ).toBeInTheDocument();

    // The unreachable one explains itself instead of vanishing.
    expect(
      screen.getAllByRole('button', { name: 'Show predictions' })
    ).toHaveLength(2);
  });

  test('a disabled fixture cannot start a request', async () => {
    await fetchSchedule([FAR_GAME]);
    await screen.findByText(/Detroit Pistons/);

    fireEvent.click(screen.getByRole('button', { name: 'Show predictions' }));

    expect(createPrediction).not.toHaveBeenCalled();
  });
});

describe('prediction view', () => {
  async function drillIn() {
    await fetchSchedule([NEXT_GAME, FAR_GAME]);
    await screen.findByText(/Atlanta Hawks \(ATL\)/);
    createPrediction.mockResolvedValue(PREDICTION_RESULT);
    createQuarterHalfPrediction.mockResolvedValue(QUARTER_HALF_RESULT);

    fireEvent.click(
      within(cardFor('Atlanta Hawks')).getByRole('button', {
        name: 'Show predictions',
      })
    );
    await screen.findByRole('heading', { name: 'BOS @ ATL' });
  }

  test('shows all seven model outputs for the chosen game', async () => {
    await drillIn();

    expect(createPrediction).toHaveBeenCalledWith({
      homeTeamId: 1610612737,
      awayTeamId: 1610612738,
      gameDate: '2026-04-13',
    });

    // Scoped to the full-game block, not the whole panel. Since the
    // quarter/half markets landed alongside these, several values collide
    // at one decimal place - the full-game home margin and the Q1 margin
    // both render as -1.1 - so an unscoped query now matches two elements
    // and would fail for a reason that has nothing to do with the models.
    const fullGame = screen
      .getByRole('heading', { name: 'Full game' })
      .closest('section');

    expect(within(fullGame).getByText('44.9%')).toBeInTheDocument();
    expect(within(fullGame).getByText('-1.1')).toBeInTheDocument();
    expect(within(fullGame).getByText('229.6')).toBeInTheDocument();
    expect(within(fullGame).getByText('-0.3')).toBeInTheDocument();
    expect(within(fullGame).getByText('88.8')).toBeInTheDocument();
    expect(within(fullGame).getByText('2.3')).toBeInTheDocument();
    expect(within(fullGame).getByText('51.1')).toBeInTheDocument();
    expect(within(fullGame).getByText('STALE')).toBeInTheDocument();
  });

  test('back returns to the same list without refetching', async () => {
    await drillIn();

    fireEvent.click(screen.getByRole('button', { name: '← Back' }));

    expect(
      await screen.findByText(/Boston Celtics \(BOS\) @ Atlanta Hawks \(ATL\)/)
    ).toBeInTheDocument();
    expect(screen.getByText(/Detroit Pistons \(DET\)/)).toBeInTheDocument();
    expect(getSchedule).toHaveBeenCalledTimes(1);
    expect(getHealth).toHaveBeenCalledTimes(1);
  });
});

test('a refused prediction still shows the real reason', async () => {
  // Defence in depth: the disabled state should make this unreachable, so
  // this proves the message survives if it is ever reached anyway.
  await fetchSchedule([NEXT_GAME]);
  await screen.findByText(/Atlanta Hawks \(ATL\)/);

  createPrediction.mockRejectedValue(
    new Error(
      'Inference service rejected the request: {"detail":"game_date 2026-10-20 is more than 1 day past the newest game in the data (2026-04-12)."}'
    )
  );

  fireEvent.click(screen.getByRole('button', { name: 'Show predictions' }));

  expect(
    await screen.findByText(/more than 1 day past the newest game/)
  ).toBeInTheDocument();
  expect(screen.queryByText(/Request failed with status/)).not.toBeInTheDocument();
});

describe('two-tab dashboard', () => {
  /** Into the detail view, with both eager calls armed. */
  async function drillIn() {
    await fetchSchedule([NEXT_GAME]);
    await screen.findByText(/Atlanta Hawks \(ATL\)/);
    createPrediction.mockResolvedValue(PREDICTION_RESULT);
    createQuarterHalfPrediction.mockResolvedValue(QUARTER_HALF_RESULT);
    getPlayerPropPredictions.mockResolvedValue(PLAYER_PROPS_RESULT);

    fireEvent.click(screen.getByRole('button', { name: 'Show predictions' }));
    await screen.findByRole('heading', { name: 'BOS @ ATL' });
  }

  /** The visible panel. getByRole skips hidden ones, which is the point. */
  function panel() {
    return screen.getByRole('tabpanel');
  }

  async function openPlayerTab() {
    fireEvent.click(screen.getByRole('tab', { name: 'Player Predictions' }));
    await screen.findByRole('checkbox', { name: 'Home One' });
  }

  function check(name) {
    fireEvent.click(screen.getByRole('checkbox', { name }));
  }

  /** Section headings currently rendered, in DOM order. */
  function openSections() {
    return within(panel())
      .queryAllByRole('heading', { level: 3 })
      .map((node) => node.textContent)
      .filter((text) => text.includes('model'))
      .map((text) => text.replace(/(linear|xgb) model/, '').trim());
  }

  // 1 - both eager calls land before the view paints, so the default tab
  // has no loading gap.
  test('team-level and quarter/half are both fetched before the view renders', async () => {
    await drillIn();

    expect(createPrediction).toHaveBeenCalledTimes(1);
    expect(createQuarterHalfPrediction).toHaveBeenCalledTimes(1);
    expect(createQuarterHalfPrediction).toHaveBeenCalledWith({
      homeTeamId: 1610612737,
      awayTeamId: 1610612738,
      gameDate: '2026-04-13',
    });

    // Both blocks are already on screen, with no loading state between.
    expect(within(panel()).getByText('44.9%')).toBeInTheDocument();
    expect(within(panel()).getByText('44.7%')).toBeInTheDocument();
    expect(screen.queryByText(/Loading/)).not.toBeInTheDocument();
    // The heavy call has NOT happened.
    expect(getPlayerPropPredictions).not.toHaveBeenCalled();
  });

  // 5 - the qualifiers are rendered, not merely present in the payload.
  test('Q1 winner shows its low-confidence tag and interpretation text', async () => {
    await drillIn();

    expect(within(panel()).getByText('low confidence')).toBeInTheDocument();
    expect(
      within(panel()).getAllByText('P(home leads | not tied)')
    ).toHaveLength(2);
    // The 1H winner is labelled differently, so the tag is per-market.
    expect(within(panel()).getByText('medium confidence')).toBeInTheDocument();
  });

  // 2 - lazy once, cached thereafter.
  test('player rosters fetch once, on first open, and never again', async () => {
    await drillIn();
    await openPlayerTab();

    expect(getPlayerPropPredictions).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('tab', { name: 'Game Predictions' }));
    await screen.findByText('44.9%');
    fireEvent.click(screen.getByRole('tab', { name: 'Player Predictions' }));
    await screen.findByRole('checkbox', { name: 'Home One' });

    expect(getPlayerPropPredictions).toHaveBeenCalledTimes(1);
  });

  // 3 - sections stack in selection order, not roster order.
  test('sections appear in the order players were checked', async () => {
    await drillIn();
    await openPlayerTab();

    check('Away Two');
    check('Home One');
    check('Away Five');

    expect(openSections()).toEqual(['Away Two', 'Home One', 'Away Five']);
  });

  // 4 - removing from the middle leaves the rest in place.
  test('unchecking the middle selection removes only that section', async () => {
    await drillIn();
    await openPlayerTab();

    check('Away Two');
    check('Home One');
    check('Away Five');
    check('Home One');

    expect(openSections()).toEqual(['Away Two', 'Away Five']);
    expect(screen.getByRole('checkbox', { name: 'Home One' })).not.toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Away Two' })).toBeChecked();
    expect(getPlayerPropPredictions).toHaveBeenCalledTimes(1);
  });

  test('each team column shows its availability caveat', async () => {
    await drillIn();
    await openPlayerTab();

    expect(
      within(panel()).getAllByText(/AVAILABILITY UNKNOWN/)
    ).toHaveLength(2);
  });

  test('the selection survives switching tabs', async () => {
    await drillIn();
    await openPlayerTab();

    check('Home Three');
    expect(openSections()).toEqual(['Home Three']);

    fireEvent.click(screen.getByRole('tab', { name: 'Game Predictions' }));
    await screen.findByText('44.9%');
    fireEvent.click(screen.getByRole('tab', { name: 'Player Predictions' }));

    expect(openSections()).toEqual(['Home Three']);
    expect(screen.getByRole('checkbox', { name: 'Home Three' })).toBeChecked();
  });
});
