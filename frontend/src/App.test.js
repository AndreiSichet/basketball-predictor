import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import App from './App';
import { createPrediction, getHealth, getSchedule } from './api';

jest.mock('./api');

const HEALTH = {
  status: 'ok',
  modelsLoaded: 7,
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

    expect(screen.getByText('Basketball Predictor')).toBeInTheDocument();
    expect(screen.getByText(/Seven self-trained models/)).toBeInTheDocument();
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

    expect(screen.getByText('44.9%')).toBeInTheDocument();
    expect(screen.getByText('-1.1')).toBeInTheDocument();
    expect(screen.getByText('229.6')).toBeInTheDocument();
    expect(screen.getByText('-0.3')).toBeInTheDocument();
    expect(screen.getByText('88.8')).toBeInTheDocument();
    expect(screen.getByText('2.3')).toBeInTheDocument();
    expect(screen.getByText('51.1')).toBeInTheDocument();
    expect(screen.getByText('STALE')).toBeInTheDocument();
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
