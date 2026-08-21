import { useEffect, useState } from 'react';
import { createPrediction, getTeams } from './api';

/**
 * Form for requesting a prediction on a matchup.
 *
 * Hands the created GameSummaryDto straight back to the parent via
 * onCreated rather than refetching: the POST returns the same shape the
 * list already renders, so a second round trip would buy nothing.
 */
function CreatePredictionForm({ onCreated }) {
  const [teams, setTeams] = useState([]);
  const [homeTeamId, setHomeTeamId] = useState('');
  const [awayTeamId, setAwayTeamId] = useState('');
  const [gameDate, setGameDate] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await getTeams();
        if (!cancelled) {
          setTeams(data);
        }
      } catch (failure) {
        if (!cancelled) {
          setFormError(failure.message);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, []);

  // The backend rejects this too, but the answer is knowable here, so
  // there is no reason to spend a round trip finding it out.
  const sameTeam = homeTeamId !== '' && homeTeamId === awayTeamId;
  const incomplete = homeTeamId === '' || awayTeamId === '' || gameDate === '';

  async function handleSubmit(event) {
    event.preventDefault();
    setSubmitting(true);

    try {
      const game = await createPrediction({
        homeTeamId: Number(homeTeamId),
        awayTeamId: Number(awayTeamId),
        gameDate,
      });
      setFormError(null);
      onCreated(game);
    } catch (failure) {
      // Where the backend's real reason — unknown team, date too far
      // ahead, inference service down — reaches a person.
      setFormError(failure.message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="prediction-form" onSubmit={handleSubmit}>
      <h2>New prediction</h2>

      <div className="form-row">
        <label htmlFor="homeTeam">Home</label>
        <select
          id="homeTeam"
          value={homeTeamId}
          onChange={(event) => setHomeTeamId(event.target.value)}
        >
          <option value="">Select a team</option>
          {teams.map((team) => (
            <option key={team.id} value={team.id}>
              {team.name} ({team.abbreviation})
            </option>
          ))}
        </select>
      </div>

      <div className="form-row">
        <label htmlFor="awayTeam">Away</label>
        <select
          id="awayTeam"
          value={awayTeamId}
          onChange={(event) => setAwayTeamId(event.target.value)}
        >
          <option value="">Select a team</option>
          {teams.map((team) => (
            <option key={team.id} value={team.id}>
              {team.name} ({team.abbreviation})
            </option>
          ))}
        </select>
      </div>

      <div className="form-row">
        <label htmlFor="gameDate">Date</label>
        <input
          id="gameDate"
          type="date"
          value={gameDate}
          onChange={(event) => setGameDate(event.target.value)}
        />
      </div>

      {sameTeam && (
        <p className="form-error">Home and away teams must differ</p>
      )}
      {formError && <p className="form-error">{formError}</p>}

      <button type="submit" disabled={submitting || sameTeam || incomplete}>
        {submitting ? 'Predicting…' : 'Predict'}
      </button>
    </form>
  );
}

export default CreatePredictionForm;
