import { useState } from 'react';
import PlayerPredictionSection from './PlayerPredictionSection';

/**
 * Both rosters, and a stack of prediction sections for whoever is checked.
 *
 * SELECTION ORDER IS THE MECHANIC, and it is held as an ordered array
 * rather than a set or a per-player boolean. Checking a player appends him;
 * unchecking removes that specific entry from wherever it sits. Because the
 * sections below are a direct render of the array, "opens at the bottom,
 * closes in place, everything below shifts up" falls out of the data
 * structure instead of needing any bookkeeping of its own.
 *
 * A checkbox's checked state is DERIVED from array membership, never
 * tracked alongside it. Two sources of truth for the same fact is how a
 * checkbox ends up ticked with no section under it.
 */
function PlayerPredictionsTab({ homeTeam, awayTeam }) {
  const [selectedPlayers, setSelectedPlayers] = useState([]);

  function isSelected(player) {
    return selectedPlayers.some((entry) => entry.playerId === player.playerId);
  }

  function toggle(player) {
    setSelectedPlayers((current) =>
      current.some((entry) => entry.playerId === player.playerId)
        ? // Remove this one only. Everyone else keeps their relative
          // position, so the surviving sections do not reorder.
          current.filter((entry) => entry.playerId !== player.playerId)
        : // Append, so the newest selection opens at the bottom.
          [...current, player]
    );
  }

  return (
    <div>
      <div className="roster-columns">
        <RosterColumn team={homeTeam} side="Home" isSelected={isSelected} onToggle={toggle} />
        <RosterColumn team={awayTeam} side="Away" isSelected={isSelected} onToggle={toggle} />
      </div>

      {selectedPlayers.length === 0 ? (
        <p className="status">
          Select a player to see their predicted line.
        </p>
      ) : (
        <div className="player-sections">
          {selectedPlayers.map((player) => (
            <PlayerPredictionSection key={player.playerId} player={player} />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * One team's checkbox list.
 *
 * The availability note is rendered in the column header whenever
 * availability is not known, rather than tucked away. Right now that is
 * always - the inference container cannot fetch the injury report - so
 * every roster here is "everyone with recent history", not "everyone
 * expected to play. Saying so is the point; a roster that quietly omitted
 * the caveat would read as a confirmed lineup.
 */
function RosterColumn({ team, side, isSelected, onToggle }) {
  return (
    <section className="roster-column" aria-label={`${team.teamAbbreviation} roster`}>
      <h3>
        {side}: {team.teamAbbreviation}
      </h3>

      {!team.availabilityKnown && (
        <p className="availability-note">{team.availabilityNote}</p>
      )}

      <ul className="roster-list">
        {team.players.map((player) => (
          <li key={player.playerId}>
            <label>
              <input
                type="checkbox"
                checked={isSelected(player)}
                onChange={() => onToggle(player)}
              />
              {player.playerName}
            </label>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default PlayerPredictionsTab;
