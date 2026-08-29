package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * Team resolution and find-or-create for a Game.
 *
 * Extracted alongside InferenceClient and for the same reason: three
 * prediction services now need identical behaviour here, and three copies
 * of "find the game or make one" is how the same fixture ends up with three
 * rows in the games table.
 *
 * ONE GAME PER MATCHUP PER DATE, whichever markets are being priced. The
 * full-game, quarter/half and player-prop endpoints all resolve to the same
 * Game row, because they describe the same fixture. Predictions are
 * append-only and immutable; the Game they hang off is reused.
 */
@Component
public class GameLookup {

    private final TeamRepository teamRepository;
    private final GameRepository gameRepository;

    public GameLookup(TeamRepository teamRepository, GameRepository gameRepository) {
        this.teamRepository = teamRepository;
        this.gameRepository = gameRepository;
    }

    /** The team, or a 400 naming the id that was not recognised. */
    public Team requireTeam(Long teamId) {
        if (teamId == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Team id is required.");
        }
        return teamRepository.findById(teamId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.BAD_REQUEST, "Unknown team id: " + teamId));
    }

    /**
     * The existing Game for this matchup and date, or a new one.
     *
     * Callers must run this AFTER the inference call, never before. That
     * ordering is not stylistic: creating the row first left an orphan Game
     * behind for every rejected request, which is how BOS-vs-BOS rows once
     * appeared in the database.
     */
    public Game findOrCreateGame(Team homeTeam, Team awayTeam, LocalDate gameDate) {
        return gameRepository.findByHomeTeamAndAwayTeamAndGameDate(homeTeam, awayTeam, gameDate)
                .orElseGet(() -> {
                    Game game = new Game();
                    game.setHomeTeam(homeTeam);
                    game.setAwayTeam(awayTeam);
                    game.setGameDate(gameDate);
                    game.setPlayed(false);
                    return gameRepository.save(game);
                });
    }
}
