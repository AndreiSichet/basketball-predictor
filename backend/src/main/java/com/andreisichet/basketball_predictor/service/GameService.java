package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.PredictionDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Prediction;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.PredictionRepository;

@Service
public class GameService {

    private final GameRepository gameRepository;
    private final PredictionRepository predictionRepository;

    public GameService(GameRepository gameRepository, PredictionRepository predictionRepository) {
        this.gameRepository = gameRepository;
        this.predictionRepository = predictionRepository;
    }

    /**
     * Upcoming games, soonest first, each with its newest prediction.
     *
     * "Upcoming" means today or later, not merely "not marked played".
     * Nothing sets played = true yet, so without the date bound every
     * cached fixture would age into the past and stay in this list
     * permanently. See the repository method for the full reasoning.
     *
     * TWO QUERIES, REGARDLESS OF HOW MANY GAMES. This used to run one
     * lookup per game inside the mapping loop - fine when the table held a
     * handful of rows someone had actually predicted, and a measured 486
     * SQL selects for a single call once ScheduleSyncService began caching
     * hundreds of fixtures ahead of time. The N+1 was documented as a known
     * trade-off long before it bit; caching fixtures is what turned "~16
     * queries at a full slate" into something worth fixing.
     *
     * Transactional because Game's team relations are LAZY and
     * open-in-view is off: the session has to still be open when they are
     * read, and they have to be read here rather than during serialization.
     */
    @Transactional(readOnly = true)
    public List<GameSummaryDto> getUpcomingGames() {
        List<Game> games = gameRepository
                .findByPlayedFalseAndGameDateGreaterThanEqualOrderByGameDateAsc(
                        LocalDate.now());

        if (games.isEmpty()) {
            return List.of();
        }

        Map<Long, Prediction> latestByGameId = latestPredictions(games);

        return games.stream()
                .map(game -> toSummary(game, latestByGameId.get(game.getId())))
                .toList();
    }

    /**
     * The newest prediction for each of these games, in one query.
     *
     * The repository returns every prediction for the batch ordered newest
     * first; the merge function keeps whichever arrived first for a given
     * game id, which is therefore its newest. See the note on
     * findByGameIdInOrderByPredictedAtDesc for why this fold lives here
     * rather than in a derived query.
     */
    private Map<Long, Prediction> latestPredictions(List<Game> games) {
        List<Long> gameIds = games.stream().map(Game::getId).toList();

        return predictionRepository.findByGameIdInOrderByPredictedAtDesc(gameIds).stream()
                .collect(Collectors.toMap(
                        prediction -> prediction.getGame().getId(),
                        Function.identity(),
                        (newest, older) -> newest));
    }

    private GameSummaryDto toSummary(Game game, Prediction latest) {
        return new GameSummaryDto(
                game.getId(),
                game.getHomeTeam().getAbbreviation(),
                game.getAwayTeam().getAbbreviation(),
                game.getGameDate(),
                game.isPlayed(),
                latest == null ? null : PredictionDto.from(latest));
    }
}
