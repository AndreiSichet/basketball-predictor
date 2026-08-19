package com.andreisichet.basketball_predictor.service;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.PredictionDto;
import com.andreisichet.basketball_predictor.model.Game;
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
     * Transactional because the Team relations are LAZY and open-in-view is
     * off: the session has to still be open when they are read, and it has
     * to be read here rather than during serialization.
     */
    @Transactional(readOnly = true)
    public List<GameSummaryDto> getUpcomingGames() {
        return gameRepository.findByPlayedFalseOrderByGameDateAsc().stream()
                .map(this::toSummary)
                .toList();
    }

    private GameSummaryDto toSummary(Game game) {
        PredictionDto latestPrediction = predictionRepository
                .findTopByGameIdOrderByPredictedAtDesc(game.getId())
                .map(PredictionDto::from)
                .orElse(null);

        return new GameSummaryDto(
                game.getId(),
                game.getHomeTeam().getAbbreviation(),
                game.getAwayTeam().getAbbreviation(),
                game.getGameDate(),
                game.isPlayed(),
                latestPrediction);
    }
}
