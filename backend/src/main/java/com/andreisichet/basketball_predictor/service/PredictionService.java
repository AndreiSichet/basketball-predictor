package com.andreisichet.basketball_predictor.service;

import java.time.Instant;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;
import com.andreisichet.basketball_predictor.dto.PredictionDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Prediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.PredictionRepository;

/**
 * The seven full-game models.
 *
 * The RestClient, the 4xx/5xx translation and the find-or-create rule used
 * to live here. They moved to InferenceClient and GameLookup when the
 * quarter/half and player-prop services arrived and needed identical
 * behaviour - extracting at two or three call sites rather than waiting for
 * the copies to drift.
 */
@Service
public class PredictionService {

    private final GameLookup gameLookup;
    private final PredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;

    public PredictionService(
            GameLookup gameLookup,
            PredictionRepository predictionRepository,
            InferenceClient inferenceClient) {
        this.gameLookup = gameLookup;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
    }

    /**
     * Transactional so a rejected request writes nothing. The inference call
     * also runs before any insert, so the usual failure never reaches the
     * database at all and rollback is only a backstop.
     *
     * Returns a DTO, not the entity: with open-in-view off, serializing an
     * entity outside the session fails on its LAZY relations.
     */
    @Transactional
    public GameSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferenceResponse inference = inferenceClient.predict(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(), request.gameDate()));

        Game game = gameLookup.findOrCreateGame(homeTeam, awayTeam, request.gameDate());
        Prediction prediction = predictionRepository.save(toPrediction(game, inference));

        return new GameSummaryDto(
                game.getId(),
                homeTeam.getAbbreviation(),
                awayTeam.getAbbreviation(),
                game.getGameDate(),
                game.isPlayed(),
                PredictionDto.from(prediction));
    }

    private Prediction toPrediction(Game game, InferenceResponse inference) {
        InferenceResponse.Predictions values = inference.predictions();

        Prediction prediction = new Prediction();
        prediction.setGame(game);
        prediction.setHomeWinProbability(values.homeWinProbability());
        prediction.setHomeMargin(values.homeMargin());
        prediction.setTotalPoints(values.totalPoints());
        prediction.setReboundMargin(values.reboundMargin());
        prediction.setTotalRebounds(values.totalRebounds());
        prediction.setAssistMargin(values.assistMargin());
        prediction.setTotalAssists(values.totalAssists());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        prediction.setPredictedAt(Instant.now());
        return prediction;
    }
}
