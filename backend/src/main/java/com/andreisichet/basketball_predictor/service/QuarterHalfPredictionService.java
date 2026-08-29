package com.andreisichet.basketball_predictor.service;

import java.time.Instant;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.InferenceQuarterHalfResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.dto.QuarterHalfSummaryDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.QuarterHalfPrediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.QuarterHalfPredictionRepository;

/**
 * Q1 and first-half markets.
 *
 * A sibling of PredictionService, not an extension of it - the same
 * decision the Python service made for the same reason. The request shape
 * happens to match, but the responses do not: six markets with confidence
 * labels and conditional-probability caveats is a different thing from
 * seven plain numbers, and forcing them into one endpoint would mean a
 * response where half the fields are always null.
 *
 * Reuses GameLookup so all three prediction types share one games table and
 * one find-or-create rule.
 */
@Service
public class QuarterHalfPredictionService {

    // Market names as the Python service emits them. Named constants
    // because a typo here would surface as "no market named ..." at
    // runtime rather than as a compile error.
    private static final String Q1_SPREAD = "q1_home_margin";
    private static final String Q1_TOTAL = "q1_total_points";
    private static final String Q1_WINNER = "q1_home_win_probability";
    private static final String HALF1_SPREAD = "half1_home_margin";
    private static final String HALF1_TOTAL = "half1_total_points";
    private static final String HALF1_WINNER = "half1_home_win_probability";

    private final GameLookup gameLookup;
    private final QuarterHalfPredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;

    public QuarterHalfPredictionService(
            GameLookup gameLookup,
            QuarterHalfPredictionRepository predictionRepository,
            InferenceClient inferenceClient) {
        this.gameLookup = gameLookup;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
    }

    /**
     * Transactional so a rejected request writes nothing, and the inference
     * call runs BEFORE any insert so the usual failure never reaches the
     * database at all. Same ordering as PredictionService, and for the same
     * reason: the original endpoint once left orphan Game rows behind for
     * every rejected request.
     */
    @Transactional
    public QuarterHalfSummaryDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferenceQuarterHalfResponse inference = inferenceClient.predictQuarterHalf(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(), request.gameDate()));

        Game game = gameLookup.findOrCreateGame(homeTeam, awayTeam, request.gameDate());
        QuarterHalfPrediction saved = predictionRepository.save(toEntity(game, inference));

        return new QuarterHalfSummaryDto(
                game.getId(),
                homeTeam.getAbbreviation(),
                awayTeam.getAbbreviation(),
                game.getGameDate(),
                QuarterHalfSummaryDto.Prediction.of(
                        saved,
                        inference,
                        inference.market(Q1_WINNER),
                        inference.market(HALF1_WINNER)));
    }

    private QuarterHalfPrediction toEntity(Game game, InferenceQuarterHalfResponse inference) {
        QuarterHalfPrediction prediction = new QuarterHalfPrediction();
        prediction.setGame(game);
        prediction.setQ1Spread(inference.market(Q1_SPREAD).value());
        prediction.setQ1Total(inference.market(Q1_TOTAL).value());
        prediction.setQ1WinnerProbability(inference.market(Q1_WINNER).value());
        prediction.setHalf1Spread(inference.market(HALF1_SPREAD).value());
        prediction.setHalf1Total(inference.market(HALF1_TOTAL).value());
        prediction.setHalf1WinnerProbability(inference.market(HALF1_WINNER).value());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        prediction.setPredictedAt(Instant.now());
        return prediction;
    }
}
