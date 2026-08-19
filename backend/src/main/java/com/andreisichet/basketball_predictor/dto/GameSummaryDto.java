package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/**
 * A game as listed by the API.
 *
 * latestPrediction is null when nothing has been predicted for the game
 * yet, which will be normal once schedule fetching creates games ahead of
 * any prediction request.
 */
public record GameSummaryDto(
        Long id,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        boolean played,
        PredictionDto latestPrediction) {
}
