package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/** Incoming body for POST /api/predictions. */
public record PredictionRequest(Long homeTeamId, Long awayTeamId, LocalDate gameDate) {
}
