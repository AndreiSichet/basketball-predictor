package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Response from the Python inference service.
 *
 * Only the fields used here are declared; Spring Boot's Jackson ignores the
 * rest (the service also echoes back the team ids and game date).
 */
public record InferenceResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        Predictions predictions) {

    public record Predictions(
            @JsonProperty("home_win_probability") double homeWinProbability,
            @JsonProperty("home_margin") double homeMargin,
            @JsonProperty("total_points") double totalPoints,
            @JsonProperty("rebound_margin") double reboundMargin,
            @JsonProperty("total_rebounds") double totalRebounds,
            @JsonProperty("assist_margin") double assistMargin,
            @JsonProperty("total_assists") double totalAssists) {
    }
}
