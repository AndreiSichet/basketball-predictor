package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Body sent to the Python inference service.
 *
 * Field names are snake_case there, so they are mapped explicitly rather
 * than relying on a global naming strategy that would affect every DTO.
 */
public record InferenceRequest(
        @JsonProperty("home_team_id") Long homeTeamId,
        @JsonProperty("away_team_id") Long awayTeamId,
        @JsonProperty("game_date") LocalDate gameDate) {
}
