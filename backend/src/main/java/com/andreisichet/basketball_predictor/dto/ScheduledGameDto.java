package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/**
 * An upcoming fixture offered to the client as a candidate to predict.
 *
 * Deliberately not a GameSummaryDto and deliberately not backed by a Game
 * row: nothing is persisted until someone actually asks for a prediction.
 * Whether a fixture is predictable at all is decided by the client against
 * the freshness data from /api/health, and enforced for real by the
 * inference service.
 */
public record ScheduledGameDto(
        Long homeTeamId,
        String homeTeamAbbr,
        String homeTeamName,
        Long awayTeamId,
        String awayTeamAbbr,
        String awayTeamName,
        LocalDate gameDate) {
}
