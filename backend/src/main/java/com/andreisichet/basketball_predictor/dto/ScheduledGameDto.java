package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/**
 * An upcoming fixture offered to the client as a candidate to predict.
 *
 * Deliberately not a GameSummaryDto. Whether a fixture is predictable at
 * all is decided by the client against the freshness data from
 * /api/health, and enforced for real by the inference service.
 *
 * IT IS NOW BACKED BY A Game ROW, which reverses an earlier decision worth
 * naming. Fixtures used to be pure pass-through, persisted only when
 * someone asked for a prediction. ScheduleSyncService now caches them
 * ahead of time so the browse view reads from Postgres instead of the live
 * NBA API. The row is still empty of predictions until one is requested,
 * and both paths create it through the same GameLookup.findOrCreateGame,
 * so a cached fixture and a predicted one are the same row.
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
