package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

import com.andreisichet.basketball_predictor.model.PlayerPropPrediction;

/** Both sides' prop boards for one fixture, plus the usual freshness block. */
public record PlayerPropsResponseDto(
        Long gameId,
        LocalDate gameDate,
        TeamBoard homeTeam,
        TeamBoard awayTeam,
        LocalDate dataAsOf,
        boolean stale,
        int daysBehind,
        Instant predictedAt) {

    /**
     * One side's board.
     *
     * availabilityNote IS ALWAYS PRESENT, in both states. Today it always
     * carries the unknown-availability caveat, because the container cannot
     * fetch the injury report - but the field is deliberately modelled as
     * "what is known about availability", not as "the warning". Once the
     * October packaging work lands, availabilityKnown becomes true and this
     * carries the confirmed case instead. A field that only exists while
     * something is broken gets deleted the moment it is fixed, and then the
     * good news has nowhere to go.
     */
    public record TeamBoard(
            Long teamId,
            String teamAbbreviation,
            boolean availabilityKnown,
            String availabilityNote,
            List<PlayerLine> players) {
    }

    /**
     * One player's five predictions.
     *
     * modelUsed is exposed rather than inferred: the hybrid routes complete
     * rolling histories to a linear model and everything else to XGBoost,
     * and which one answered is a real property of the number. Same
     * transparency standard as the stale badge and the confidence label.
     */
    public record PlayerLine(
            Long playerId,
            String playerName,
            double predictedPoints,
            double predictedRebounds,
            double predictedAssists,
            double predictedThreesMade,
            double predictedPra,
            String modelUsed) {

        public static PlayerLine from(PlayerPropPrediction saved) {
            return new PlayerLine(
                    saved.getPlayer().getId(),
                    saved.getPlayer().getName(),
                    saved.getPredictedPoints(),
                    saved.getPredictedRebounds(),
                    saved.getPredictedAssists(),
                    saved.getPredictedThreesMade(),
                    saved.getPredictedPra(),
                    saved.getModelUsed());
        }
    }
}
