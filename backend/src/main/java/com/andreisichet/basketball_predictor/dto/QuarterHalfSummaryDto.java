package com.andreisichet.basketball_predictor.dto;

import java.time.Instant;
import java.time.LocalDate;

import com.andreisichet.basketball_predictor.model.QuarterHalfPrediction;

/**
 * What POST /api/predictions/quarter-half returns.
 *
 * Mirrors GameSummaryDto - the game it belongs to, then the prediction
 * nested inside - so a client handles all three prediction endpoints the
 * same way.
 */
public record QuarterHalfSummaryDto(
        Long gameId,
        String homeTeamAbbreviation,
        String awayTeamAbbreviation,
        LocalDate gameDate,
        Prediction prediction) {

    /**
     * THE FOUR QUALIFIER FIELDS ARE REQUIRED, NEVER OMITTED, and that is
     * the whole reason this record is shaped the way it is.
     *
     * q1WinnerConfidence / half1WinnerConfidence - q1_winner scores 0.5796
     * accuracy against a 0.5184 always-home baseline. It ships labelled
     * "low" rather than hidden, so a client rendering it beside the others
     * without the label would present a near-coin-flip as an equal peer.
     *
     * q1WinnerInterpretation / half1WinnerInterpretation - always
     * "P(home leads | not tied)". These two models were trained only on
     * periods that had a winner, because 611 tied first quarters have no
     * binary label. The number is NOT the same quantity as the full-game
     * moneyline and must not be displayed as if it were.
     *
     * None of the four are read from the database. They are static facts
     * about the models, copied from the inference payload - see the comment
     * on the QuarterHalfPrediction entity for why storing them would mean
     * duplicating a constant once per row.
     */
    public record Prediction(
            double q1Spread,
            double q1Total,
            double q1WinnerProbability,
            String q1WinnerConfidence,
            String q1WinnerInterpretation,
            double half1Spread,
            double half1Total,
            double half1WinnerProbability,
            String half1WinnerConfidence,
            String half1WinnerInterpretation,
            LocalDate dataAsOf,
            boolean stale,
            int daysBehind,
            Instant predictedAt) {

        /**
         * Numbers from the persisted row, qualifiers from the live payload.
         *
         * The split is deliberate: what was stored is what this system
         * produced and can be audited later, while the labels describe the
         * models themselves and would be identical on every row.
         */
        public static Prediction of(
                QuarterHalfPrediction saved,
                InferenceQuarterHalfResponse inference,
                InferenceQuarterHalfResponse.Market q1Winner,
                InferenceQuarterHalfResponse.Market half1Winner) {
            return new Prediction(
                    saved.getQ1Spread(),
                    saved.getQ1Total(),
                    saved.getQ1WinnerProbability(),
                    q1Winner.confidence(),
                    q1Winner.interpretation(),
                    saved.getHalf1Spread(),
                    saved.getHalf1Total(),
                    saved.getHalf1WinnerProbability(),
                    half1Winner.confidence(),
                    half1Winner.interpretation(),
                    saved.getDataAsOf(),
                    saved.isStale(),
                    inference.daysBehind(),
                    saved.getPredictedAt());
        }
    }
}
