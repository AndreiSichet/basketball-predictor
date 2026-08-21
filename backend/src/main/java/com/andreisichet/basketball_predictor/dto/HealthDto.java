package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;

/**
 * Dataset freshness as sent to clients.
 *
 * Separate from InferenceHealth so the snake_case field names stay on the
 * Python boundary and this API keeps one naming convention, the same split
 * InferenceResponse and PredictionDto already use.
 *
 * dataAsOf is what lets the client mark fixtures it cannot get a
 * prediction for, rather than letting someone click into a certain 400.
 */
public record HealthDto(
        String status,
        int modelsLoaded,
        LocalDate dataAsOf,
        int daysBehind,
        boolean stale) {

    public static HealthDto from(InferenceHealth health) {
        return new HealthDto(
                health.status(),
                health.modelsLoaded(),
                health.dataAsOf(),
                health.daysBehind(),
                health.stale());
    }
}
