package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

/**
 * Dataset freshness as sent to clients.
 *
 * Separate from InferenceHealth so the snake_case field names stay on the
 * Python boundary and this API keeps one naming convention, the same split
 * InferenceResponse and PredictionDto already use.
 *
 * dataAsOf is what lets the client mark fixtures it cannot get a
 * prediction for, rather than letting someone click into a certain 400.
 *
 * modelsLoaded is PASSED THROUGH rather than dropped, even though the
 * frontend reads only dataAsOf and stale. This is a health endpoint, and
 * the per-family breakdown is exactly the diagnostic the inference service
 * broke it out to provide - discarding it here would mean the one place an
 * operator looks cannot say which model family failed to load. A client
 * that does not want it simply ignores it.
 */
public record HealthDto(
        String status,
        Map<String, Integer> modelsLoaded,
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
