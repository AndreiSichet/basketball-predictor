package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Response from the Python service's GET /health.
 *
 * modelsLoaded IS A MAP, NOT A COUNT, and the reason is worth keeping.
 * The inference service used to return a single integer; it now returns a
 * per-family breakdown - {"team": 7, "quarter_half": 6, "player_props": 10}
 * - so that a failure names which family broke instead of showing a total
 * that looks plausible when one family loaded twice and another not at all.
 *
 * This record was not updated when that changed, and the mismatch was
 * invisible until runtime: Jackson cannot map a JSON object onto an int, so
 * GET /api/health started failing with "Error while extracting response for
 * type InferenceHealth" while all three prediction endpoints kept working.
 * The browse view was the only casualty, because it is the only caller that
 * needs the freshness metadata before it can decide which fixtures are
 * reachable.
 *
 * InferenceHealthTest pins the shape so the two sides cannot drift apart
 * silently a second time.
 */
public record InferenceHealth(
        String status,
        @JsonProperty("models_loaded") Map<String, Integer> modelsLoaded,
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        @JsonProperty("days_behind") int daysBehind,
        boolean stale) {
}
