package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Wire shape of POST /predict/quarter-half on the Python service.
 *
 * The predictions arrive as a LIST of markets rather than an object with
 * six named fields, so this mirrors that rather than flattening it here.
 * The service chose a list because each entry carries its own confidence
 * and, on the two winner markets, an interpretation string; a flat object
 * would have needed eighteen fields to say the same thing.
 *
 * interpretation is null on the four regression markets, and that is the
 * point of it being null rather than an empty string: a client can test for
 * presence.
 */
public record InferenceQuarterHalfResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        List<Market> predictions) {

    public record Market(
            String market,
            double value,
            String confidence,
            String interpretation) {
    }

    /**
     * One market by name.
     *
     * Throws rather than returning null if the name is absent: a missing
     * market means the Python service changed shape, and silently writing a
     * 0.0 into the database would be far worse than failing the request.
     */
    public Market market(String name) {
        return predictions.stream()
                .filter(entry -> entry.market().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "Inference service returned no market named " + name
                                + ". Its response shape has changed."));
    }
}
