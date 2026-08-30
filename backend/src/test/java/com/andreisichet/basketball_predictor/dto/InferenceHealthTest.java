package com.andreisichet.basketball_predictor.dto;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDate;

import org.junit.jupiter.api.Test;

// Spring Boot 4 ships JACKSON 3, whose databind lives under tools.jackson,
// not com.fasterxml. Only the annotations stayed on the old coordinates
// (com.fasterxml.jackson.annotations:2.21), which is why @JsonProperty
// imports elsewhere in this package still look 2.x. Jackson 3 also has
// java.time support built in, so no JavaTimeModule to register.
import tools.jackson.databind.ObjectMapper;

/**
 * Pins the wire shape of the inference service's GET /health.
 *
 * THIS TEST EXISTS BECAUSE THE TWO SIDES DRIFTED ONCE. models_loaded was
 * changed on the Python side from a single integer to a per-family object,
 * and this record kept declaring an int. Nothing caught it: the backend
 * compiled, started, and served all three prediction endpoints correctly -
 * the only symptom was GET /api/health failing at runtime with "Error while
 * extracting response for type InferenceHealth", which took the entire
 * browse view down with it, since that is the one caller needing freshness
 * metadata before it can decide which fixtures are reachable.
 *
 * A compiler cannot see across an HTTP boundary, so the shape has to be
 * asserted somewhere. No Spring context here on purpose: this is about
 * Jackson and the record's declarations, and a context-free test runs in
 * milliseconds inside the CI job that already exists.
 */
class InferenceHealthTest {

    // Copied verbatim from a real response, not hand-written from the
    // record. A fixture derived from the Java side could not have caught
    // the original bug, because it would have drifted with it.
    private static final String REAL_PAYLOAD = """
            {
              "status": "ok",
              "models_loaded": {
                "team": 7,
                "quarter_half": 6,
                "player_props": 10
              },
              "data_as_of": "2026-04-12",
              "days_behind": 139,
              "stale": true
            }
            """;

    private final ObjectMapper mapper = new ObjectMapper();

    @Test
    void deserializesThePerFamilyModelBreakdown() throws Exception {
        InferenceHealth health = mapper.readValue(REAL_PAYLOAD, InferenceHealth.class);

        assertThat(health.status()).isEqualTo("ok");
        assertThat(health.dataAsOf()).isEqualTo(LocalDate.of(2026, 4, 12));
        assertThat(health.daysBehind()).isEqualTo(139);
        assertThat(health.stale()).isTrue();

        // The field that broke: an object, not a count.
        assertThat(health.modelsLoaded())
                .containsExactlyInAnyOrderEntriesOf(
                        java.util.Map.of("team", 7, "quarter_half", 6, "player_props", 10));
    }

    @Test
    void carriesTheBreakdownThroughToTheClientDto() throws Exception {
        InferenceHealth health = mapper.readValue(REAL_PAYLOAD, InferenceHealth.class);

        HealthDto dto = HealthDto.from(health);

        // Passed through rather than flattened or dropped - the per-family
        // split is the diagnostic the endpoint exists to expose.
        assertThat(dto.modelsLoaded()).isEqualTo(health.modelsLoaded());
        assertThat(dto.dataAsOf()).isEqualTo(health.dataAsOf());
        assertThat(dto.stale()).isTrue();
    }
}
