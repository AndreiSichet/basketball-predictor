package com.andreisichet.basketball_predictor.service;

import java.util.List;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.InferenceHealth;
import com.andreisichet.basketball_predictor.dto.InferencePlayerPropsResponse;
import com.andreisichet.basketball_predictor.dto.InferenceQuarterHalfResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;
import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;

/**
 * The one place that talks to the Python inference service.
 *
 * That sentence is now literally true, and it was not until this class
 * absorbed getHealth(). It arrived with three of the five calls, took
 * fetchSchedule when the schedule caching landed, and left /health behind
 * in ScheduleService with a second RestClient - so the class documented
 * itself as the single point of contact while a second one quietly existed
 * beside it. A comment that overstates is worse than no comment, because
 * the next person trusts it instead of checking.
 *
 * There is now exactly ONE RestClient instance in the backend, built once
 * against the configured base URL and shared by all five calls.
 *
 * THE STATUS MAPPING IS THE POINT, and it is deliberately not symmetric:
 *
 *   4xx from the service  -> 400 here. It rejected the input (unknown team,
 *                            date too far ahead, history too short). The
 *                            caller sent something wrong, so this must not
 *                            become a 500.
 *   5xx from the service  -> 502. It was reachable and broke on its own.
 *   unreachable           -> 503. The request was fine; we were not.
 *
 * The service's own message is always forwarded. spring.web.error
 * .include-message=always is what lets the frontend read it, so discarding
 * it here would make that setting pointless.
 */
@Component
public class InferenceClient {

    private static final ParameterizedTypeReference<List<InferenceScheduledGame>> SCHEDULE_TYPE =
            new ParameterizedTypeReference<>() {
            };

    private final RestClient client;

    public InferenceClient(
            RestClient.Builder builder,
            @Value("${inference.service.url}") String inferenceServiceUrl) {
        this.client = builder.baseUrl(inferenceServiceUrl).build();
    }

    /** The seven full-game models. */
    public InferenceResponse predict(InferenceRequest body) {
        return post("/predict", body, InferenceResponse.class);
    }

    /** The six Q1 / first-half models. */
    public InferenceQuarterHalfResponse predictQuarterHalf(InferenceRequest body) {
        return post("/predict/quarter-half", body, InferenceQuarterHalfResponse.class);
    }

    /** Both teams' prop boards in one call. */
    public InferencePlayerPropsResponse predictPlayerProps(InferenceRequest body) {
        return post("/predict/player-props", body, InferencePlayerPropsResponse.class);
    }

    /**
     * Upcoming regular-season fixtures, straight from nba_api.
     *
     * Read on a timer by ScheduleSyncService, not per request. A 5xx here
     * usually means the NBA API failed rather than the inference service,
     * which is why it maps to 502 like the others: reachable, but broken on
     * its own.
     */
    public List<InferenceScheduledGame> fetchSchedule(int daysAhead) {
        try {
            List<InferenceScheduledGame> fixtures = client.get()
                    .uri(builder -> builder.path("/schedule")
                            .queryParam("days_ahead", daysAhead)
                            .build())
                    .retrieve()
                    .body(SCHEDULE_TYPE);
            return fixtures == null ? List.of() : fixtures;
        } catch (RestClientResponseException error) {
            throw rejected("Schedule lookup failed", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /**
     * How fresh the underlying pipeline data is.
     *
     * Called live on every browse request and never cached: data_as_of
     * tracks the pipeline's own data, so a cached copy would have the
     * client deciding which fixtures are predictable from an out-of-date
     * cutoff - wrong at exactly the moment it matters most, the first
     * request after a pipeline rerun.
     *
     * NOTE, because this is a real behaviour change: while this lived in
     * ScheduleService it caught RestClientException only, so a 500 from the
     * inference service surfaced as 503 "unreachable". It now follows the
     * same mapping as every other call, and a 5xx becomes 502. That is the
     * more accurate answer - reachable but broken is not the same as
     * absent - and it makes all five calls behave identically.
     */
    public InferenceHealth getHealth() {
        try {
            return client.get()
                    .uri("/health")
                    .retrieve()
                    .body(InferenceHealth.class);
        } catch (RestClientResponseException error) {
            throw rejected("Health check failed", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    private <T> T post(String path, InferenceRequest body, Class<T> responseType) {
        try {
            return client.post()
                    .uri(path)
                    .body(body)
                    .retrieve()
                    .body(responseType);
        } catch (RestClientResponseException error) {
            throw rejected("Inference service rejected the request", error);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /**
     * The service answered, but with a failure. 4xx stays 4xx so a caller
     * error is not reported as ours; anything else is 502.
     *
     * Extracted at the point there would otherwise have been three
     * identical copies - the same reason this class exists at all.
     */
    private ResponseStatusException rejected(String context, RestClientResponseException error) {
        HttpStatus status = error.getStatusCode().is4xxClientError()
                ? HttpStatus.BAD_REQUEST
                : HttpStatus.BAD_GATEWAY;
        return new ResponseStatusException(
                status, context + ": " + error.getResponseBodyAsString());
    }

    /** Nothing answered at all. The request was fine; the dependency is not. */
    private ResponseStatusException unreachable(RestClientException error) {
        return new ResponseStatusException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: " + error.getMessage());
    }
}
