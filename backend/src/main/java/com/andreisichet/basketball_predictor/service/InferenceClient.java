package com.andreisichet.basketball_predictor.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.InferencePlayerPropsResponse;
import com.andreisichet.basketball_predictor.dto.InferenceQuarterHalfResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;

/**
 * The one place that talks to the Python inference service.
 *
 * Extracted when the second and third callers arrived, not after. Until now
 * PredictionService owned both the RestClient and the rule for turning the
 * Python service's failures into Java statuses; copying that rule into two
 * more services is how one endpoint quietly starts translating a 400 into a
 * 500 while the others do not.
 *
 * It also collapses three RestClient instances into one. Each service used
 * to build its own from the injected builder against the same base URL,
 * which was noted as acceptable at two and would have been careless at
 * three.
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

    private final RestClient client;

    public InferenceClient(
            RestClient.Builder builder,
            @Value("${inference.service.url}") String inferenceServiceUrl) {
        this.client = builder.baseUrl(inferenceServiceUrl).build();
    }

    /** The seven full-game models. */
    public InferenceResponse predict(InferenceRequest body) {
        return call("/predict", body, InferenceResponse.class);
    }

    /** The six Q1 / first-half models. */
    public InferenceQuarterHalfResponse predictQuarterHalf(InferenceRequest body) {
        return call("/predict/quarter-half", body, InferenceQuarterHalfResponse.class);
    }

    /** Both teams' prop boards in one call. */
    public InferencePlayerPropsResponse predictPlayerProps(InferenceRequest body) {
        return call("/predict/player-props", body, InferencePlayerPropsResponse.class);
    }

    private <T> T call(String path, InferenceRequest body, Class<T> responseType) {
        try {
            return client.post()
                    .uri(path)
                    .body(body)
                    .retrieve()
                    .body(responseType);
        } catch (RestClientResponseException error) {
            HttpStatus status = error.getStatusCode().is4xxClientError()
                    ? HttpStatus.BAD_REQUEST
                    : HttpStatus.BAD_GATEWAY;
            throw new ResponseStatusException(
                    status,
                    "Inference service rejected the request: " + error.getResponseBodyAsString());
        } catch (RestClientException error) {
            throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Inference service unreachable: " + error.getMessage());
        }
    }
}
