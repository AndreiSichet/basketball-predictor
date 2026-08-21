package com.andreisichet.basketball_predictor.service;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.dto.InferenceHealth;
import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * Read-only pass-through to the inference service's schedule and health.
 *
 * Nothing here touches Game or Prediction: a fixture only becomes a row
 * when someone requests a prediction for it, through PredictionService.
 * The point of proxying at all is that the frontend keeps one base URL.
 */
@Service
public class ScheduleService {

    private static final ParameterizedTypeReference<List<InferenceScheduledGame>> SCHEDULE_TYPE =
            new ParameterizedTypeReference<>() {
            };

    private final TeamRepository teamRepository;
    private final RestClient inferenceClient;

    public ScheduleService(
            TeamRepository teamRepository,
            RestClient.Builder builder,
            @Value("${inference.service.url}") String inferenceServiceUrl) {
        this.teamRepository = teamRepository;
        this.inferenceClient = builder.baseUrl(inferenceServiceUrl).build();
    }

    /**
     * Upcoming fixtures with team names attached.
     *
     * All 30 teams are loaded once and joined in memory. Looking each side
     * up individually would be two queries per fixture, and a couple of
     * hundred fixtures is a normal result here.
     */
    public List<ScheduledGameDto> getSchedule(int daysAhead) {
        List<InferenceScheduledGame> fixtures = fetchSchedule(daysAhead);

        if (fixtures.isEmpty()) {
            return List.of();
        }

        Map<Long, Team> teamsById = teamRepository.findAll().stream()
                .collect(Collectors.toMap(Team::getId, Function.identity()));

        return fixtures.stream()
                .filter(fixture -> teamsById.containsKey(fixture.homeTeamId())
                        && teamsById.containsKey(fixture.awayTeamId()))
                .map(fixture -> toDto(fixture, teamsById))
                .toList();
    }

    public HealthDto getHealth() {
        try {
            InferenceHealth health = inferenceClient.get()
                    .uri("/health")
                    .retrieve()
                    .body(InferenceHealth.class);
            return HealthDto.from(health);
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    private List<InferenceScheduledGame> fetchSchedule(int daysAhead) {
        try {
            List<InferenceScheduledGame> fixtures = inferenceClient.get()
                    .uri(builder -> builder.path("/schedule")
                            .queryParam("days_ahead", daysAhead)
                            .build())
                    .retrieve()
                    .body(SCHEDULE_TYPE);
            return fixtures == null ? List.of() : fixtures;
        } catch (RestClientResponseException error) {
            // A 502 from there means the NBA API failed, not the caller.
            HttpStatus status = error.getStatusCode().is4xxClientError()
                    ? HttpStatus.BAD_REQUEST
                    : HttpStatus.BAD_GATEWAY;
            throw new ResponseStatusException(
                    status, "Schedule lookup failed: " + error.getResponseBodyAsString());
        } catch (RestClientException error) {
            throw unreachable(error);
        }
    }

    /**
     * A fixture whose team ids are not both in the table is skipped rather
     * than failing the whole request: an expansion team or a re-seeded
     * table should cost one row, not the entire browse view.
     */
    private ScheduledGameDto toDto(InferenceScheduledGame fixture, Map<Long, Team> teamsById) {
        Team home = teamsById.get(fixture.homeTeamId());
        Team away = teamsById.get(fixture.awayTeamId());

        return new ScheduledGameDto(
                home.getId(),
                home.getAbbreviation(),
                home.getName(),
                away.getId(),
                away.getAbbreviation(),
                away.getName(),
                fixture.gameDate());
    }

    private ResponseStatusException unreachable(RestClientException error) {
        return new ResponseStatusException(
                HttpStatus.SERVICE_UNAVAILABLE,
                "Inference service unreachable: " + error.getMessage());
    }
}
