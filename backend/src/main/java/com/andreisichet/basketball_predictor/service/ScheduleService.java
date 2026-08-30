package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.dto.InferenceHealth;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * Serves upcoming fixtures and dataset freshness.
 *
 * TWO SOURCES NOW, DELIBERATELY DIFFERENT, and that split is the point of
 * this class:
 *
 *   Fixtures come from the DATABASE, cached by ScheduleSyncService every
 *   six hours. They describe a season calendar, which changes about as
 *   often as one would expect a calendar to. Fetching them live on every
 *   click meant a multi-second third-party call per button press.
 *
 *   Freshness comes LIVE from the inference service on every request, and
 *   must keep doing so. data_as_of tracks the pipeline's data, so caching
 *   it alongside the fixtures would mean the client computing which games
 *   are predictable from a stale cutoff - and it would go wrong at exactly
 *   the moment it matters most, the first request after a pipeline rerun.
 *
 * The external contract is unchanged: same route, same query parameter,
 * same response shape. Only where the fixtures come from moved.
 */
@Service
public class ScheduleService {

    private final GameRepository gameRepository;
    private final TeamRepository teamRepository;
    private final RestClient inferenceClient;

    public ScheduleService(
            GameRepository gameRepository,
            TeamRepository teamRepository,
            RestClient.Builder builder,
            @org.springframework.beans.factory.annotation.Value("${inference.service.url}")
            String inferenceServiceUrl) {
        this.gameRepository = gameRepository;
        this.teamRepository = teamRepository;
        this.inferenceClient = builder.baseUrl(inferenceServiceUrl).build();
    }

    /**
     * Cached fixtures inside the requested window, with team names attached.
     *
     * Bounded in the query, not in memory: the sync job caches months of
     * fixtures, so filtering a findAll() would load every one of them to
     * answer a 14-day question.
     *
     * Transactional because Game's team relations are LAZY and open-in-view
     * is off. All 30 teams are still loaded once and joined from a map
     * rather than resolved per fixture, which is what keeps this one query
     * for the games and one for the teams regardless of how many fixtures
     * come back.
     *
     * NOTE: this can only return what the sync job has cached. A window
     * wider than schedule.sync.days-ahead returns fewer fixtures than the
     * old live call would have.
     */
    @Transactional(readOnly = true)
    public List<ScheduledGameDto> getSchedule(int daysAhead) {
        LocalDate today = LocalDate.now();
        List<Game> games = gameRepository
                .findByPlayedFalseAndGameDateBetweenOrderByGameDateAsc(
                        today, today.plusDays(daysAhead));

        if (games.isEmpty()) {
            return List.of();
        }

        Map<Long, Team> teamsById = teamRepository.findAll().stream()
                .collect(Collectors.toMap(Team::getId, Function.identity()));

        return games.stream()
                .map(game -> toDto(game, teamsById))
                .filter(dto -> dto != null)
                .toList();
    }

    /**
     * Live on every call, never cached. See the class comment.
     */
    public HealthDto getHealth() {
        try {
            InferenceHealth health = inferenceClient.get()
                    .uri("/health")
                    .retrieve()
                    .body(InferenceHealth.class);
            return HealthDto.from(health);
        } catch (RestClientException error) {
            throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE,
                    "Inference service unreachable: " + error.getMessage());
        }
    }

    /**
     * A fixture whose team ids are not both in the table is skipped rather
     * than failing the whole request: an expansion team or a re-seeded
     * table should cost one row, not the entire browse view.
     */
    private ScheduledGameDto toDto(Game game, Map<Long, Team> teamsById) {
        Team home = teamsById.get(game.getHomeTeam().getId());
        Team away = teamsById.get(game.getAwayTeam().getId());

        if (home == null || away == null) {
            return null;
        }

        return new ScheduledGameDto(
                home.getId(),
                home.getAbbreviation(),
                home.getName(),
                away.getId(),
                away.getAbbreviation(),
                away.getName(),
                game.getGameDate());
    }
}
