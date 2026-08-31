package com.andreisichet.basketball_predictor.service;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.HealthDto;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * Serves upcoming fixtures and dataset freshness.
 *
 * TWO SOURCES, DELIBERATELY DIFFERENT, and that split is the point of this
 * class:
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
 *   the moment it matters, the first request after a pipeline rerun.
 *
 * This class no longer holds a RestClient of its own. It asks
 * InferenceClient, like every other caller, which is what makes that
 * class's "one place that talks to the Python service" claim true rather
 * than aspirational.
 */
@Service
public class ScheduleService {

    private final GameRepository gameRepository;
    private final TeamRepository teamRepository;
    private final InferenceClient inferenceClient;

    public ScheduleService(
            GameRepository gameRepository,
            TeamRepository teamRepository,
            InferenceClient inferenceClient) {
        this.gameRepository = gameRepository;
        this.teamRepository = teamRepository;
        this.inferenceClient = inferenceClient;
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
     * rather than resolved per fixture, which keeps this one query for the
     * games and one for the teams regardless of how many come back.
     *
     * NOTE: this can only return what the sync job has cached. A window
     * wider than schedule.sync.days-ahead returns fewer fixtures than the
     * old live call would have.
     *
     * THERE IS NO LONGER A SKIP PATH HERE. This method used to drop any
     * fixture whose team ids were not both in the Team table - a real
     * concern when fixtures arrived straight off the NBA API, where
     * undetermined playoff slots carry a placeholder team id of 0. It
     * cannot happen now: a fixture only reaches this method by first
     * becoming a Game row, and game.home_team_id / away_team_id are
     * foreign keys onto team.id, so the database itself refuses a row this
     * method could not resolve. The skip still exists where it is still
     * possible - in ScheduleSyncService, before a row is created - and it
     * is counted in that job's log line rather than being silent.
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
                .toList();
    }

    /** Live on every call, never cached. See the class comment. */
    public HealthDto getHealth() {
        return HealthDto.from(inferenceClient.getHealth());
    }

    private ScheduledGameDto toDto(Game game, Map<Long, Team> teamsById) {
        Team home = teamsById.get(game.getHomeTeam().getId());
        Team away = teamsById.get(game.getAwayTeam().getId());

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
