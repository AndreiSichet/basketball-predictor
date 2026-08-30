package com.andreisichet.basketball_predictor.service;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.InferenceScheduledGame;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * Caches upcoming NBA fixtures into the games table.
 *
 * WHY THIS EXISTS. GET /api/games/schedule used to reach the live NBA API
 * on every click, through the inference service. That is a multi-second
 * call against a third party, repeated for every user who presses a button,
 * to fetch data that changes about as often as a season calendar does.
 * Fetching it on a cadence and serving it from Postgres turns a slow
 * dependency into a fast local read.
 *
 * IT REUSES GameLookup.findOrCreateGame, which is the whole reason this is
 * safe. A fixture cached here and the same fixture requested for a
 * prediction later resolve to ONE row, because both go through the same
 * find-or-create on the same key. No new duplicate-prevention logic exists,
 * and none should: a second implementation is how the two paths would
 * eventually disagree.
 *
 * IT NEVER THROWS. A cycle that cannot reach the inference service logs and
 * returns; the next one tries again. Crashing a background job because a
 * dependency was restarting would take the whole application down for a
 * transient condition, which is the same reasoning behind the injury
 * report's NoReportAvailable path degrading rather than failing.
 *
 * THE SYNC HORIZON BOUNDS WHAT THE ENDPOINT CAN RETURN. This caches
 * `schedule.sync.days-ahead` days; a request for a window wider than that
 * gets only what has been cached. Previously the endpoint asked the live
 * API for whatever was requested, so this is a real behavioural change, and
 * the default is set generously for that reason.
 */
@Service
public class ScheduleSyncService {

    private static final Logger log = LoggerFactory.getLogger(ScheduleSyncService.class);

    private final InferenceClient inferenceClient;
    private final GameLookup gameLookup;
    private final GameRepository gameRepository;
    private final TeamRepository teamRepository;
    private final int daysAhead;

    public ScheduleSyncService(
            InferenceClient inferenceClient,
            GameLookup gameLookup,
            GameRepository gameRepository,
            TeamRepository teamRepository,
            @Value("${schedule.sync.days-ahead:120}") int daysAhead) {
        this.inferenceClient = inferenceClient;
        this.gameLookup = gameLookup;
        this.gameRepository = gameRepository;
        this.teamRepository = teamRepository;
        this.daysAhead = daysAhead;
    }

    /**
     * Fetch the upcoming schedule and make sure every fixture has a row.
     *
     * Transactional so a failure part-way through does not leave the table
     * half-updated. Idempotent by construction: findOrCreateGame is a
     * no-op for a fixture already present, so running this twice in a row
     * creates nothing the second time.
     */
    @Transactional
    public void sync() {
        List<InferenceScheduledGame> fixtures;

        try {
            fixtures = inferenceClient.fetchSchedule(daysAhead);
        } catch (Exception error) {
            // Deliberately broad: an unreachable service, a 502 from the
            // NBA API behind it, a malformed body. None of them are worth
            // stopping the application for, and all of them are fixed by
            // the next cycle succeeding.
            log.warn("Schedule sync skipped - could not fetch fixtures: {}", error.getMessage());
            return;
        }

        if (fixtures.isEmpty()) {
            // A real answer in the offseason, not a failure.
            log.info("Schedule sync: 0 fixtures returned for the next {} days, nothing to cache.",
                    daysAhead);
            return;
        }

        Map<Long, Team> teamsById = teamRepository.findAll().stream()
                .collect(Collectors.toMap(Team::getId, Function.identity()));

        long before = gameRepository.count();
        int skipped = 0;

        for (InferenceScheduledGame fixture : fixtures) {
            Team home = teamsById.get(fixture.homeTeamId());
            Team away = teamsById.get(fixture.awayTeamId());

            // Same rule the endpoint already applied: an unknown team id
            // costs one fixture, never the whole cycle.
            if (home == null || away == null) {
                skipped++;
                continue;
            }

            gameLookup.findOrCreateGame(home, away, fixture.gameDate());
        }

        long created = gameRepository.count() - before;
        log.info("Schedule sync: {} fixtures fetched ({} days ahead), {} new, {} already present{}.",
                fixtures.size(),
                daysAhead,
                created,
                fixtures.size() - skipped - created,
                skipped > 0 ? ", " + skipped + " skipped (unknown team)" : "");
    }
}
