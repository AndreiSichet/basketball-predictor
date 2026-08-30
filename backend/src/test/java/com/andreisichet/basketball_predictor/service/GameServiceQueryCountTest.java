package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.hibernate.SessionFactory;
import org.hibernate.stat.Statistics;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;

import jakarta.persistence.EntityManagerFactory;

/**
 * Pins GET /api/games/upcoming to a constant number of queries.
 *
 * WHY A TEST AND NOT A ONE-OFF MEASUREMENT. This endpoint mapped each game
 * to its latest prediction with a lookup inside the loop. That was harmless
 * while the games table held only fixtures somebody had actually predicted,
 * and it was documented as a known trade-off for exactly that reason. Then
 * ScheduleSyncService started caching hundreds of fixtures ahead of time
 * and the same code issued 486 selects for a single call. Nothing failed;
 * it just got slow, which is the kind of regression that returns quietly
 * unless something asserts against it.
 *
 * The assertion is a small ceiling rather than an exact number. What must
 * hold is that the count does not scale with the number of games - pinning
 * it to precisely 2 would break on an unrelated change to how Hibernate
 * batches or how the entity graph is fetched, which is not what this is
 * guarding.
 *
 * Statistics rather than log scraping: SQL log output is buffered and
 * interleaved with startup noise, which makes counting "select" in a log
 * file quietly unreliable - it double-counted a request during development
 * of this very fix.
 */
@SpringBootTest
@TestPropertySource(properties = {
        "spring.jpa.properties.hibernate.generate_statistics=true",
        // Nothing here should reach the inference service; the startup sync
        // logs and skips when it cannot, which is the behaviour under test
        // everywhere else.
        "schedule.sync.cron=0 0 0 29 2 ?",
})
class GameServiceQueryCountTest {

    /**
     * Two queries is the design: one for the games, one for every latest
     * prediction in the batch. The ceiling leaves room for Hibernate's own
     * bookkeeping without leaving room for a per-game lookup to creep back.
     */
    private static final int QUERY_CEILING = 6;

    // Measured on 453 cached fixtures: 486 before the fix, 32 after the
    // batched prediction lookup alone (30 of them lazy Team proxies, which
    // this ceiling caught), and 2 once the entity graph loaded the teams
    // alongside the games.

    @Autowired
    private GameService gameService;

    @Autowired
    private EntityManagerFactory entityManagerFactory;

    @Test
    void upcomingGamesDoesNotScaleQueriesWithGameCount() {
        Statistics statistics = entityManagerFactory
                .unwrap(SessionFactory.class)
                .getStatistics();
        statistics.clear();

        List<GameSummaryDto> games = gameService.getUpcomingGames();

        long queries = statistics.getPrepareStatementCount();

        assertThat(queries)
                .as("one call returning %d games issued %d queries - it must not "
                        + "scale with the number of games", games.size(), queries)
                .isLessThanOrEqualTo(QUERY_CEILING);
    }
}
