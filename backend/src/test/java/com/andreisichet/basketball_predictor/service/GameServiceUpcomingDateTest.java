package com.andreisichet.basketball_predictor.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.time.LocalDate;
import java.util.List;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.TestPropertySource;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * "Upcoming" must mean today or later, not merely "not marked played".
 *
 * WHY THIS IS A TEST AND NOT A ONE-OFF CHECK. Nothing in this system ever
 * sets played = true. That was a harmless footnote while the games table
 * held only fixtures somebody had asked about, and it became a real defect
 * the moment ScheduleSyncService started caching hundreds of them: every
 * cached fixture ages into the past and, without a date bound, stays in the
 * "upcoming" list permanently. The list would grow monotonically and never
 * shed anything.
 *
 * The scenario asserted here is exactly that one - a real fixture whose
 * date has passed but which was never marked played. It is the case a
 * manual check would confirm once and then stop watching.
 */
@SpringBootTest
@TestPropertySource(properties = {
        // 29 February in a non-leap year: a valid cron that never fires.
        // The schedule sync must not run during this test and quietly
        // insert hundreds of fixtures alongside the fixture under test.
        "schedule.sync.cron=0 0 0 29 2 ?",
})
class GameServiceUpcomingDateTest {

    @Autowired
    private GameService gameService;

    @Autowired
    private GameRepository gameRepository;

    @Autowired
    private TeamRepository teamRepository;

    private Game inserted;

    @AfterEach
    void removeInsertedGame() {
        // This test writes to the same database the application uses, so
        // it cleans up after itself rather than leaving a permanent
        // past-dated row behind for the next reader to puzzle over.
        if (inserted != null) {
            gameRepository.delete(inserted);
            inserted = null;
        }
    }

    @Test
    void aPastFixtureThatWasNeverMarkedPlayedIsNotUpcoming() {
        List<Team> teams = teamRepository.findAll();
        assertThat(teams)
                .as("the team table must be seeded for this test to mean anything")
                .hasSizeGreaterThanOrEqualTo(2);

        LocalDate yesterday = LocalDate.now().minusDays(1);

        inserted = new Game();
        inserted.setHomeTeam(teams.get(0));
        inserted.setAwayTeam(teams.get(1));
        inserted.setGameDate(yesterday);
        inserted.setPlayed(false);            // the whole point: never flipped
        inserted = gameRepository.save(inserted);

        List<GameSummaryDto> upcoming = gameService.getUpcomingGames();

        assertThat(upcoming)
                .as("a fixture dated %s with played=false must not be reported "
                        + "as upcoming", yesterday)
                .noneMatch(game -> game.id().equals(inserted.getId()));

        assertThat(upcoming)
                .as("nothing dated before today may appear at all")
                .allMatch(game -> !game.gameDate().isBefore(LocalDate.now()));
    }

    @Test
    void aFixtureLaterTodayIsStillUpcoming() {
        // The boundary is inclusive: a game played tonight has not happened
        // yet, and excluding it would be a different bug in the other
        // direction. Worth pinning, because "greater than" and "greater
        // than or equal" are one keyword apart in the method name.
        List<Team> teams = teamRepository.findAll();

        inserted = new Game();
        inserted.setHomeTeam(teams.get(0));
        inserted.setAwayTeam(teams.get(1));
        inserted.setGameDate(LocalDate.now());
        inserted.setPlayed(false);
        inserted = gameRepository.save(inserted);

        List<GameSummaryDto> upcoming = gameService.getUpcomingGames();

        assertThat(upcoming)
                .as("a fixture dated today must still count as upcoming")
                .anyMatch(game -> game.id().equals(inserted.getId()));
    }
}
