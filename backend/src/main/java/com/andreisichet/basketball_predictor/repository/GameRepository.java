package com.andreisichet.basketball_predictor.repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.EntityGraph;
import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;

public interface GameRepository extends JpaRepository<Game, Long> {

    /**
     * Genuinely upcoming games, soonest first, with both teams loaded.
     *
     * THE DATE BOUND IS NOT COSMETIC. "played = false" alone means "nobody
     * has marked this finished", which is not the same thing as "still to
     * come" - nothing in this system ever sets played = true. That was a
     * footnote while the table held only fixtures somebody had asked about;
     * once ScheduleSyncService began caching hundreds of them, every one
     * would age into the past and stay in this list forever.
     *
     * This is the cheap half of the fix: it stops past fixtures being
     * reported as upcoming. It does NOT set played, which needs the
     * pipeline to know a game finished and is tracked with the October
     * rerun work.
     *
     * The entity graph is what makes this one query instead of thirty-one.
     * Game's team relations are LAZY, so mapping each game to its
     * abbreviation initialised a proxy per distinct team - bounded at 30
     * rather than growing with the games, which is exactly why it survived
     * unnoticed behind the far louder per-game prediction lookup. It is
     * carried over from the previous version of this method deliberately;
     * dropping it here would silently restore that N+1.
     */
    @EntityGraph(attributePaths = {"homeTeam", "awayTeam"})
    List<Game> findByPlayedFalseAndGameDateGreaterThanEqualOrderByGameDateAsc(
            LocalDate date);

    /**
     * Cached fixtures inside a date window, soonest first.
     *
     * Bounded in the query rather than filtered in memory: the sync job
     * caches months of fixtures, so an unbounded findAll-then-filter would
     * load every one of them to answer a 14-day question.
     *
     * `Between` is inclusive on both ends in Spring Data, which is what
     * "today through today + daysAhead" should mean.
     */
    List<Game> findByPlayedFalseAndGameDateBetweenOrderByGameDateAsc(
            LocalDate from, LocalDate to);

    /** Used to reuse an existing game instead of creating a duplicate. */
    Optional<Game> findByHomeTeamAndAwayTeamAndGameDate(Team homeTeam, Team awayTeam, LocalDate gameDate);
}
