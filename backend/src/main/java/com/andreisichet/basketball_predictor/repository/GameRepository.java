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
     * Upcoming games, soonest first, with both teams already loaded.
     *
     * The entity graph is what makes this one query instead of thirty-one.
     * Game's team relations are LAZY, so mapping each game to its
     * abbreviation initialised a proxy per distinct team - bounded at 30
     * rather than growing with the games, which is exactly why it survived
     * unnoticed behind the far louder per-game prediction lookup. Fetching
     * them alongside the games costs nothing extra and removes the round
     * trips entirely.
     */
    @EntityGraph(attributePaths = {"homeTeam", "awayTeam"})
    List<Game> findByPlayedFalseOrderByGameDateAsc();

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
