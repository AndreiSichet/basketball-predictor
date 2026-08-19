package com.andreisichet.basketball_predictor.repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Team;

public interface GameRepository extends JpaRepository<Game, Long> {

    /** Upcoming games, soonest first. */
    List<Game> findByPlayedFalseOrderByGameDateAsc();

    /** Used to reuse an existing game instead of creating a duplicate. */
    Optional<Game> findByHomeTeamAndAwayTeamAndGameDate(Team homeTeam, Team awayTeam, LocalDate gameDate);
}
