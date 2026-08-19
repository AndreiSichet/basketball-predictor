package com.andreisichet.basketball_predictor.model;

import java.time.LocalDate;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * One game, upcoming or completed.
 *
 * The id is generated because the app creates games for matchups that have
 * no official id yet. nbaGameId holds the real GAME_ID once schedule
 * integration exists, and is null until then.
 *
 * played is false for a scheduled game and flips to true once the pipeline
 * picks up the result.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Game {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    /** Real nba_api GAME_ID, null until the game has one. */
    private Long nbaGameId;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "home_team_id")
    private Team homeTeam;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "away_team_id")
    private Team awayTeam;

    private LocalDate gameDate;

    private boolean played;
}
