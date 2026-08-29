package com.andreisichet.basketball_predictor.model;

import java.time.Instant;

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
 * One player's five prop predictions for one game.
 *
 * One row per player per call, so a full board is up to 20 rows across both
 * sides. team records which side he was on for THIS call - a player can be
 * traded, and a historical row should still say who he was playing for.
 *
 * modelUsed IS stored, unlike the quarter/half confidence labels, and the
 * difference is real: which of the hybrid's two models answered varies per
 * player and per request, because it depends on whether that player had a
 * complete rolling window at that moment. It is a fact about this row, not
 * a constant about the market.
 *
 * WHAT IS NOT STORED: availabilityKnown and its caveat text. Those describe
 * a whole team's roster for one request, not a player, so writing them here
 * would repeat the same value across ten rows and invite the two copies to
 * disagree. They are assembled at the response layer.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PlayerPropPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "player_id")
    private Player player;

    /** The side this player was on for this prediction. */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "team_id")
    private Team team;

    private double predictedPoints;

    private double predictedRebounds;

    private double predictedAssists;

    private double predictedThreesMade;

    /** Points + rebounds + assists, the common combined market. */
    private double predictedPra;

    /** "linear" or "xgb" - which half of the hybrid answered. */
    private String modelUsed;

    private Instant predictedAt;
}
