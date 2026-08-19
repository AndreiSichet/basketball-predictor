package com.andreisichet.basketball_predictor.model;

import java.time.Instant;
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
 * One set of model outputs for one game.
 *
 * The id is generated here, unlike Team and Game: this is new data the
 * system produces, with no external identifier to reuse.
 *
 * The seven prediction fields match the inference service response exactly.
 * dataAsOf and stale are its freshness metadata, stored rather than
 * discarded so an old prediction can be recognised as one later.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Prediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    private double homeWinProbability;

    private double homeMargin;

    private double totalPoints;

    private double reboundMargin;

    private double totalRebounds;

    private double assistMargin;

    private double totalAssists;

    private LocalDate dataAsOf;

    private boolean stale;

    private Instant predictedAt;
}
