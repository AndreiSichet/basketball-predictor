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
 * One set of Q1 / first-half model outputs for one game.
 *
 * Reuses the existing Game entity rather than introducing a parallel one:
 * the same fixture is the same fixture whichever markets are being priced,
 * so all three prediction types hang off one games table.
 *
 * WHAT IS DELIBERATELY NOT STORED: the per-market confidence label and the
 * "P(home leads | not tied)" interpretation. Those are static facts about
 * which model produced a number, identical on every row this table will
 * ever hold - q1_winner is always low-confidence because of how it scored,
 * not because of anything about this particular request. Persisting them
 * would duplicate a constant once per prediction. They are attached at
 * response-build time from the inference payload instead, the same way
 * nothing about how the original seven models were selected is stored per
 * row today.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class QuarterHalfPrediction {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "game_id")
    private Game game;

    private double q1Spread;

    private double q1Total;

    /** P(home leads Q1 | Q1 is not tied). See the class comment. */
    private double q1WinnerProbability;

    private double half1Spread;

    private double half1Total;

    /** P(home leads at half | the half is not tied). */
    private double half1WinnerProbability;

    private LocalDate dataAsOf;

    private boolean stale;

    private Instant predictedAt;
}
