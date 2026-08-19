package com.andreisichet.basketball_predictor.repository;

import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Prediction;

public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    List<Prediction> findByGameId(Long gameId);

    /** Most recent prediction for a game, since a game accumulates them. */
    Optional<Prediction> findTopByGameIdOrderByPredictedAtDesc(Long gameId);
}
