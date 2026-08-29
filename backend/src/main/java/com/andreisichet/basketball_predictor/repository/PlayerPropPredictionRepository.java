package com.andreisichet.basketball_predictor.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.PlayerPropPrediction;

public interface PlayerPropPredictionRepository
        extends JpaRepository<PlayerPropPrediction, Long> {
}
