package com.andreisichet.basketball_predictor.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.QuarterHalfPrediction;

public interface QuarterHalfPredictionRepository
        extends JpaRepository<QuarterHalfPrediction, Long> {
}
