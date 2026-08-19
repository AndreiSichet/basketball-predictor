package com.andreisichet.basketball_predictor.repository;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Team;

public interface TeamRepository extends JpaRepository<Team, Long> {
}
