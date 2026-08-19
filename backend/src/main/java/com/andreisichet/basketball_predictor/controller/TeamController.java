package com.andreisichet.basketball_predictor.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.TeamDto;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

/**
 * Calls the repository directly. There is no business logic to put in a
 * service, and Team has no lazy relations, so nothing needs a transaction.
 */
@RestController
@RequestMapping("/api/teams")
public class TeamController {

    private final TeamRepository teamRepository;

    public TeamController(TeamRepository teamRepository) {
        this.teamRepository = teamRepository;
    }

    @GetMapping
    public List<TeamDto> all() {
        return teamRepository.findAll().stream()
                .map(TeamDto::from)
                .toList();
    }
}
