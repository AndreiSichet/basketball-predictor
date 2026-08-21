package com.andreisichet.basketball_predictor.controller;

import java.util.List;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.ScheduledGameDto;
import com.andreisichet.basketball_predictor.service.GameService;
import com.andreisichet.basketball_predictor.service.ScheduleService;

@RestController
@RequestMapping("/api/games")
public class GameController {

    private final GameService gameService;
    private final ScheduleService scheduleService;

    public GameController(GameService gameService, ScheduleService scheduleService) {
        this.gameService = gameService;
        this.scheduleService = scheduleService;
    }

    @GetMapping("/upcoming")
    public List<GameSummaryDto> upcoming() {
        return gameService.getUpcomingGames();
    }

    /**
     * Real NBA fixtures, straight from the schedule. These are candidates
     * to predict, not stored games - most of them are further out than the
     * inference service will accept.
     */
    @GetMapping("/schedule")
    public List<ScheduledGameDto> schedule(
            @RequestParam(defaultValue = "14") int daysAhead) {
        return scheduleService.getSchedule(daysAhead);
    }
}
