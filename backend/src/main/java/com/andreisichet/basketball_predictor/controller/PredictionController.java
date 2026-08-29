package com.andreisichet.basketball_predictor.controller;

import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.PlayerPropsResponseDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.dto.QuarterHalfSummaryDto;
import com.andreisichet.basketball_predictor.service.PlayerPropPredictionService;
import com.andreisichet.basketball_predictor.service.PredictionService;
import com.andreisichet.basketball_predictor.service.QuarterHalfPredictionService;

/**
 * Three sibling prediction endpoints, mirroring the inference service's own
 * split rather than folding everything into one.
 *
 * The request body is identical for all three, but the responses are not:
 * seven plain numbers, six markets carrying confidence and
 * conditional-probability caveats, and two nested rosters of five
 * predictions each. One endpoint returning a union of those would be mostly
 * nulls whichever way it was called.
 */
@RestController
@RequestMapping("/api/predictions")
public class PredictionController {

    private final PredictionService predictionService;
    private final QuarterHalfPredictionService quarterHalfPredictionService;
    private final PlayerPropPredictionService playerPropPredictionService;

    public PredictionController(
            PredictionService predictionService,
            QuarterHalfPredictionService quarterHalfPredictionService,
            PlayerPropPredictionService playerPropPredictionService) {
        this.predictionService = predictionService;
        this.quarterHalfPredictionService = quarterHalfPredictionService;
        this.playerPropPredictionService = playerPropPredictionService;
    }

    @PostMapping
    public GameSummaryDto create(@RequestBody PredictionRequest request) {
        return predictionService.predict(request);
    }

    @PostMapping("/quarter-half")
    public QuarterHalfSummaryDto createQuarterHalf(@RequestBody PredictionRequest request) {
        return quarterHalfPredictionService.predict(request);
    }

    @PostMapping("/player-props")
    public PlayerPropsResponseDto createPlayerProps(@RequestBody PredictionRequest request) {
        return playerPropPredictionService.predict(request);
    }
}
