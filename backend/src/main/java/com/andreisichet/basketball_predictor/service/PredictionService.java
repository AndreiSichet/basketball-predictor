package com.andreisichet.basketball_predictor.service;

import java.time.Instant;
import java.time.LocalDate;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import com.andreisichet.basketball_predictor.dto.GameSummaryDto;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.InferenceResponse;
import com.andreisichet.basketball_predictor.dto.PredictionDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Prediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.GameRepository;
import com.andreisichet.basketball_predictor.repository.PredictionRepository;
import com.andreisichet.basketball_predictor.repository.TeamRepository;

@Service
public class PredictionService {

    private final TeamRepository teamRepository;
    private final GameRepository gameRepository;
    private final PredictionRepository predictionRepository;
    private final RestClient inferenceClient;

    public PredictionService(
            TeamRepository teamRepository,
            GameRepository gameRepository,
            PredictionRepository predictionRepository,
            RestClient.Builder builder,
            @Value("${inference.service.url}") String inferenceServiceUrl) {
        this.teamRepository = teamRepository;
        this.gameRepository = gameRepository;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = builder.baseUrl(inferenceServiceUrl).build();
    }

    /**
     * Transactional so a rejected request writes nothing. The inference call
     * also runs before any insert, so the usual failure never reaches the
     * database at all and rollback is only a backstop.
     *
     * Returns a DTO, not the entity: with open-in-view off, serializing an
     * entity outside the session fails on its LAZY relations.
     */
    @Transactional
    public GameSummaryDto predict(PredictionRequest request) {
        Team homeTeam = requireTeam(request.homeTeamId());
        Team awayTeam = requireTeam(request.awayTeamId());

        InferenceResponse inference = callInferenceService(request);

        Game game = findOrCreateGame(homeTeam, awayTeam, request.gameDate());
        Prediction prediction = predictionRepository.save(toPrediction(game, inference));

        return new GameSummaryDto(
                game.getId(),
                homeTeam.getAbbreviation(),
                awayTeam.getAbbreviation(),
                game.getGameDate(),
                game.isPlayed(),
                PredictionDto.from(prediction));
    }

    private Team requireTeam(Long teamId) {
        if (teamId == null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Team id is required.");
        }
        return teamRepository.findById(teamId)
                .orElseThrow(() -> new ResponseStatusException(
                        HttpStatus.BAD_REQUEST, "Unknown team id: " + teamId));
    }

    private Game findOrCreateGame(Team homeTeam, Team awayTeam, LocalDate gameDate) {
        return gameRepository.findByHomeTeamAndAwayTeamAndGameDate(homeTeam, awayTeam, gameDate)
                .orElseGet(() -> {
                    Game game = new Game();
                    game.setHomeTeam(homeTeam);
                    game.setAwayTeam(awayTeam);
                    game.setGameDate(gameDate);
                    game.setPlayed(false);
                    return gameRepository.save(game);
                });
    }

    /**
     * Call the Python service, turning its failures into sensible statuses.
     *
     * A 4xx from there means bad input (stale data, unknown team, date too
     * far ahead), so it stays a 4xx here instead of becoming a 500. An
     * unreachable service is a 503, since the request itself was fine.
     */
    private InferenceResponse callInferenceService(PredictionRequest request) {
        InferenceRequest body = new InferenceRequest(
                request.homeTeamId(), request.awayTeamId(), request.gameDate());

        try {
            return inferenceClient.post()
                    .uri("/predict")
                    .body(body)
                    .retrieve()
                    .body(InferenceResponse.class);
        } catch (RestClientResponseException error) {
            HttpStatus status = error.getStatusCode().is4xxClientError()
                    ? HttpStatus.BAD_REQUEST
                    : HttpStatus.BAD_GATEWAY;
            throw new ResponseStatusException(
                    status, "Inference service rejected the request: " + error.getResponseBodyAsString());
        } catch (RestClientException error) {
            throw new ResponseStatusException(
                    HttpStatus.SERVICE_UNAVAILABLE, "Inference service unreachable: " + error.getMessage());
        }
    }

    private Prediction toPrediction(Game game, InferenceResponse inference) {
        InferenceResponse.Predictions values = inference.predictions();

        Prediction prediction = new Prediction();
        prediction.setGame(game);
        prediction.setHomeWinProbability(values.homeWinProbability());
        prediction.setHomeMargin(values.homeMargin());
        prediction.setTotalPoints(values.totalPoints());
        prediction.setReboundMargin(values.reboundMargin());
        prediction.setTotalRebounds(values.totalRebounds());
        prediction.setAssistMargin(values.assistMargin());
        prediction.setTotalAssists(values.totalAssists());
        prediction.setDataAsOf(inference.dataAsOf());
        prediction.setStale(inference.stale());
        prediction.setPredictedAt(Instant.now());
        return prediction;
    }
}
