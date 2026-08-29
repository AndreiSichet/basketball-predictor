package com.andreisichet.basketball_predictor.service;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.andreisichet.basketball_predictor.dto.InferencePlayerPropsResponse;
import com.andreisichet.basketball_predictor.dto.InferenceRequest;
import com.andreisichet.basketball_predictor.dto.PlayerPropsResponseDto;
import com.andreisichet.basketball_predictor.dto.PredictionRequest;
import com.andreisichet.basketball_predictor.model.Game;
import com.andreisichet.basketball_predictor.model.Player;
import com.andreisichet.basketball_predictor.model.PlayerPropPrediction;
import com.andreisichet.basketball_predictor.model.Team;
import com.andreisichet.basketball_predictor.repository.PlayerPropPredictionRepository;
import com.andreisichet.basketball_predictor.repository.PlayerRepository;

/**
 * Player prop markets, both sides of a fixture in one call.
 *
 * Both rosters together because that is the real unit of use - a prop board
 * is a game's worth of players, not a team's - and because the two sides
 * share the validation, the freshness block and the Game row.
 *
 * PLAYERS ARE CREATED ON DEMAND. Unlike the fixed 30 teams, which are
 * seeded at startup, the player universe is thousands strong and mostly
 * irrelevant to any one request. A Player row appears the first time a
 * prediction actually returns that player and is reused afterwards, so the
 * table grows to exactly what has been asked for.
 */
@Service
public class PlayerPropPredictionService {

    private static final String POINTS = "PTS";
    private static final String REBOUNDS = "REB";
    private static final String ASSISTS = "AST";
    private static final String THREES = "FG3M";
    private static final String PRA = "PRA";

    private final GameLookup gameLookup;
    private final PlayerRepository playerRepository;
    private final PlayerPropPredictionRepository predictionRepository;
    private final InferenceClient inferenceClient;

    public PlayerPropPredictionService(
            GameLookup gameLookup,
            PlayerRepository playerRepository,
            PlayerPropPredictionRepository predictionRepository,
            InferenceClient inferenceClient) {
        this.gameLookup = gameLookup;
        this.playerRepository = playerRepository;
        this.predictionRepository = predictionRepository;
        this.inferenceClient = inferenceClient;
    }

    /**
     * Same discipline as the other two: inference first, writes only after.
     * A rejected request must leave no Game, no Player and no prediction
     * rows behind.
     */
    @Transactional
    public PlayerPropsResponseDto predict(PredictionRequest request) {
        Team homeTeam = gameLookup.requireTeam(request.homeTeamId());
        Team awayTeam = gameLookup.requireTeam(request.awayTeamId());

        InferencePlayerPropsResponse inference = inferenceClient.predictPlayerProps(
                new InferenceRequest(
                        request.homeTeamId(), request.awayTeamId(), request.gameDate()));

        Game game = gameLookup.findOrCreateGame(homeTeam, awayTeam, request.gameDate());
        Instant predictedAt = Instant.now();

        PlayerPropsResponseDto.TeamBoard home =
                persistBoard(game, homeTeam, inference.board(true), predictedAt);
        PlayerPropsResponseDto.TeamBoard away =
                persistBoard(game, awayTeam, inference.board(false), predictedAt);

        return new PlayerPropsResponseDto(
                game.getId(),
                game.getGameDate(),
                home,
                away,
                inference.dataAsOf(),
                inference.stale(),
                inference.daysBehind(),
                predictedAt);
    }

    /**
     * Persist one side's board and shape it for the response.
     *
     * availabilityKnown and its note travel to the DTO but are NOT written
     * to any row: they describe the roster for this request, so storing
     * them per player would repeat one value across ten rows and let the
     * copies drift apart.
     */
    private PlayerPropsResponseDto.TeamBoard persistBoard(
            Game game,
            Team team,
            InferencePlayerPropsResponse.TeamBoard board,
            Instant predictedAt) {

        List<PlayerPropsResponseDto.PlayerLine> lines = new ArrayList<>();
        for (InferencePlayerPropsResponse.PlayerLine line : board.players()) {
            Player player = findOrCreatePlayer(line);
            PlayerPropPrediction saved =
                    predictionRepository.save(toEntity(game, player, team, line, predictedAt));
            lines.add(PlayerPropsResponseDto.PlayerLine.from(saved));
        }

        return new PlayerPropsResponseDto.TeamBoard(
                team.getId(),
                team.getAbbreviation(),
                board.availabilityKnown(),
                board.availabilityNote(),
                lines);
    }

    /**
     * The Player row for this id, created if this is the first time it has
     * been seen. The name is taken from the inference payload, which reads
     * it from the same box-score history the models were trained on.
     */
    private Player findOrCreatePlayer(InferencePlayerPropsResponse.PlayerLine line) {
        return playerRepository.findById(line.playerId())
                .orElseGet(() -> {
                    Player player = new Player();
                    player.setId(line.playerId());
                    player.setName(line.playerName());
                    return playerRepository.save(player);
                });
    }

    private PlayerPropPrediction toEntity(
            Game game,
            Player player,
            Team team,
            InferencePlayerPropsResponse.PlayerLine line,
            Instant predictedAt) {

        PlayerPropPrediction prediction = new PlayerPropPrediction();
        prediction.setGame(game);
        prediction.setPlayer(player);
        prediction.setTeam(team);
        prediction.setPredictedPoints(line.value(POINTS));
        prediction.setPredictedRebounds(line.value(REBOUNDS));
        prediction.setPredictedAssists(line.value(ASSISTS));
        prediction.setPredictedThreesMade(line.value(THREES));
        prediction.setPredictedPra(line.value(PRA));
        prediction.setModelUsed(line.modelUsed());
        prediction.setPredictedAt(predictedAt);
        return prediction;
    }
}
