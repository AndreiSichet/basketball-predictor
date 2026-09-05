package com.andreisichet.basketball_predictor.dto;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.json.JsonTest;

import tools.jackson.core.type.TypeReference;
import tools.jackson.databind.ObjectMapper;

/**
 * Pins the wire shape of every inference-service response the backend
 * deserialises.
 *
 * WHY THIS EXISTS. The two sides drifted once and nothing caught it:
 * models_loaded changed on the Python side from an integer to a per-family
 * object while InferenceHealth still declared an int. The backend compiled,
 * started, and served all three prediction endpoints correctly - the only
 * symptom was GET /api/health failing at runtime, which took the browse
 * view down with it. A compiler cannot see across an HTTP boundary, so the
 * shape has to be asserted somewhere.
 *
 * FIVE RECORDS DESERIALISE PYTHON RESPONSES, and all five are covered here.
 * They are the types InferenceClient passes to .body(...), which is the
 * authoritative list - the dto package also holds the backend's own
 * frontend-facing shapes, which cannot be affected by Python-side drift.
 * InferenceRequest is deliberately absent: it is serialised outbound and
 * never read back from a response.
 *
 * ASSERTIONS REACH INTO THE NESTING ON PURPOSE. A top-level object
 * deserialising while a nested list arrives empty, or a nested field
 * arrives null, is the quietest possible drift and the shape a shallow test
 * would pass straight over. So the player-props case asserts a named
 * player's actual predicted values three levels down, not merely that the
 * board is non-null.
 *
 * ---------------------------------------------------------------------
 * WHAT THESE TESTS DO NOT CATCH, as a stated decision rather than an
 * accident. Measured, not assumed - a throwaway probe was run against both
 * candidate mappers before this was written:
 *
 *     unknown property  -> ACCEPTED, silently ignored
 *     wrong type        -> REJECTED, MismatchedInputException
 *
 * So a field Python ADDS cannot fail these tests, while a field it REMOVES
 * or RETYPES will. That is the right way round: an added field is a
 * backwards-compatible change the backend should tolerate, and the
 * incidents worth catching are the two destructive ones. If a future change
 * makes tolerating new fields unsafe, this comment is the place that
 * decision was recorded.
 *
 * THE MAPPER IS THE ONE PRODUCTION USES. @JsonTest supplies Spring Boot's
 * auto-configured mapper (a tools.jackson.databind.json.JsonMapper), the
 * same instance RestClient serialises through in InferenceClient. An
 * earlier version of this suite used a bare `new ObjectMapper()`, which is
 * a different object testing a different configuration. The probe found
 * they happen to agree on both behaviours above today - so that was a
 * latent gap rather than a live bug - but a mapper configured by
 * spring.jackson.* properties would diverge, and a test that validates the
 * wrong mapper cannot be relied on to say so. It is a JSON slice, so no
 * database is involved and the whole class runs in milliseconds.
 */
@JsonTest
class InferenceWireShapeTest {

    @Autowired
    private ObjectMapper mapper;

    /** All fixtures come from this matchup, the one used throughout the project. */
    private static final LocalDate DATA_AS_OF = LocalDate.of(2026, 4, 12);
    private static final int DAYS_BEHIND = 146;

    @Nested
    class Health {

        @Test
        void deserialisesThePerFamilyModelBreakdown() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            assertThat(health.status()).isEqualTo("ok");
            assertThat(health.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(health.daysBehind()).isEqualTo(DAYS_BEHIND);
            assertThat(health.stale()).isTrue();

            // The field that actually broke: an object, not a count.
            assertThat(health.modelsLoaded()).containsExactlyInAnyOrderEntriesOf(
                    Map.of("team", 7, "quarter_half", 6, "player_props", 10));
        }

        @Test
        void carriesTheBreakdownThroughToTheClientDto() {
            InferenceHealth health =
                    mapper.readValue(Fixture.read("health.json"), InferenceHealth.class);

            HealthDto dto = HealthDto.from(health);

            // Passed through rather than flattened or dropped - the
            // per-family split is the diagnostic the endpoint exists for.
            assertThat(dto.modelsLoaded()).isEqualTo(health.modelsLoaded());
            assertThat(dto.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(dto.stale()).isTrue();
        }
    }

    @Nested
    class Schedule {

        @Test
        void deserialisesTheListAndEveryFieldInAnElement() {
            List<InferenceScheduledGame> games = mapper.readValue(
                    Fixture.read("schedule.json"), new TypeReference<>() {
                    });

            // A list, not an envelope object - the one endpoint shaped this
            // way, which is why InferenceClient needs a
            // ParameterizedTypeReference for it and Class<T> for the rest.
            assertThat(games).hasSize(14);

            InferenceScheduledGame first = games.get(0);
            assertThat(first.homeTeamId()).isEqualTo(1610612765L);
            assertThat(first.awayTeamId()).isEqualTo(1610612738L);
            assertThat(first.gameDate()).isEqualTo(LocalDate.of(2026, 10, 20));

            // Every element parses, not just the one asserted above: a
            // null id here is how a renamed field would present.
            assertThat(games).allSatisfy(game -> {
                assertThat(game.homeTeamId()).isNotNull();
                assertThat(game.awayTeamId()).isNotNull();
                assertThat(game.gameDate()).isNotNull();
            });
        }
    }

    @Nested
    class FullGame {

        @Test
        void deserialisesAllSevenPredictionsInsideTheNestedObject() {
            InferenceResponse response =
                    mapper.readValue(Fixture.read("predict.json"), InferenceResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(response.stale()).isTrue();
            assertThat(response.daysBehind()).isEqualTo(DAYS_BEHIND);

            // Into the nesting: seven values one level down, each named
            // separately on the Python side.
            InferenceResponse.Predictions p = response.predictions();
            assertThat(p).isNotNull();
            assertThat(p.homeWinProbability()).isEqualTo(0.42541608214378357);
            assertThat(p.homeMargin()).isEqualTo(1.4149187803268433);
            assertThat(p.totalPoints()).isEqualTo(232.93231201171875);
            assertThat(p.reboundMargin()).isEqualTo(1.5239256620407104);
            assertThat(p.totalRebounds()).isEqualTo(89.01725769042969);
            assertThat(p.assistMargin()).isEqualTo(1.986781120300293);
            assertThat(p.totalAssists()).isEqualTo(50.692169189453125);
        }

        @Test
        void homeWinProbabilityStillMatchesTheRecordedReferenceValue() {
            InferenceResponse response =
                    mapper.readValue(Fixture.read("predict.json"), InferenceResponse.class);

            // CLAUDE.md section 22 carries this number as the canonical
            // 38-feature reference for ATL vs BOS on 2026-04-13. It is
            // asserted here so the fixture cannot quietly be replaced with
            // a capture from a differently-trained model.
            assertThat(response.predictions().homeWinProbability())
                    .isEqualTo(0.42541608214378357, within(1e-15));
        }
    }

    @Nested
    class QuarterHalf {

        @Test
        void deserialisesSixMarketsAsAListNotSixNamedFields() {
            InferenceQuarterHalfResponse response = mapper.readValue(
                    Fixture.read("predict-quarter-half.json"),
                    InferenceQuarterHalfResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(response.daysBehind()).isEqualTo(DAYS_BEHIND);
            assertThat(response.predictions()).hasSize(6);

            assertThat(response.predictions())
                    .extracting(InferenceQuarterHalfResponse.Market::market)
                    .containsExactly(
                            "q1_home_margin", "q1_total_points",
                            "half1_home_margin", "half1_total_points",
                            "q1_home_win_probability", "half1_home_win_probability");
        }

        @Test
        void carriesTheQualifiersThatMakeAWinnerMarketReadable() {
            InferenceQuarterHalfResponse response = mapper.readValue(
                    Fixture.read("predict-quarter-half.json"),
                    InferenceQuarterHalfResponse.class);

            // q1_winner is the weakest of the six and ships labelled rather
            // than hidden, so its confidence must survive the wire.
            InferenceQuarterHalfResponse.Market q1Winner =
                    response.market("q1_home_win_probability");
            assertThat(q1Winner.value()).isEqualTo(0.4469873035961624);
            assertThat(q1Winner.confidence()).isEqualTo("low");
            assertThat(q1Winner.interpretation()).isEqualTo("P(home leads | not tied)");

            // And a regression market, where interpretation is NULL rather
            // than an empty string - a client tests for presence, so the
            // difference is load-bearing.
            InferenceQuarterHalfResponse.Market q1Margin = response.market("q1_home_margin");
            assertThat(q1Margin.value()).isEqualTo(-1.124112908717173);
            assertThat(q1Margin.confidence()).isEqualTo("medium");
            assertThat(q1Margin.interpretation()).isNull();
        }
    }

    @Nested
    class PlayerProps {

        @Test
        void deserialisesThreeLevelsDownToANamedPlayersValues() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props.json"),
                    InferencePlayerPropsResponse.class);

            assertThat(response.dataAsOf()).isEqualTo(DATA_AS_OF);
            assertThat(response.teams()).hasSize(2);

            // Level two: the board, with the two availability fields that
            // sit at TEAM grain because one injury report covers a roster.
            InferencePlayerPropsResponse.TeamBoard home = response.board(true);
            assertThat(home.teamId()).isEqualTo(1610612737L);
            assertThat(home.isHome()).isTrue();
            assertThat(home.availabilityKnown()).isFalse();
            assertThat(home.availabilityNote()).contains("AVAILABILITY UNKNOWN");
            assertThat(home.players()).hasSize(10);

            // Level three: one named player's actual numbers. This is the
            // assertion a shallow test would omit, and the drift it would
            // then miss.
            InferencePlayerPropsResponse.PlayerLine first = home.players().get(0);
            assertThat(first.playerId()).isEqualTo(1629638L);
            assertThat(first.playerName()).isEqualTo("Nickeil Alexander-Walker");
            assertThat(first.modelUsed()).isEqualTo("linear");
            assertThat(first.value("PTS")).isEqualTo(22.477595340281667);
            assertThat(first.value("REB")).isEqualTo(3.388430796609306);
            assertThat(first.value("AST")).isEqualTo(3.711617752945157);
            assertThat(first.value("FG3M")).isEqualTo(3.535938657494989);
            assertThat(first.value("PRA")).isEqualTo(29.577643889836146);
        }

        @Test
        void bothBoardsArriveAndTheAwaySideIsNotAHomeCopy() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props.json"),
                    InferencePlayerPropsResponse.class);

            InferencePlayerPropsResponse.TeamBoard away = response.board(false);
            assertThat(away.teamId()).isEqualTo(1610612738L);
            assertThat(away.isHome()).isFalse();
            assertThat(away.players()).hasSize(10);
            assertThat(away.players().get(0).playerName()).isEqualTo("Jaylen Brown");
            assertThat(away.players().get(0).value("PTS")).isEqualTo(28.10797718167896);

            // is_home is what board() routes on, so a boolean that failed to
            // bind would silently return the wrong roster rather than fail.
            assertThat(response.teams())
                    .extracting(InferencePlayerPropsResponse.TeamBoard::isHome)
                    .containsExactlyInAnyOrder(true, false);
        }

        @Test
        void everyPlayerLineIsFullyPopulated() {
            InferencePlayerPropsResponse response = mapper.readValue(
                    Fixture.read("predict-player-props.json"),
                    InferencePlayerPropsResponse.class);

            // Twenty lines, five targets each. A renamed key inside the
            // predictions map presents as a missing target rather than a
            // parse failure, so value() has to be exercised on all of them.
            assertThat(response.teams()).allSatisfy(board ->
                    assertThat(board.players()).allSatisfy(player -> {
                        assertThat(player.playerId()).isNotNull();
                        assertThat(player.playerName()).isNotBlank();
                        assertThat(player.modelUsed()).isNotBlank();
                        for (String target : List.of("PTS", "REB", "AST", "FG3M", "PRA")) {
                            assertThat(player.value(target)).isNotNaN();
                        }
                    }));
        }
    }
}
