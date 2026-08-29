package com.andreisichet.basketball_predictor.dto;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Wire shape of POST /predict/player-props on the Python service.
 *
 * Nested by team, then player. availabilityKnown and availabilityNote sit
 * at the TEAM level because that is the grain they actually describe: one
 * injury report covers a roster, not a person.
 *
 * Each player's five predictions arrive as a map keyed by target name
 * (PTS, REB, AST, FG3M, PRA), matching what the Python side produces. It is
 * unpacked into named columns on the way into the database, where a map
 * would be the wrong shape.
 */
public record InferencePlayerPropsResponse(
        @JsonProperty("data_as_of") LocalDate dataAsOf,
        boolean stale,
        @JsonProperty("days_behind") int daysBehind,
        List<TeamBoard> teams) {

    public record TeamBoard(
            @JsonProperty("team_id") Long teamId,
            @JsonProperty("is_home") boolean isHome,
            @JsonProperty("availability_known") boolean availabilityKnown,
            @JsonProperty("availability_note") String availabilityNote,
            List<PlayerLine> players) {
    }

    public record PlayerLine(
            @JsonProperty("player_id") Long playerId,
            @JsonProperty("player_name") String playerName,
            @JsonProperty("model_used") String modelUsed,
            Map<String, Double> predictions) {

        /**
         * One target's value.
         *
         * Throws on an unknown key for the same reason the market lookup
         * does: a silently-defaulted 0.0 would be persisted as if it were a
         * real prediction.
         */
        public double value(String target) {
            Double found = predictions.get(target);
            if (found == null) {
                throw new IllegalStateException(
                        "Inference service returned no " + target + " for player "
                                + playerId + ". Its response shape has changed.");
            }
            return found;
        }
    }

    /** The board for one side, by whether it is the home team. */
    public TeamBoard board(boolean home) {
        return teams.stream()
                .filter(entry -> entry.isHome() == home)
                .findFirst()
                .orElseThrow(() -> new IllegalStateException(
                        "Inference service returned no " + (home ? "home" : "away")
                                + " team board. Its response shape has changed."));
    }
}
