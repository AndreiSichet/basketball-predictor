package com.andreisichet.basketball_predictor.repository;

import java.util.Collection;
import java.util.List;
import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.andreisichet.basketball_predictor.model.Prediction;

public interface PredictionRepository extends JpaRepository<Prediction, Long> {

    List<Prediction> findByGameId(Long gameId);

    /** Most recent prediction for a game, since a game accumulates them. */
    Optional<Prediction> findTopByGameIdOrderByPredictedAtDesc(Long gameId);

    /**
     * Every prediction for a batch of games, newest first.
     *
     * NOT findTopByGameIdIn, and the difference matters. Spring Data's
     * `Top`/`First` keywords limit the WHOLE result set, not each group -
     * findTopByGameIdIn(ids) would compile, run, and return exactly one
     * Prediction for the entire batch, silently giving 449 of 450 games a
     * null latest prediction. There is no derived-query spelling of
     * "newest row per group"; it needs either a window function in native
     * SQL or a fold in the caller.
     *
     * The caller folds. Ordering by predictedAt descending means the first
     * row seen for a game id is its newest, so the grouping keeps that one
     * and discards the rest - which also makes a tie on predictedAt
     * resolve deterministically rather than returning two rows for a game.
     *
     * Volume is not a concern despite fetching every prediction rather than
     * one per game: the schedule sync caches hundreds of fixtures, and the
     * overwhelming majority have never been predicted at all, so they
     * contribute no rows here.
     */
    List<Prediction> findByGameIdInOrderByPredictedAtDesc(Collection<Long> gameIds);
}
