package com.andreisichet.basketball_predictor.model;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * An NBA player.
 *
 * The id is the real nba_api PLAYER_ID, not generated - the same convention
 * as Team, and for the same reason: it is already stable and unique, so
 * rows line up with the pipeline CSVs without a lookup table.
 *
 * NOT BULK-SEEDED, unlike the 30 teams. The player universe is thousands
 * strong and mostly irrelevant to any given request, so rows are created
 * find-or-create style the first time a prediction actually returns that
 * player. Seeding it would mean shipping and maintaining a large static
 * list to support the handful of rows a single prop board touches.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Player {

    @Id
    private Long id;

    private String name;
}
