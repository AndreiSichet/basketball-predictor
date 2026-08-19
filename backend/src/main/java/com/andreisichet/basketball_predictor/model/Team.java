package com.andreisichet.basketball_predictor.model;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * An NBA team.
 *
 * The id is the real NBA team id from the pipeline data (e.g. 1610612737 for
 * Atlanta), not generated. It is already stable and unique, and keeping it
 * means rows line up with the CSVs without a lookup table.
 */
@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Team {

    @Id
    private Long id;

    private String name;

    private String abbreviation;
}
