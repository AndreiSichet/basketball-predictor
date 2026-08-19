package com.andreisichet.basketball_predictor.dto;

import com.andreisichet.basketball_predictor.model.Team;

public record TeamDto(Long id, String name, String abbreviation) {

    public static TeamDto from(Team team) {
        return new TeamDto(team.getId(), team.getName(), team.getAbbreviation());
    }
}
