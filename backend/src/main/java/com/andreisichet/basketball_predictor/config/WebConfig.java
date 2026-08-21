package com.andreisichet.basketball_predictor.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * Allows the React dev server to call the API.
 *
 * The frontend uses an absolute base URL rather than a same-origin path, so
 * every request is cross-origin and the browser blocks it without this.
 * Scoped to /api and to one configured origin; once both are served from a
 * single origin behind Docker, this can go away.
 */
@Configuration
public class WebConfig implements WebMvcConfigurer {

    private final String frontendOrigin;

    public WebConfig(@Value("${frontend.origin}") String frontendOrigin) {
        this.frontendOrigin = frontendOrigin;
    }

    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins(frontendOrigin)
                .allowedMethods("GET", "POST");
    }
}
