package com.andreisichet.basketball_predictor;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * @EnableScheduling is load-bearing and easy to lose. Without it every
 * @Scheduled method in the application is silently inert - no error, no
 * warning, the method simply never fires. ScheduleSyncJob is the only user
 * of it today; anything else added on a timer depends on it too.
 */
@SpringBootApplication
@EnableScheduling
public class BasketballPredictorApplication {

	public static void main(String[] args) {
		SpringApplication.run(BasketballPredictorApplication.class, args);
	}

}
