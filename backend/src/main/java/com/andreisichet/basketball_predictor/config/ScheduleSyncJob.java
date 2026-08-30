package com.andreisichet.basketball_predictor.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import com.andreisichet.basketball_predictor.service.ScheduleSyncService;

/**
 * When the schedule cache refreshes. Nothing about HOW - that is
 * ScheduleSyncService's job.
 *
 * Split for the same reason TeamSeeder is separate from TeamRepository: a
 * trigger and the work it triggers change for different reasons, and a
 * cron expression buried inside business logic is one nobody finds.
 *
 * TWO TRIGGERS, and the second is not optional. The cron alone would leave
 * a fresh deployment - or a fresh Docker volume, which starts with an empty
 * games table - serving an empty schedule for up to six hours before the
 * first cycle fired. That is a visible regression from the old always-live
 * behaviour, so startup runs one sync immediately, IN ADDITION to the
 * cadence rather than instead of it.
 *
 * ApplicationReadyEvent rather than a constructor or @PostConstruct: the
 * context has to be fully built before this touches the database or makes
 * an HTTP call, and the inference service may still be coming up alongside
 * it. A failed startup sync is logged and skipped like any other cycle.
 *
 * REQUIRES @EnableScheduling on the application class. Without it @Scheduled
 * is silently inert - the method simply never runs, with no error anywhere,
 * which is the failure mode that makes this worth a comment.
 */
@Component
public class ScheduleSyncJob {

    private static final Logger log = LoggerFactory.getLogger(ScheduleSyncJob.class);

    private final ScheduleSyncService scheduleSyncService;

    public ScheduleSyncJob(ScheduleSyncService scheduleSyncService) {
        this.scheduleSyncService = scheduleSyncService;
    }

    /**
     * Every six hours. The NBA schedule changes rarely - postponements and
     * the occasional reschedule - so this is about staying current within a
     * day, not within a minute.
     *
     * Overridable so the cadence can be shortened for a smoke test without
     * a rebuild, and so a deployment can tune it without touching code.
     */
    @Scheduled(cron = "${schedule.sync.cron:0 0 */6 * * *}")
    public void run() {
        scheduleSyncService.sync();
    }

    @EventListener(ApplicationReadyEvent.class)
    public void syncOnStartup() {
        log.info("Running the initial schedule sync so a fresh database is not empty.");
        scheduleSyncService.sync();
    }
}
