package com.andreisichet.basketball_predictor.dto;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;

/**
 * Loads a raw inference-service response from src/test/resources/fixtures/.
 *
 * WHY THESE ARE FILES AND NOT STRING CONSTANTS. Each fixture was captured
 * byte-for-byte from a running inference service with curl, never
 * hand-written and never re-serialised through an intermediate parser. A
 * fixture typed out from the Java record's own fields would drift with that
 * record and so could never detect the drift it exists to catch - which is
 * exactly the bug InferenceHealthTest was written for.
 *
 * Read as UTF-8 explicitly. The first capture attempt used PowerShell's
 * Invoke-WebRequest, which decoded the body as Latin-1 and turned
 * "Vit Krejci" into double-encoded mojibake; curl writes the response bytes
 * unchanged. Player names are the only place non-ASCII appears, and
 * predict-player-props.json still carries some, so this matters.
 */
final class Fixture {

    private Fixture() {
    }

    static String read(String name) {
        String path = "/fixtures/" + name;
        try (InputStream stream = Fixture.class.getResourceAsStream(path)) {
            if (stream == null) {
                throw new IllegalStateException("no fixture on the classpath at " + path);
            }
            return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException error) {
            throw new UncheckedIOException("could not read " + path, error);
        }
    }
}
