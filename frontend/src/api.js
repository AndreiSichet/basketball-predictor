/**
 * Thin fetch wrapper around the Spring Boot API.
 *
 * No client library yet — three calls do not justify one.
 */

const API_BASE = 'http://localhost:8080/api';

/**
 * Turn a non-2xx response into a thrown Error carrying the backend's own
 * message. Spring puts the real reason in `message` only because
 * spring.web.error.include-message=always is set; without reading it here
 * that setting would buy the frontend nothing.
 */
async function readError(response) {
  let message = null;

  try {
    const body = await response.json();
    message = body.message || body.error || null;
  } catch {
    // A body that isn't JSON (or is empty) leaves the status as the only
    // thing worth reporting.
  }

  return new Error(message || `Request failed with status ${response.status}`);
}

async function request(path, options) {
  const response = await fetch(`${API_BASE}${path}`, options);

  if (!response.ok) {
    throw await readError(response);
  }

  return response.json();
}

export function getUpcomingGames() {
  return request('/games/upcoming');
}

export function getTeams() {
  return request('/teams');
}

/**
 * Real upcoming NBA fixtures. These are candidates to display, not
 * promises: most are further out than the inference service will predict.
 */
export function getSchedule(daysAhead = 14) {
  return request(`/games/schedule?daysAhead=${daysAhead}`);
}

/**
 * Dataset freshness. dataAsOf is what tells the browse view which
 * fixtures are actually within reach.
 */
export function getHealth() {
  return request('/health');
}

export function createPrediction(payload) {
  return request('/predictions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}
