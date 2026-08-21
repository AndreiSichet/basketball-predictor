/**
 * Date helpers for the browse view.
 *
 * Everything is a plain YYYY-MM-DD string. ISO dates compare correctly as
 * strings, so no Date objects are needed anywhere except to add days.
 */

/** YYYY-MM-DD, `days` after the given YYYY-MM-DD. */
export function addDays(isoDate, days) {
  // Parsed with an explicit time so the browser reads it as local rather
  // than UTC, which would shift the date west of Greenwich.
  const parsed = new Date(`${isoDate}T00:00:00`);
  parsed.setDate(parsed.getDate() + days);

  const month = String(parsed.getMonth() + 1).padStart(2, '0');
  const day = String(parsed.getDate()).padStart(2, '0');
  return `${parsed.getFullYear()}-${month}-${day}`;
}

/**
 * The furthest date the inference service will predict.
 *
 * MAX_DAYS_AHEAD is 1 and it is not a staleness workaround: REST_DAYS is
 * measured from the last game in the dataset, so a fixture with unplayed
 * games in between would have its rest measured against the wrong one.
 * That stays true no matter how fresh the data is.
 */
export const MAX_DAYS_AHEAD = 1;

export function latestPredictableDate(dataAsOf) {
  return dataAsOf ? addDays(dataAsOf, MAX_DAYS_AHEAD) : null;
}
