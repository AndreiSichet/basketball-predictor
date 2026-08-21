import { addDays, latestPredictableDate } from './dates';

test('adds days within a month', () => {
  expect(addDays('2026-04-12', 1)).toBe('2026-04-13');
});

test('crosses a month boundary', () => {
  expect(addDays('2026-04-30', 1)).toBe('2026-05-01');
});

test('crosses a year boundary', () => {
  expect(addDays('2026-12-31', 1)).toBe('2027-01-01');
});

test('the cutoff is one day past the data', () => {
  expect(latestPredictableDate('2026-04-12')).toBe('2026-04-13');
});

test('no data means no cutoff', () => {
  expect(latestPredictableDate(null)).toBeNull();
});
