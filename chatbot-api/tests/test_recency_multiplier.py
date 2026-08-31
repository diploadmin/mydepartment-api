"""Unit tests for optional recency boost multiplier."""

import unittest
from datetime import date, timedelta

from app.ai.ai_services.retrievers.recency import (
    parse_publish_date,
    recency_multiplier,
)


class TestParsePublishDate(unittest.TestCase):
    def test_iso_date_string(self):
        self.assertEqual(parse_publish_date("2024-06-15"), date(2024, 6, 15))

    def test_iso_datetime_z(self):
        self.assertEqual(parse_publish_date("2024-06-15T10:00:00Z"), date(2024, 6, 15))

    def test_empty_returns_none(self):
        self.assertIsNone(parse_publish_date(None))
        self.assertIsNone(parse_publish_date(""))

    def test_date_object(self):
        d = date(2020, 1, 1)
        self.assertEqual(parse_publish_date(d), d)


class TestRecencyMultiplier(unittest.TestCase):
    ref = date(2026, 6, 2)

    def test_today_gets_max_boost(self):
        mult = recency_multiplier(
            self.ref,
            max_age_days=1825,
            half_life_days=900,
            max_boost=0.35,
            reference_date=self.ref,
        )
        self.assertAlmostEqual(mult, 1.35, places=2)

    def test_six_years_old_neutral(self):
        old = self.ref - timedelta(days=6 * 365)
        mult = recency_multiplier(
            old,
            max_age_days=1825,
            half_life_days=900,
            max_boost=0.35,
            reference_date=self.ref,
        )
        self.assertEqual(mult, 1.0)

    def test_three_years_within_window(self):
        d = self.ref - timedelta(days=3 * 365)
        mult = recency_multiplier(
            d,
            max_age_days=1825,
            half_life_days=900,
            max_boost=0.35,
            reference_date=self.ref,
        )
        self.assertGreater(mult, 1.05)
        self.assertLess(mult, 1.35)

    def test_missing_date_neutral(self):
        self.assertEqual(
            recency_multiplier(
                None,
                max_age_days=1825,
                half_life_days=900,
                max_boost=0.35,
                reference_date=self.ref,
            ),
            1.0,
        )

    def test_never_below_one(self):
        mult = recency_multiplier(
            self.ref - timedelta(days=100),
            max_age_days=1825,
            half_life_days=900,
            max_boost=0.35,
            reference_date=self.ref,
        )
        self.assertGreaterEqual(mult, 1.0)


if __name__ == "__main__":
    unittest.main()
