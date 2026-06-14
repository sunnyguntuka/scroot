"""Analytics router - /api/analytics endpoints."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Query


def analytics_router(store):
    router = APIRouter()

    @router.get("")
    def summary(time_range: str = Query("30d", alias="range")):
        """Unified analytics endpoint - returns all charts in one call."""
        records = store.get_all()
        hours = {"24h": 24, "7d": 168, "30d": 720}.get(time_range, 720)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        total = len(records)
        iqs_vals = [r.scores.get("iqs", 0) for r in records if isinstance(r.scores, dict)]
        avg_iqs = round(sum(iqs_vals) / len(iqs_vals), 3) if iqs_vals else 0.0

        pending_review = sum(1 for r in records if getattr(r, "status", "pending") == "pending")

        # IQS trend - daily buckets
        from collections import defaultdict
        daily: dict[str, list[float]] = defaultdict(list)
        for r in records:
            try:
                dt = datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if dt >= cutoff:
                key = dt.strftime("%Y-%m-%d")
                daily[key].append(r.scores.get("iqs", 0) if isinstance(r.scores, dict) else 0)
        iqs_trend = [
            {"date": d, "avg_iqs": round(sum(v) / len(v), 3)}
            for d, v in sorted(daily.items()) if v
        ]

        # Flag frequency - object keyed by metric name
        flag_counter: Counter = Counter()
        for r in records:
            for f in (r.flags or []):
                flag_counter[f] += 1
        flag_metrics = ["groundedness", "completeness", "relevance", "consistency", "confidence"]
        flag_frequency = {m: flag_counter.get(m, flag_counter.get(f"low_{m}", 0)) for m in flag_metrics}

        # IQS distribution - 5 buckets
        buckets = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
        dist = Counter()
        for v in iqs_vals:
            idx = min(4, int(v * 5))
            dist[idx] += 1
        iqs_distribution = [{"bucket": buckets[i], "count": dist.get(i, 0)} for i in range(5)]

        # Per-agent breakdown
        agent_map: dict[str, list[float]] = defaultdict(list)
        for r in records:
            aid = r.corrected_by or "unknown"
            iqs = r.scores.get("iqs", 0) if isinstance(r.scores, dict) else 0
            agent_map[aid].append(iqs)
        per_agent = sorted(
            [
                {"agent_id": aid, "avg_iqs": round(sum(v) / len(v), 3), "count": len(v)}
                for aid, v in agent_map.items()
            ],
            key=lambda x: x["avg_iqs"],
        )

        # Avg IQS today
        today = datetime.now(timezone.utc).date().isoformat()
        today_vals = [
            r.scores.get("iqs", 0) for r in records
            if isinstance(r.scores, dict) and r.timestamp[:10] == today
        ]
        avg_iqs_today = round(sum(today_vals) / len(today_vals), 3) if today_vals else avg_iqs

        threshold = 0.70
        pass_count = sum(1 for v in iqs_vals if v >= threshold)
        warn_count = sum(1 for v in iqs_vals if threshold * 0.7 <= v < threshold)
        fail_count = sum(1 for v in iqs_vals if v < threshold * 0.7)

        return {
            "total_scored": total,
            "avg_iqs": avg_iqs,
            "avg_iqs_today": avg_iqs_today,
            "avg_iqs_delta": 0.0,
            "pending_review": pending_review,
            "pass_count": pass_count,
            "warn_count": warn_count,
            "fail_count": fail_count,
            "iqs_trend": iqs_trend,
            "flag_frequency": flag_frequency,
            "iqs_distribution": iqs_distribution,
            "per_agent": per_agent,
        }

    @router.get("/iqs-trend")
    def iqs_trend(time_range: str = Query("7d", alias="range"), agent: Optional[str] = Query(None)):
        records = store.get_all()
        if not records:
            return {"points": []}

        # Determine bucket size
        hours = {"24h": 24, "7d": 168, "30d": 720}.get(time_range, 168)
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours)

        # Bucket records by hour
        hour_buckets: dict[str, list[float]] = defaultdict(list)
        for r in records:
            try:
                dt = datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if dt < start:
                continue
            if time_range in ("24h", "7d"):
                key = dt.strftime("%Y-%m-%dT%H:00:00Z")
            else:
                key = dt.strftime("%Y-%m-%dT00:00:00Z")
            iqs = r.scores.get("iqs", 0) if isinstance(r.scores, dict) else 0
            hour_buckets[key].append(iqs)

        points = []
        for ts in sorted(hour_buckets):
            vals = hour_buckets[ts]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            p10 = vals_sorted[max(0, int(n * 0.1) - 1)]
            p90 = vals_sorted[min(n - 1, int(n * 0.9))]
            points.append({
                "timestamp": ts,
                "mean_iqs": round(sum(vals) / len(vals), 4),
                "p10": round(p10, 4),
                "p90": round(p90, 4),
                "flagged_count": sum(1 for r in records
                                     if r.timestamp[:len(ts)] >= ts[:10]
                                     and r.flags),
            })

        return {"points": points}

    @router.get("/flag-distribution")
    def flag_distribution(time_range: str = Query("7d", alias="range")):
        records = store.get_all()
        hours = {"24h": 24, "7d": 168, "30d": 720}.get(time_range, 168)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        flag_counter: Counter = Counter()
        total = 0
        for r in records:
            try:
                dt = datetime.fromisoformat(r.timestamp.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue
            if dt >= cutoff:
                total += 1
                for f in (r.flags or []):
                    flag_counter[f] += 1

        if total == 0:
            return {"flags": []}

        flag_types = ["hallucination_risk", "incomplete", "off_topic", "self_contradictory", "ungrounded"]
        return {
            "flags": [
                {
                    "type": ft,
                    "count": flag_counter.get(ft, 0),
                    "pct": round(flag_counter.get(ft, 0) / total * 100, 1) if total else 0,
                    "trend_pct": 0.0,  # TODO: compare to previous period
                }
                for ft in flag_types
            ]
        }

    @router.get("/before-after")
    def before_after(correction_id: Optional[str] = Query(None)):
        records = store.get_all()
        reviewed = [r for r in records if getattr(r, "status", "pending") in ("reviewed", "applied")]

        def histogram(vals, bins=10):
            if not vals:
                return []
            step = 1.0 / bins
            counts = [0] * bins
            for v in vals:
                idx = min(int(v * bins), bins - 1)
                counts[idx] += 1
            return [{"x": round(i * step, 1), "y": c} for i, c in enumerate(counts)]

        before_iqs = [r.scores.get("iqs", 0) for r in records if isinstance(r.scores, dict)]
        after_iqs = [r.corrected_response_iqs for r in reviewed
                     if getattr(r, "corrected_response_iqs", None) is not None]

        before_mean = sum(before_iqs) / len(before_iqs) if before_iqs else 0
        after_mean = sum(after_iqs) / len(after_iqs) if after_iqs else 0

        return {
            "before": {"histogram": histogram(before_iqs), "mean": round(before_mean, 3)},
            "after":  {"histogram": histogram(after_iqs),  "mean": round(after_mean, 3)},
            "delta":  round(after_mean - before_mean, 3),
        }

    @router.get("/reviewer-throughput")
    def reviewer_throughput():
        records = store.get_all()
        reviewed = [r for r in records if getattr(r, "status", "pending") in ("reviewed", "applied", "rejected")]

        # Count by day for last 7 days
        today = datetime.now(timezone.utc).date()
        by_day = []
        for i in range(6, -1, -1):
            day = (today - timedelta(days=i)).isoformat()
            count = sum(1 for r in reviewed if r.timestamp[:10] == day)
            by_day.append(count)

        today_count = by_day[-1]
        week_total = sum(by_day)
        avg_per_day = round(week_total / 7, 1)

        return {
            "reviews_today": today_count,
            "avg_time_per_review_s": 0.0,  # requires session tracking
            "reviews_this_week": by_day,
            "week_total": week_total,
            "avg_per_day": avg_per_day,
        }

    return router
