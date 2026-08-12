"""Static guards for the "Follow Live" scheduler in the web dashboard.

The Follow Live scheduler must track loop progress independently of
which detail view (Loop detail vs. agent chat) the user is currently
in. It does that by subscribing to the wrapper SSE bus directly for
the lifetime of ``followMode``. Two behavioural invariants:

1. Empty-window state shows "following (idle)" — i.e. when the
   followed child has already terminated but the next ``spawn_child``
   has not yet landed, the pill must downgrade rather than vanish.

2. While follow is active, sidebar selection clicks are inert — the
   user has to disable Follow Live first. This deletes the older
   "follow (stuck)" / "Resume Follow" recovery path entirely.

These tests guard the markup/script tokens that encode those rules
so a future refactor cannot silently regress them.
"""

from __future__ import annotations

import unittest
from pathlib import Path

MONOREPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_HTML = MONOREPO_ROOT / "web" / "static" / "index.html"
APP_JS = MONOREPO_ROOT / "web" / "static" / "js" / "app.js"


class TestFollowLiveScheduler(unittest.TestCase):
    def setUp(self) -> None:
        self.index = INDEX_HTML.read_text(encoding="utf-8")
        self.app = APP_JS.read_text(encoding="utf-8")

    # ---- dead-symbol removal ----------------------------------------

    def test_app_js_drops_stuck_state_and_resume_path(self) -> None:
        """``followStuck`` + ``resumeFollow`` belonged to the old
        "user click breaks follow, then re-arm via Resume button"
        flow. With clicks now inert while follow is on, both must
        be gone from ``app.js``."""
        for token in (
            "followStuck",
            "resumeFollow",
            "_watchFollowChild",
            "_followChildSse",
            "_closeFollowChildSse",
        ):
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    self.app,
                    f"app.js still references dead follow symbol '{token}'",
                )

    def test_index_html_drops_resume_button_and_stuck_label(self) -> None:
        for token in (
            "Resume Follow",
            "follow (stuck)",
            "followStuck",
            "resumeFollow",
        ):
            with self.subTest(token=token):
                self.assertNotIn(
                    token,
                    self.index,
                    f"index.html still references dead follow symbol '{token}'",
                )

    def test_loop_sse_handler_no_longer_owns_follow_scheduling(self) -> None:
        """``_connectLoopSse`` used to walk the messages tail looking
        for ``spawn_child`` events to drive ``_followSelectAgent``.
        That scheduling now lives in the dedicated follow-bus
        subscription, which runs whether or not the Loop detail pane
        is open. The Loop-detail SSE handler must not also call into
        ``_followSelectAgent`` (double-fire) or the per-view
        ``_loopLogMessageCount`` counter (race with the dedicated
        scheduler's counter)."""
        # Anchor on the function definition (4-space indent + open
        # brace), not on its call site.
        idx = self.app.find("\n    _connectLoopSse(loopId) {\n")
        self.assertGreater(idx, 0, "_connectLoopSse definition must exist")
        end_marker = self.app.find("\n    // ===", idx + 1)
        if end_marker == -1:
            end_marker = idx + 8000
        body = self.app[idx:end_marker]
        self.assertNotIn("_followSelectAgent", body)
        self.assertNotIn("spawn_child", body)

    # ---- new behaviour ----------------------------------------------

    def test_app_js_has_dedicated_follow_bus_subscription(self) -> None:
        """The scheduler subscribes to ``WrapperBus`` directly so it
        keeps running when the user navigates from the Loop detail
        pane into a child-agent chat. The subscribe call must be
        emitted by the follow scheduler (not by ``_connectLoopSse``
        / ``_connectAgentLoopMiniSse``, which exist for rendering)."""
        for token in (
            "_subscribeFollowBus",
            "_unsubscribeFollowBus",
            "_followBusUnsub",
        ):
            with self.subTest(token=token):
                self.assertIn(
                    token,
                    self.app,
                    f"app.js missing follow-bus helper '{token}'",
                )

    def test_app_js_blocks_sidebar_clicks_while_following(self) -> None:
        """While ``followMode`` is on, user-initiated sidebar
        selections must be ignored (with a toast hint). The
        scheduler's own programmatic selection bypasses the guard
        via ``_followInternalSwitch``. Both sidebar entry points
        (agent + loop) must enforce the guard."""
        for fn in ("selectSidebarAgent(agentId)", "selectSidebarLoop(loopId)"):
            with self.subTest(fn=fn):
                idx = self.app.find(fn)
                self.assertGreater(idx, 0, f"{fn} missing")
                body = self.app[idx : idx + 1500]
                self.assertIn("this.followMode", body)
                self.assertIn("Follow Live", body, "expected toast text mentioning Follow Live")
        # The scheduler must keep its bypass flag — the guard
        # otherwise blocks the auto-switch as well.
        self.assertIn("_followInternalSwitch", self.app)

    def test_app_js_exposes_follow_idle_state(self) -> None:
        """``followIsIdle()`` derives idleness from the followed
        child's live state in ``managedAgents`` — that is the
        signal the pill uses to render "following (idle)" during
        the gap between one child exiting and the next ``spawn_child``."""
        self.assertIn("followIsIdle", self.app)
        self.assertIn("followCurrentAgentId", self.app)

    def test_index_html_pill_renders_idle_variant(self) -> None:
        """The top-right pill must render ``following`` /
        ``following (idle)`` based on ``followIsIdle()``, replacing
        the old ``follow`` / ``follow (stuck)`` text."""
        self.assertIn("following (idle)", self.index)
        self.assertIn("followIsIdle", self.index)

    def test_sidebar_agent_preserves_engagement_on_internal_switch(self) -> None:
        """When ``_followInternalSwitch`` is set and the agent is not yet
        in ``managedAgents`` (freshly spawned, list hasn't polled), the
        else branch must NOT call ``_disengageLoop`` which would kill
        Follow Live. Instead it should preserve the engagement using
        ``engagedLoopId``."""
        idx = self.app.find("selectSidebarAgent(agentId) {")
        self.assertGreater(idx, 0, "selectSidebarAgent must exist")
        body = self.app[idx : idx + 2500]
        self.assertIn("_followInternalSwitch", body)
        self.assertIn("engagedLoopId", body)

    # ---- rotation ----------------------------------------------------

    def test_app_js_defines_rotation_interval_constant(self) -> None:
        """Rotation cadence must live in a single module-level
        constant (``FOLLOW_ROTATE_INTERVAL_MS``) so tuning the
        dwell does not require chasing literals through the
        scheduler body. Default is 10 s, matching the documented
        behavior."""
        self.assertIn("FOLLOW_ROTATE_INTERVAL_MS = 10000", self.app)

    def test_app_js_has_rotation_helpers(self) -> None:
        """The scheduler exposes start / stop / tick + pool query
        helpers. Their absence means rotation is silently disabled."""
        for token in (
            "_restartFollowRotate",
            "_stopFollowRotate",
            "_followRotateTick",
            "_followLiveChildren",
            "_followRotateTimer",
        ):
            with self.subTest(token=token):
                self.assertIn(
                    token,
                    self.app,
                    f"app.js missing rotation helper '{token}'",
                )

    def test_rotation_pool_is_live_children_only(self) -> None:
        """``_followLiveChildren`` must filter ``managedAgents`` to
        the engaged loop's running children, excluding the wrapper.
        Without that filter the rotation could land on the wrapper
        itself or on terminated children."""
        idx = self.app.find("_followLiveChildren(loopId) {")
        self.assertGreater(idx, 0, "_followLiveChildren must exist")
        body = self.app[idx : idx + 600]
        self.assertIn("loop_id === loopId", body)
        self.assertIn("kind !== 'loop_wrapper'", body)
        self.assertIn("state === 'running'", body)

    def test_subscribe_starts_and_unsubscribe_stops_rotation(self) -> None:
        """Rotation lifetime must be bound to the follow-bus
        subscription so it dies with follow mode and not before
        (e.g. when the user navigates between detail views)."""
        sub = self.app.find("_subscribeFollowBus(loopId) {")
        unsub = self.app.find("_unsubscribeFollowBus() {")
        self.assertGreater(sub, 0)
        self.assertGreater(unsub, 0)
        sub_body = self.app[sub : sub + 1000]
        unsub_body = self.app[unsub : unsub + 500]
        self.assertIn("_restartFollowRotate(loopId)", sub_body)
        self.assertIn("_stopFollowRotate()", unsub_body)

    def test_select_agent_resets_rotation_interval(self) -> None:
        """A spawn-driven (or initial /active-driven) switch must
        reset the rotation timer; otherwise a child landing 9 s
        into the interval would be visible for only ~1 s before
        the tick rotated away."""
        idx = self.app.find("_followSelectAgent(agentId) {")
        self.assertGreater(idx, 0)
        body = self.app[idx : idx + 800]
        self.assertIn("_restartFollowRotate", body)

    def test_rotation_tick_skips_when_pool_lt_two(self) -> None:
        """Single-child (stage1) loops must not flap focus on
        every tick — the tick must early-return when the live
        pool has zero or one element."""
        idx = self.app.find("_followRotateTick(loopId) {")
        self.assertGreater(idx, 0)
        body = self.app[idx : idx + 600]
        self.assertIn("pool.length < 2", body)


if __name__ == "__main__":
    unittest.main()
