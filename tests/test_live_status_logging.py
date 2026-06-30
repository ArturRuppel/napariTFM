"""Live/interactive stage progress should echo to the console like batch does.

Worklist §1 (unify logging): batch mode redirects stdout through its TeeLogger
and prints timestamped progress lines, while live mode reported progress only
through the Qt status label. ``napariTFMWidget._relay_stage_status`` is the single
funnel every interactive stage's ``progress_updated`` signal passes through, so it
now mirrors each message to stdout in batch's ``[timestamp] message`` format while
still driving the UI label.

These tests exercise ``_relay_stage_status`` as an unbound method against a minimal
stub, so they need neither a napari viewer nor the full widget tree.
"""

import re

from napariTFM.widgets._widget import napariTFMWidget


class _StatusLabelStub:
    def __init__(self):
        self.text = None

    def setText(self, text):
        self.text = text


class _WidgetStub:
    """Just enough of ``napariTFMWidget`` for ``_relay_stage_status``."""

    def __init__(self):
        self.status_label = _StatusLabelStub()
        # No stage sections wired up — _relay_stage_status's spine-progress
        # forwarding must be a no-op against an empty mapping, not require one.
        self._stage_sections_by_key = {}


# A leading "[YYYY-MM-DD HH:MM:SS] " stamp matching TeeLogger's format.
_STAMP = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] ")


def test_relay_status_updates_label_and_echoes_to_console(capsys):
    stub = _WidgetStub()

    napariTFMWidget._relay_stage_status(stub, "Preprocessing", "Processing beads: Frame 1/20")

    # UI label is unchanged in behaviour: "{stage} — {message}".
    assert stub.status_label.text == "Preprocessing — Processing beads: Frame 1/20"

    out = capsys.readouterr().out.strip()
    # Console line carries batch's timestamp prefix...
    assert _STAMP.match(out)
    # ...and the same stage-labelled message shown in the UI.
    assert out.endswith("Preprocessing — Processing beads: Frame 1/20")


def test_relay_status_echoes_each_stage(capsys):
    stub = _WidgetStub()

    for stage, message in (
        ("Displacement", "Computing optical flow"),
        ("Force", "FTTC frame 3/10"),
        ("Stress", "Solving monolayer stress"),
    ):
        napariTFMWidget._relay_stage_status(stub, stage, message)

    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 3
    for line, (stage, message) in zip(
        lines,
        (("Displacement", "Computing optical flow"),
         ("Force", "FTTC frame 3/10"),
         ("Stress", "Solving monolayer stress")),
    ):
        assert _STAMP.match(line)
        assert line.endswith(f"{stage} — {message}")
