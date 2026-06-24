"""Labeled-slider parameter controls, ported from CellFlow's UI vocabulary."""
from __future__ import annotations

from qtpy.QtCore import QEvent, QObject, QSize, Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QSizePolicy,
    QStyle,
    QStyleOption,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider, QLabeledSlider

from napariTFM.widgets._ui_style import mono_font


def _patch_label_autosize(label) -> None:
    """Size the editable slider label to fit its widest formatted value."""
    def _get_size():
        dec = label.decimals() if hasattr(label, "decimals") else 0

        def _fmt(v):
            return f"{v:.{dec}f}" if dec else f"{int(v)}"

        lo, hi = label.minimum(), label.maximum()
        sample = max((_fmt(lo), _fmt(hi)), key=len)
        if not sample.startswith("-"):
            sample = "-" + sample
        fm = label.fontMetrics()
        prefix = label.prefix() or ""
        suffix = label.suffix() or ""
        w = fm.horizontalAdvance(prefix + sample + suffix) + 18
        h = label.sizeHint().height()
        opt = QStyleOption()
        return label.style().sizeFromContents(
            QStyle.ContentsType.CT_LineEdit, opt, QSize(w, h), label
        )

    label._get_size = _get_size
    label._update_size()


def _slider_step_button(text: str, object_name: str, tooltip: str) -> QToolButton:
    button = QToolButton()
    button.setText(text)
    button.setObjectName(object_name)
    button.setToolTip(tooltip)
    button.setAutoRepeat(True)
    button.setFixedSize(18, 18)
    button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return button


class _StepButtonStateSyncer(QObject):
    def __init__(self, sync) -> None:
        super().__init__()
        self._sync = sync

    def eventFilter(self, _watched, event) -> bool:
        if event.type() == QEvent.Type.EnabledChange:
            self._sync()
        return False


def _connect_step_buttons(slider):
    decrement = _slider_step_button("-", "slider_decrement_button", "Decrease by one step")
    increment = _slider_step_button("+", "slider_increment_button", "Increase by one step")

    def _step(direction: int) -> None:
        if not slider.isEnabled():
            return
        slider.setValue(slider.value() + direction * slider.singleStep())

    def _sync(*_a) -> None:
        enabled = slider.isEnabled()
        decrement.setEnabled(enabled and slider.value() > slider.minimum())
        increment.setEnabled(enabled and slider.value() < slider.maximum())

    decrement.clicked.connect(lambda: _step(-1))
    increment.clicked.connect(lambda: _step(1))
    slider.valueChanged.connect(_sync)
    slider.rangeChanged.connect(_sync)
    syncer = _StepButtonStateSyncer(_sync)
    syncer.setParent(slider)
    slider.installEventFilter(syncer)
    slider._step_button_state_syncer = syncer
    _sync()
    return decrement, increment


def _stack_label_above(slider, *, step_buttons: bool) -> None:
    label = slider._label
    label.setFont(mono_font())
    track = slider._slider
    label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
    label.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
    _patch_label_autosize(label)

    old_layout = slider.layout()
    if old_layout is not None:
        old_layout.removeWidget(label)
        old_layout.removeWidget(track)
        QWidget().setLayout(old_layout)
    label.setParent(slider)
    track.setParent(slider)
    vbox = QVBoxLayout()
    vbox.setContentsMargins(0, 0, 0, 0)
    vbox.setSpacing(0)
    vbox.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
    if step_buttons:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        dec, inc = _connect_step_buttons(slider)
        row.addWidget(dec, alignment=Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(track)
        row.addWidget(inc, alignment=Qt.AlignmentFlag.AlignVCenter)
        vbox.addLayout(row)
    else:
        vbox.addWidget(track)
    slider.setLayout(vbox)


def dslider(lo, hi, val, step=0.1, decimals=2, tooltip="", *, step_buttons=True):
    s = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _stack_label_above(s, step_buttons=step_buttons)
    return s


def islider(lo, hi, val, step=1, tooltip="", *, step_buttons=True):
    s = QLabeledSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setToolTip(tooltip)
    s.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    _stack_label_above(s, step_buttons=step_buttons)
    return s
