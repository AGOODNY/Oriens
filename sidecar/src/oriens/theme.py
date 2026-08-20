"""Oriens 彩饰手稿主题、品牌资源与自绘窗口边框。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from .paths import AppPaths


@dataclass(frozen=True, slots=True)
class ThemePalette:
    vellum: str = "#D8BF8C"
    vellum_light: str = "#E9D9B6"
    vellum_deep: str = "#C5A66B"
    umber: str = "#1B1510"
    leather: str = "#251A13"
    ink: str = "#2B2118"
    gold: str = "#C7A052"
    firefly: str = "#DCEE88"
    moss: str = "#70865B"
    ochre: str = "#B57A37"
    burgundy: str = "#7A3E35"


PALETTE = ThemePalette()
BODY_FONT_FALLBACK = '"Noto Serif SC", "Source Han Serif SC", SimSun, "Microsoft YaHei UI"'
DISPLAY_FONT_FALLBACK = '"Cinzel Decorative", "Noto Serif SC", SimSun, serif'


def _load_font(path: Path) -> str | None:
    if not path.is_file():
        return None
    font_id = QFontDatabase.addApplicationFont(str(path))
    if font_id < 0:
        return None
    families = QFontDatabase.applicationFontFamilies(font_id)
    return families[0] if families else None


def load_bundled_fonts(paths: AppPaths) -> tuple[str | None, str | None]:
    fonts = paths.ui_assets_dir / "fonts"
    body = _load_font(fonts / "NotoSerifSC-Variable.ttf")
    display = _load_font(fonts / "CinzelDecorative-Regular.ttf")
    return body, display


def application_icon(paths: AppPaths) -> QIcon:
    candidate = paths.ui_assets_dir / "oriens-app-icon.png"
    return QIcon(str(candidate)) if candidate.is_file() else QIcon()


def apply_application_theme(application: QApplication, paths: AppPaths) -> None:
    body, _display = load_bundled_fonts(paths)
    application.setFont(QFont(body or "SimSun", 10))
    icon = application_icon(paths)
    if not icon.isNull():
        application.setWindowIcon(icon)
    application.setStyleSheet(DESKTOP_STYLE)


class BookTitleBar(QFrame):
    """自绘书册标题栏；拖动交给 Windows/Qt 的系统移动实现。"""

    def __init__(self, window: QMainWindow, icon: QIcon, title: str) -> None:
        super().__init__(window)
        self._window = window
        self._press_position: QPoint | None = None
        self.setObjectName("bookTitleBar")
        self.setFixedHeight(68)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 12, 8)
        layout.setSpacing(12)
        mark = QLabel()
        mark.setObjectName("brandMark")
        mark.setFixedSize(48, 48)
        if not icon.isNull():
            mark.setPixmap(icon.pixmap(44, 44))
        else:
            mark.setText("O")
            mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(mark)

        brand = QLabel("ORIENS")
        brand.setObjectName("brandName")
        layout.addWidget(brand)
        subtitle = QLabel(title)
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(subtitle)
        layout.addStretch(1)

        self.minimize_button = QPushButton("—")
        self.minimize_button.setObjectName("chromeButton")
        self.minimize_button.setToolTip("最小化")
        self.minimize_button.clicked.connect(window.showMinimized)
        layout.addWidget(self.minimize_button)
        self.close_button = QPushButton("×")
        self.close_button.setObjectName("chromeClose")
        self.close_button.setToolTip("关闭")
        self.close_button.clicked.connect(window.close)
        layout.addWidget(self.close_button)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_position = event.globalPosition().toPoint()
            handle = self._window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            self._window.showNormal() if self._window.isMaximized() else self._window.showMaximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)


class FramelessBookWindow(QMainWindow):
    """带系统缩放命中区的无边框主窗口。"""

    _resize_margin = 8

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMouseTracking(True)

    def _resize_edges(self, position: QPoint) -> Qt.Edge:
        edges = Qt.Edge(0)
        if position.x() <= self._resize_margin:
            edges |= Qt.Edge.LeftEdge
        elif position.x() >= self.width() - self._resize_margin:
            edges |= Qt.Edge.RightEdge
        if position.y() <= self._resize_margin:
            edges |= Qt.Edge.TopEdge
        elif position.y() >= self.height() - self._resize_margin:
            edges |= Qt.Edge.BottomEdge
        return edges

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.LeftButton:
            edges = self._resize_edges(event.position().toPoint())
            handle = self.windowHandle()
            if edges and handle is not None and handle.startSystemResize(edges):
                event.accept()
                return
        super().mousePressEvent(event)

    def enable_resize_tracking(self, root: QWidget) -> None:
        for widget in (root, *root.findChildren(QWidget)):
            widget.installEventFilter(self)

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                local = self.mapFromGlobal(event.globalPosition().toPoint())
                edges = self._resize_edges(local)
                handle = self.windowHandle()
                if edges and handle is not None and handle.startSystemResize(edges):
                    event.accept()
                    return True
        return super().eventFilter(watched, event)


class StatusCard(QFrame):
    def __init__(self, title: str, icon_text: str = "•", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)
        icon = QLabel(icon_text)
        icon.setObjectName("cardGlyph")
        icon.setFixedWidth(28)
        icon.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(icon)
        text = QFrame()
        text_layout = QHBoxLayout(text)
        text_layout.setContentsMargins(0, 0, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("cardCaption")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.value.setWordWrap(True)
        text_layout.addWidget(title_label)
        text_layout.addStretch(1)
        text_layout.addWidget(self.value)
        layout.addWidget(text, 1)


DESKTOP_STYLE = f"""
QWidget {{
    color: {PALETTE.ink};
    font-family: {BODY_FONT_FALLBACK};
    font-size: 13px;
}}
QFrame#bookShell {{
    background: {PALETTE.umber};
    border: 2px solid #6f542a;
    border-radius: 10px;
}}
QFrame#bookTitleBar {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #20150f, stop:0.5 #2b1d14, stop:1 #19110d);
    border-bottom: 1px solid #806634;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QLabel#brandMark {{ background: #17100c; color: {PALETTE.gold}; border: 1px solid {PALETTE.gold}; border-radius: 4px; font-size: 25px; font-weight: 700; }}
QLabel#brandName {{ color: #dec083; font-family: {DISPLAY_FONT_FALLBACK}; font-size: 25px; letter-spacing: 5px; }}
QLabel#brandSubtitle {{ color: #c9aa70; font-size: 17px; }}
QPushButton#chromeButton, QPushButton#chromeClose {{ background: transparent; border: 0; color: #d4b977; font-size: 22px; min-width: 34px; max-width: 34px; min-height: 34px; }}
QPushButton#chromeButton:hover {{ background: #3a2a1e; }}
QPushButton#chromeClose:hover {{ background: {PALETTE.burgundy}; color: #fff2d4; }}
QFrame#bookSpine {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #18100c, stop:0.5 #2a1b13, stop:1 #19110d); border-right: 1px solid #6f542a; }}
QPushButton#chapterButton {{ text-align: left; padding: 13px 22px; background: transparent; color: #ddc89d; border: 0; border-left: 3px solid transparent; font-size: 16px; }}
QPushButton#chapterButton:hover {{ background: rgba(199,160,82,22); color: #fff0c6; }}
QPushButton#chapterButton:checked {{ background: rgba(199,160,82,35); color: #fff1c1; border-left: 3px solid {PALETTE.firefly}; }}
QLabel#spineNote {{ color: #a99065; font-size: 11px; padding: 12px 20px; }}
QFrame#parchmentPage, QStackedWidget#chapterStack {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #ead9b6, stop:0.48 {PALETTE.vellum}, stop:1 #c9a96d); }}
QScrollArea#chapterScroll, QScrollArea#chapterScroll > QWidget > QWidget {{ background: transparent; border: 0; }}
QLabel#pageTitle {{ font-size: 30px; font-weight: 700; color: #24180f; }}
QLabel#pageSubtitle {{ color: #6f5637; font-size: 13px; }}
QLabel#pageStatus {{ color: #4f633a; font-size: 22px; font-weight: 700; padding: 8px 0; }}
QLabel#sectionTitle {{ color: #3a291a; font-size: 19px; font-weight: 700; padding: 8px 0 5px 0; border-bottom: 1px solid #9f7b40; }}
QFrame#statusCard, QFrame#manuscriptPanel {{ background: rgba(244,226,185,190); border: 1px solid #9d7b43; border-radius: 4px; }}
QLabel#cardGlyph {{ color: #8d6c31; font-size: 20px; }}
QLabel#cardCaption {{ color: #6a5030; font-size: 12px; }}
QLabel#cardValue {{ color: #2a1e15; font-size: 15px; font-weight: 700; }}
QGroupBox {{ background: rgba(239,218,174,165); border: 1px solid #9c7840; border-radius: 4px; margin-top: 14px; padding: 14px 10px 10px 10px; font-weight: 700; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; color: #49331f; padding: 0 6px; }}
QPushButton {{ background: rgba(233,213,169,210); color: #2d2118; border: 1px solid #80602f; border-radius: 4px; padding: 8px 13px; min-height: 20px; }}
QPushButton:hover {{ background: #f0ddb0; border-color: #5f431f; }}
QPushButton:pressed {{ background: #c6a665; }}
QPushButton#primaryAction {{ background: #e3c786; border: 1px solid #5f431f; font-weight: 700; }}
QPushButton#dangerAction {{ color: #6b2925; border-color: #8c4b43; }}
QPushButton:disabled {{ color: #8a7658; background: rgba(210,192,154,100); border-color: #aa946e; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{ background: rgba(247,232,199,220); color: #2b2118; border: 1px solid #86683a; border-radius: 3px; padding: 7px; selection-background-color: #8a6a35; selection-color: #fff5dd; }}
QComboBox QAbstractItemView {{ background: #ead6aa; color: #2b2118; selection-background-color: #8a6a35; selection-color: #fff5dd; }}
QCheckBox {{ spacing: 8px; }}
QTableWidget {{ background: rgba(246,230,196,210); alternate-background-color: rgba(224,201,155,180); border: 1px solid #8f6f3d; gridline-color: #b59660; selection-background-color: #7d6034; selection-color: #fff4d3; }}
QHeaderView::section {{ background: #c9aa70; color: #332315; border: 0; border-right: 1px solid #94713c; padding: 7px; font-weight: 700; }}
QProgressBar {{ background: #8b754b; border: 0; height: 6px; border-radius: 3px; }}
QProgressBar::chunk {{ background: {PALETTE.moss}; border-radius: 3px; }}
QScrollBar:vertical {{ background: rgba(72,49,28,35); width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #9b7b48; border-radius: 4px; min-height: 28px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QDialog {{ background: {PALETTE.vellum}; }}
QMenu {{ background: #2a1c14; color: #e7d4ab; border: 1px solid #8a6b37; }}
QMenu::item:selected {{ background: #5a4327; }}
"""


OVERLAY_STYLE = f"""
QFrame#shell {{
    background: rgba(27, 21, 16, 244);
    border: 1px solid rgba(199, 160, 82, 210);
    border-radius: 10px;
    color: #ead9b6;
}}
QWidget#overlayContent {{ background: transparent; }}
QLabel {{ color: #ead9b6; font-family: {BODY_FONT_FALLBACK}; }}
QLabel#title {{ font-family: {DISPLAY_FONT_FALLBACK}; font-size: 21px; font-weight: 700; letter-spacing: 5px; color: #ddbd76; }}
QLabel#brandMark {{ background: #120d0a; border: 1px solid #8f6d38; border-radius: 4px; color: #d8b76d; }}
QLabel#status {{ background: rgba(70,82,42,210); color: #eef6bd; padding: 5px 10px; border: 1px solid rgba(220,238,136,90); border-radius: 4px; }}
QLabel#hint {{ color: #a99570; font-size: 11px; }}
QLabel#contextRibbon {{ color: #d8c397; background: rgba(0,0,0,35); border-top: 1px solid rgba(199,160,82,70); border-bottom: 1px solid rgba(199,160,82,70); padding: 7px 10px; }}
QFrame#card {{ background: rgba(45,34,25,210); border: 1px solid rgba(156,119,62,100); border-radius: 4px; }}
QFrame#adviceCard {{ background: rgba(39,29,21,232); border: 1px solid #9d793d; border-radius: 5px; }}
QLabel#caption {{ color: #c49c53; font-size: 12px; font-weight: 700; }}
QLabel#value {{ color: #eadcbd; font-size: 13px; }}
QLabel#advice {{ color: #fff0cf; font-size: 19px; font-weight: 700; }}
QLabel#thinking {{ color: #f1dfb9; font-size: 15px; font-weight: 700; }}
QLabel#reason {{ color: #cab894; font-size: 12px; }}
QLabel#source {{ color: #cda95f; font-size: 11px; }}
QLabel#metrics {{ color: #a7b678; font-size: 11px; }}
QScrollArea#mainScroll, QScrollArea#mainScroll > QWidget > QWidget {{ background: transparent; border: 0; }}
QLabel#events {{ color: #b9a987; font-family: {BODY_FONT_FALLBACK}; font-size: 11px; padding: 7px; }}
QPushButton#close, QPushButton#windowControl {{ background: transparent; color: #d2b979; border: 0; font-size: 20px; width: 28px; padding: 0; }}
QPushButton#close:hover {{ color: #fff0d0; background: {PALETTE.burgundy}; border-radius: 4px; }}
QPushButton#windowControl:hover {{ color: #fff0d0; background: rgba(199,160,82,40); border-radius: 4px; }}
QPushButton#primary {{ background: #dfc17d; color: #2b2118; border: 1px solid #8b6734; border-radius: 4px; padding: 8px; font-weight: 700; }}
QPushButton#primary:pressed {{ background: #bd9850; }}
QPushButton {{ background: rgba(61,46,34,230); color: #e9d8b7; border: 1px solid rgba(155,119,64,135); border-radius: 4px; padding: 7px; }}
QPushButton:hover {{ background: rgba(89,66,42,240); border-color: #c19a54; }}
QComboBox, QLineEdit {{ background: rgba(20,15,12,190); color: #f0dfbd; border: 1px solid #7d6138; border-radius: 4px; padding: 7px; selection-background-color: #806132; selection-color: #fff5db; }}
QComboBox QAbstractItemView {{ background: #251b14; color: #f0dfbd; border: 1px solid #80643b; selection-background-color: #795d32; selection-color: #fff5db; }}
QCheckBox {{ color: #ead9b6; }}
QScrollArea#eventScroll {{ background: rgba(10,8,6,80); border: 1px solid rgba(130,100,57,75); border-radius: 4px; }}
QScrollArea#eventScroll > QWidget > QWidget {{ background: transparent; }}
QProgressBar {{ background: #413621; border: 0; height: 5px; border-radius: 2px; }}
QProgressBar::chunk {{ background: {PALETTE.moss}; border-radius: 2px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #7f6236; border-radius: 3px; min-height: 26px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
"""
