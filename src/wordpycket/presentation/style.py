from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication, QFrame, QGraphicsDropShadowEffect


def font_config() -> dict[str, str]:
    if sys.platform == "darwin":
        return {
            "app_font": "SF Pro Text",
            "ui_stack": '"SF Pro Text", ".AppleSystemUIFont", "PingFang SC"',
            "display_stack": '"SF Pro Display", ".AppleSystemUIFont", "PingFang SC"',
            "word_stack": '"SF Pro Display", ".AppleSystemUIFont", "PingFang SC"',
            "icon_stack": '"SF Pro Display", ".AppleSystemUIFont", "PingFang SC"',
        }
    return {
        "app_font": "Segoe UI",
        "ui_stack": '"Segoe UI", "Microsoft YaHei UI"',
        "display_stack": '"Segoe UI", "Microsoft YaHei UI"',
        "word_stack": '"Segoe UI", "Microsoft YaHei UI"',
        "icon_stack": '"Segoe UI Symbol", "Segoe UI", "Microsoft YaHei UI"',
    }


def apply_app_style(app: QApplication) -> None:
    fonts = font_config()
    app.setFont(QFont(fonts["app_font"], 10))
    app.setStyleSheet(
        f"""
        QWidget {{
            background: transparent;
            color: #172033;
            font-family: {fonts["ui_stack"]};
            font-size: 14px;
        }}
        QWidget#appSurface {{
            background: qlineargradient(
                x1: 0, y1: 0, x2: 1, y2: 1,
                stop: 0 #f7fbff,
                stop: 0.45 #edf4ff,
                stop: 1 #f8fbff
            );
        }}
        QFrame#glassPanel {{
            background: rgba(255, 255, 255, 214);
            border: 1px solid rgba(216, 228, 242, 230);
            border-radius: 16px;
        }}
        QLabel {{
            background: transparent;
        }}
        QLabel#homeTitle {{
            font-family: {fonts["display_stack"]};
            font-size: 32px;
            font-weight: 700;
        }}
        QLabel#title {{
            font-size: 22px;
            font-weight: 700;
        }}
        QLabel#word {{
            font-family: {fonts["word_stack"]};
            font-size: 48px;
            font-weight: 800;
        }}
        QLabel#meaning {{
            font-size: 22px;
        }}
        QLabel#meta {{
            color: #596579;
        }}
        QLabel#formLabel {{
            color: #596579;
            font-weight: 700;
        }}
        QPushButton {{
            background: rgba(255, 255, 255, 230);
            border: 1px solid rgba(205, 219, 237, 230);
            border-radius: 12px;
            padding: 8px 16px;
            font-weight: 700;
            min-height: 24px;
        }}
        QPushButton:hover {{
            background: rgba(244, 249, 255, 245);
            border-color: rgba(170, 197, 228, 240);
        }}
        QPushButton:pressed {{
            background: rgba(226, 240, 255, 240);
        }}
        QPushButton:disabled {{
            color: #9aa6b5;
            background: rgba(238, 243, 249, 150);
            border-color: rgba(218, 228, 240, 180);
        }}
        QPushButton[variant="primary"] {{
            color: white;
            background: rgba(10, 116, 220, 235);
            border-color: rgba(10, 116, 220, 235);
        }}
        QPushButton[variant="primary"]:hover {{
            background: rgba(0, 99, 195, 240);
            border-color: rgba(0, 99, 195, 240);
        }}
        QPushButton[variant="danger"] {{
            color: #c3261f;
            background: rgba(255, 245, 244, 230);
            border-color: rgba(246, 196, 191, 230);
        }}
        QPushButton[variant="danger"]:hover {{
            color: #9f1f19;
            background: rgba(255, 235, 233, 240);
            border-color: rgba(232, 164, 158, 240);
        }}
        QPushButton[variant="icon"] {{
            border-radius: 12px;
            min-width: 40px;
            max-width: 40px;
            min-height: 40px;
            max-height: 40px;
            padding: 0 0 3px 0;
            font-family: {fonts["icon_stack"]};
            font-size: 24px;
            font-weight: 400;
        }}
        QPushButton[variant="icon"]:pressed {{
            padding: 0 0 3px 0;
        }}
        QLineEdit, QPlainTextEdit {{
            background: rgba(255, 255, 255, 235);
            border: 1px solid rgba(205, 219, 237, 230);
            border-radius: 12px;
            padding: 8px 10px;
            color: #172033;
            selection-background-color: #d8ebff;
        }}
        QDialog,
        QInputDialog {{
            background: #f7fbff;
            color: #172033;
        }}
        QInputDialog QLabel {{
            background: transparent;
            color: #172033;
        }}
        QInputDialog QLineEdit {{
            background: #ffffff;
            color: #172033;
            border: 1px solid #d8e4f2;
            border-radius: 12px;
            padding: 7px 10px;
            selection-background-color: #d8ebff;
            selection-color: #172033;
        }}
        QInputDialog QPushButton {{
            background: #eef5ff;
            color: #172033;
            border: 1px solid #d8e4f2;
            border-radius: 12px;
            padding: 7px 18px;
            min-width: 72px;
            min-height: 22px;
        }}
        QInputDialog QPushButton:hover {{
            background: #e2efff;
        }}
        QInputDialog QPushButton:pressed {{
            background: #d8ebff;
        }}
        QMessageBox {{
            background: #f7fbff;
            color: #172033;
        }}
        QMessageBox QLabel {{
            background: transparent;
            color: #172033;
        }}
        QMessageBox QPushButton {{
            background: #eef5ff;
            color: #172033;
            border: 1px solid #d8e4f2;
            border-radius: 12px;
            padding: 7px 18px;
            min-width: 72px;
            min-height: 22px;
        }}
        QMessageBox QPushButton:hover {{
            background: #e2efff;
        }}
        QMessageBox QPushButton:pressed {{
            background: #d8ebff;
        }}
        QTableWidget {{
            background: rgba(255, 255, 255, 226);
            border: 1px solid rgba(216, 228, 242, 230);
            border-radius: 14px;
            gridline-color: transparent;
            alternate-background-color: rgba(247, 250, 255, 118);
            selection-background-color: rgba(212, 226, 244, 178);
            selection-color: #172033;
        }}
        QTableWidget::item {{
            border: none;
            padding: 2px 6px;
        }}
        QTableWidget::item:hover {{
            background: rgba(235, 243, 253, 130);
        }}
        QTableWidget::item:selected {{
            background: rgba(212, 226, 244, 178);
            color: #172033;
            border: none;
        }}
        QTableWidget::item:selected:active,
        QTableWidget::item:selected:!active {{
            background: rgba(212, 226, 244, 178);
            color: #172033;
        }}
        QTableWidget::item:focus {{
            outline: none;
        }}
        QHeaderView::section {{
            background: rgba(238, 244, 252, 185);
            color: #596579;
            border: none;
            border-bottom: 1px solid rgba(216, 228, 242, 150);
            padding: 8px;
            font-weight: 700;
        }}
        QScrollBar:vertical {{
            background: rgba(255, 255, 255, 80);
            border: none;
            border-radius: 7px;
            width: 14px;
            margin: 2px;
        }}
        QScrollBar::handle:vertical {{
            background: rgba(89, 101, 121, 95);
            border-radius: 6px;
            min-height: 34px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: rgba(89, 101, 121, 130);
        }}
        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {{
            background: transparent;
            border: none;
            height: 0;
            width: 0;
        }}
        QScrollBar:horizontal {{
            background: rgba(255, 255, 255, 80);
            border: none;
            border-radius: 7px;
            height: 14px;
            margin: 2px;
        }}
        QScrollBar::handle:horizontal {{
            background: rgba(89, 101, 121, 95);
            border-radius: 6px;
            min-width: 34px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: rgba(89, 101, 121, 130);
        }}
        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal,
        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {{
            background: transparent;
            border: none;
            height: 0;
            width: 0;
        }}
        QProgressBar {{
            background: rgba(232, 238, 247, 170);
            border: none;
            border-radius: 5px;
            height: 10px;
            text-align: center;
        }}
        QProgressBar::chunk {{
            background: rgba(10, 132, 255, 220);
            border-radius: 5px;
        }}
        """
    )


def create_panel() -> QFrame:
    panel = QFrame()
    panel.setObjectName("glassPanel")
    effect = QGraphicsDropShadowEffect(panel)
    effect.setBlurRadius(28)
    effect.setOffset(0, 10)
    effect.setColor(QColor(78, 106, 145, 42))
    panel.setGraphicsEffect(effect)
    return panel
