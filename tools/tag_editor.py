#!/usr/bin/env python3
import sys
from pathlib import Path
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TagEditor(QMainWindow):
    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.img_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        self.current_txt_path = None
        self.current_original_tags = ""

        self.setWindowTitle(f"Dataset Tag Editor - {self.dataset_dir}")
        self.resize(1200, 800)

        self.init_ui()
        self.load_dataset()

    def init_ui(self):
        # Main splitter to hold three sections
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # 1. Image Thumbnail List Section
        self.list_widget = QListWidget(self)
        self.list_widget.setIconSize(QSize(100, 100))
        self.list_widget.itemSelectionChanged.connect(self.on_item_selected)
        main_splitter.addWidget(self.list_widget)

        # 2. Single Image Display Section
        self.image_display = QWidget(self)
        img_layout = QVBoxLayout(self.image_display)
        self.img_view = QPushButton(self)  # Flat container to easily scale and center image alignment
        self.img_view.setFlat(True)
        self.img_view.setStyleSheet("background-color: #1e1e1e; border: none;")
        img_layout.addWidget(self.img_view)
        main_splitter.addWidget(self.image_display)

        # 3. Tag Text Edit Section
        edit_container = QWidget(self)
        edit_layout = QVBoxLayout(edit_container)

        self.tag_edit = QTextEdit(self)
        self.tag_edit.setPlaceholderText("Tags separated by commas...")
        edit_layout.addWidget(self.tag_edit)

        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save", self)
        self.save_btn.clicked.connect(self.save_tags)
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.clicked.connect(self.reset_tags)

        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        edit_layout.addLayout(btn_layout)

        main_splitter.addWidget(edit_container)

        # Set initial layout ratio (1:2:1)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 2)
        main_splitter.setStretchFactor(2, 1)

        self.setCentralWidget(main_splitter)

    def load_dataset(self):
        if not self.dataset_dir.exists() or not self.dataset_dir.is_dir():
            print(f"Error: Directory '{self.dataset_dir}' does not exist.")
            return

        # Find and sort all .txt files
        txt_files = sorted(list(self.dataset_dir.glob("*.txt")))
        
        for txt_path in txt_files:
            # Skip files in trash if the directory contains it
            if txt_path.parent.name == "trash":
                continue

            # Check if matching image exists
            img_path = None
            for ext in self.img_extensions:
                possible_img = txt_path.with_suffix(ext)
                if possible_img.exists():
                    img_path = possible_img
                    break

            if img_path:
                item = QListWidgetItem(txt_path.stem)
                # Load small thumbnail
                pixmap = QPixmap(str(img_path))
                if not pixmap.isNull():
                    thumb = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    item.setIcon(thumb)
                
                # Store absolute paths inside item UserRole
                item.setData(Qt.ItemDataRole.UserRole, (str(img_path), str(txt_path)))
                self.list_widget.addItem(item)

    def on_item_selected(self):
        selected_items = self.list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        img_path, txt_path = item.data(Qt.ItemDataRole.UserRole)
        self.current_txt_path = Path(txt_path)

        # Display full/larger image
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            # Dynamically scale to fit current display container size
            scaled_pixmap = pixmap.scaled(self.img_view.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.img_view.setIcon(scaled_pixmap)
            self.img_view.setIconSize(self.img_view.size())

        # Load Tags to textbox
        try:
            with open(self.current_txt_path, "r", encoding="utf-8") as f:
                self.current_original_tags = f.read()
            self.tag_edit.setPlainText(self.current_original_tags)
        except Exception as e:
            self.tag_edit.setPlainText(f"Error reading file: {e}")

    def save_tags(self):
        if not self.current_txt_path:
            return
        
        updated_tags = self.tag_edit.toPlainText()
        try:
            with open(self.current_txt_path, "w", encoding="utf-8") as f:
                f.write(updated_tags)
            self.current_original_tags = updated_tags
            print(f"Successfully saved: {self.current_txt_path.name}")
        except Exception as e:
            print(f"Error saving tags: {e}")

    def reset_tags(self):
        if self.current_txt_path:
            self.tag_edit.setPlainText(self.current_original_tags)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tag-editor.py <dataset_dir>")
        sys.exit(1)

    app = QApplication(sys.argv)
    editor = TagEditor(sys.argv[1])
    editor.show()
    sys.exit(app.exec())
