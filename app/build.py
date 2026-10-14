import customtkinter as ctk
import os
import json
import time
import re
from collections import Counter
import threading
from pathlib import Path

from text_to_language import (
    train_language_model,
    load_language_model
)

ctk.set_default_color_theme("theme.json")
ctk.set_appearance_mode("system")


# =========================
# Color variables
# =========================

# Accent
ACCENT = ("#B7F34A", "#B7F34A")
ACCENT_HOVER = ("#A5DC3C", "#C7FF66")
ACCENT_TEXT = ("#25271E", "#25271E")

# Backgrounds
BG_MAIN = ("#F5F4EF", "#262624")
BG_FRAME = ("#FFFFFF", "#30302D")
BG_SECONDARY = ("#F0EFE9", "#383834")
BG_HOVER = ("#EAE9E1", "#383834")

# Borders
BORDER = ("#E5E3DA", "#454540")
BORDER_ACCENT = ("#A5DC3C", "#A5DC3C")

# Text
TEXT_PRIMARY = ("#292923", "#F0EFE7")
TEXT_SECONDARY = ("#77776D", "#AAA99F")
TEXT_ACCENT = ("#729F24", "#B7F34A")

# Buttons
BTN_DEFAULT = ("#F5F4EF", "#262624")
BTN_HOVER = ("#EAE9E1", "#383834")


class App:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.geometry("1200x900")
        self.root.minsize(600, 400)
        self.root.title("Jaimes")

        # State
        self.active_sidebar = True
        self.active_chats = False
        self.devmode = True

        self.current_appearance = "System"
        self.current_selection = None
        self.item_buttons = []

        self.chats = os.listdir("/home/james/PycharmProjects/Jaimes/chats")
        self.projects = [el["name"] for el in json.load(open("/home/james/PycharmProjects/Jaimes/projects.json"))["projects"]]

        self.root.grid_rowconfigure(index=0, weight=1)
        self.root.grid_columnconfigure(index=0, weight=0)
        self.root.grid_columnconfigure(index=1, weight=1)
        self.root.grid_columnconfigure(index=2, weight=1)

        self.base_dir = Path(__file__).resolve().parent

        self.dataset_path = Path("/home/james/PycharmProjects/Jaimes/data/languages.csv")
        self.model_path = self.base_dir / "models" / "language_detector.npz"

        self.language_model = load_language_model(
            str(self.model_path)
        )

        self.training_thread = None
        self.training_stop_event = threading.Event()
        self.training_history = []
        self.training_stats = None

        self.init_sidebar()
        self.init_chat()
        if self.devmode:
            self.init_devpanel()

        self.root.mainloop()

    # =========================
    # Sidebar
    # =========================

    def init_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self.root,
            width=300,
            fg_color=BG_SECONDARY,
            corner_radius=0,
            border_width=0
        )
        self.sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.sidebar.grid_columnconfigure(0, weight=1)

        self.sidebar.grid_rowconfigure(0, weight=0)
        self.sidebar.grid_rowconfigure(1, weight=0)
        self.sidebar.grid_rowconfigure(2, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=0)

        # -------------------------
        # Header
        # -------------------------

        self.header = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent",
            corner_radius=0
        )

        self.header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=10,
            pady=(10, 5)
        )

        self.header.grid_columnconfigure(0, weight=1)

        self.sidebar_title = ctk.CTkLabel(
            self.header,
            text="Jaimes",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_PRIMARY
        )

        self.sidebar_title.grid(
            row=0,
            column=0,
            sticky="w",
            padx=5
        )

        self.btn_collapse = ctk.CTkButton(
            self.root,
            text="‹",
            width=35,
            height=35,
            font=ctk.CTkFont(size=22),
            fg_color=BTN_DEFAULT,
            hover_color=BTN_HOVER,
            text_color=TEXT_PRIMARY,
            command=self.toggle_sidebar
        )

        self.btn_collapse.place(
            x=315,
            y=10,
            anchor="nw"
        )

        # -------------------------
        # Search bar
        # -------------------------

        self.search_bar = ctk.CTkEntry(
            self.sidebar,
            height=42,
            placeholder_text="Search chats...",
            fg_color=BG_FRAME,
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_SECONDARY
        )

        self.search_bar.grid(
            row=1,
            column=0,
            padx=10,
            pady=5,
            sticky="ew"
        )

        # Search whenever the user types
        self.search_bar.bind(
            "<KeyRelease>",
            self.search_items
        )

        # -------------------------
        # Content
        # -------------------------

        self.list_canvas = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent",
            corner_radius=10
        )

        self.list_canvas.grid(
            row=2,
            column=0,
            padx=10,
            pady=5,
            sticky="nsew"
        )

        self.list_canvas.grid_columnconfigure(0, weight=1)
        self.list_canvas.grid_rowconfigure(1, weight=1)

        # Navigation buttons
        self.navigation = ctk.CTkFrame(
            self.list_canvas,
            fg_color="transparent"
        )

        self.navigation.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=0,
            pady=(0, 5)
        )

        self.navigation.grid_columnconfigure(
            (0, 1),
            weight=1
        )

        self.btn_chats = ctk.CTkButton(
            self.navigation,
            text="CHATS",
            height=40,
            corner_radius=8,
            command=self.set_chat
        )

        self.btn_chats.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 3)
        )

        self.btn_projects = ctk.CTkButton(
            self.navigation,
            text="PROJECTS",
            height=40,
            corner_radius=8,
            command=self.set_projects
        )

        self.btn_projects.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(3, 0)
        )

        # Scrollable list
        self.content_titles = ctk.CTkScrollableFrame(
            self.list_canvas,
            fg_color="transparent",
            corner_radius=8,
            scrollbar_button_color=BORDER_ACCENT[0],
            scrollbar_button_hover_color=ACCENT_HOVER[0]
        )

        self.content_titles.grid(
            row=1,
            column=0,
            sticky="nsew"
        )

        self.content_titles.grid_columnconfigure(
            0,
            weight=1
        )

        # -------------------------
        # Bottom section
        # -------------------------

        self.bottom = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent",
            corner_radius=0
        )

        self.bottom.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=10,
            pady=10
        )

        self.bottom.grid_columnconfigure(0, weight=1)

        self.btn_settings = ctk.CTkButton(
            self.bottom,
            text="⚙   Settings",
            height=42,
            anchor="w",
            fg_color=BTN_DEFAULT,
            hover_color=BTN_HOVER,
            text_color=TEXT_PRIMARY,
            border_width=1,
            border_color=BORDER,
            command=self.open_settings
        )

        self.btn_settings.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        self.account = ctk.CTkFrame(
            self.bottom,
            fg_color="transparent",
            corner_radius=0
        )

        self.account.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        self.account.grid_columnconfigure(0, weight=1)

        self.name = ctk.CTkLabel(
            self.account,
            text="James Potter",
            anchor="w",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=14, weight="bold")
        )

        self.name.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=5
        )

        self.tokens = ctk.CTkLabel(
            self.account,
            text="123K used",
            anchor="w",
            text_color=TEXT_ACCENT
        )

        self.tokens.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=5
        )

        # Initialize appearance and list
        self.set_chat()

    def init_chat(self):
        self.chat_panel = ctk.CTkFrame(
            self.root,
            fg_color=BG_MAIN,
            corner_radius=0
        )

        self.chat_panel.grid(
            row=0,
            column=1,
            padx=1,
            pady=0,
            sticky="nsew"
        )

        self.chat_panel.grid_columnconfigure(0, weight=1)
        self.chat_panel.grid_rowconfigure(1, weight=1)

        # Messages
        self.messages_frame = ctk.CTkScrollableFrame(
            self.chat_panel,
            fg_color="transparent",
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT_HOVER
        )

        self.messages_frame.grid(
            row=1,
            column=0,
            padx=20,
            pady=10,
            sticky="nsew"
        )

        self.messages_frame.grid_columnconfigure(0, weight=1)

        # Initial message
        ctk.CTkLabel(
            self.messages_frame,
            text=(
                "Send a message to analyze its language.\n"
                "The detected language and confidence will appear below."
            ),
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=14),
            justify="center"
        ).grid(
            row=0,
            column=0,
            padx=10,
            pady=30
        )

        self.message_row = 1

        # Input area
        self.input_frame = ctk.CTkFrame(
            self.chat_panel,
            fg_color=BG_FRAME,
            corner_radius=12,
            border_width=1,
            border_color=BORDER
        )

        self.input_frame.grid(
            row=2,
            column=0,
            padx=20,
            pady=(10, 20),
            sticky="ew"
        )

        self.input_frame.grid_columnconfigure(0, weight=1)

        self.message_input = ctk.CTkEntry(
            self.input_frame,
            height=45,
            placeholder_text="Enter a message...",
            fg_color="transparent",
            border_width=0,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=14)
        )

        self.message_input.grid(
            row=0,
            column=0,
            padx=(12, 5),
            pady=8,
            sticky="ew"
        )

        self.message_input.bind(
            "<Return>",
            self.send_message
        )

        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            width=80,
            height=38,
            corner_radius=8,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=ACCENT_TEXT,
            command=self.send_message
        )

        self.send_button.grid(
            row=0,
            column=1,
            padx=(5, 8),
            pady=8
        )

    def init_devpanel(self):
        self.dev_panel = ctk.CTkFrame(
            self.root,
            fg_color=BG_SECONDARY,
            corner_radius=0
        )

        self.dev_panel.grid(
            row=0,
            column=2,
            padx=(1, 0),
            pady=0,
            sticky="nsew"
        )

        self.dev_panel.grid_columnconfigure(0, weight=1)
        self.dev_panel.grid_rowconfigure(3, weight=1)

        # Statistics
        self.message_count = 0
        self.language_counts = Counter()
        self.inference_times = []

        # Header
        ctk.CTkLabel(
            self.dev_panel,
            text="Developer Panel",
            font=ctk.CTkFont(size=21, weight="bold"),
            text_color=TEXT_PRIMARY
        ).grid(
            row=0,
            column=0,
            padx=18,
            pady=(22, 5),
            sticky="w"
        )

        ctk.CTkLabel(
            self.dev_panel,
            text="Live model statistics",
            text_color=TEXT_SECONDARY
        ).grid(
            row=1,
            column=0,
            padx=18,
            pady=(0, 18),
            sticky="w"
        )

        # General statistics
        self.stats_frame = ctk.CTkFrame(
            self.dev_panel,
            fg_color=BG_FRAME,
            corner_radius=10,
            border_width=1,
            border_color=BORDER
        )

        self.stats_frame.grid(
            row=2,
            column=0,
            padx=12,
            pady=(0, 12),
            sticky="ew"
        )

        self.stats_frame.grid_columnconfigure(0, weight=1)

        self.total_label = ctk.CTkLabel(
            self.stats_frame,
            text="Messages analyzed: 0",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.total_label.grid(
            row=0,
            column=0,
            padx=12,
            pady=(12, 5),
            sticky="ew"
        )

        self.languages_label = ctk.CTkLabel(
            self.stats_frame,
            text="Languages detected: 0",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.languages_label.grid(
            row=1,
            column=0,
            padx=12,
            pady=5,
            sticky="ew"
        )

        self.average_time_label = ctk.CTkLabel(
            self.stats_frame,
            text="Avg. inference time: 0.00 ms",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.average_time_label.grid(
            row=2,
            column=0,
            padx=12,
            pady=5,
            sticky="ew"
        )

        self.last_language_label = ctk.CTkLabel(
            self.stats_frame,
            text="Last prediction: —",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.last_language_label.grid(
            row=3,
            column=0,
            padx=12,
            pady=(5, 12),
            sticky="ew"
        )

        # Language distribution
        self.distribution_frame = ctk.CTkScrollableFrame(
            self.dev_panel,
            fg_color=BG_FRAME,
            corner_radius=10,
            label_text="Language distribution",
            label_text_color=TEXT_PRIMARY,
            scrollbar_button_color=BORDER,
            scrollbar_button_hover_color=ACCENT_HOVER
        )

        self.distribution_frame.grid(
            row=3,
            column=0,
            padx=12,
            pady=(0, 12),
            sticky="nsew"
        )

        self.distribution_frame.grid_columnconfigure(0, weight=1)

        # Latest prediction
        self.latest_frame = ctk.CTkFrame(
            self.dev_panel,
            fg_color=BG_FRAME,
            corner_radius=10,
            border_width=1,
            border_color=BORDER
        )

        self.latest_frame.grid(
            row=4,
            column=0,
            padx=12,
            pady=(0, 12),
            sticky="ew"
        )

        ctk.CTkLabel(
            self.latest_frame,
            text="Latest prediction",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_PRIMARY
        ).pack(
            anchor="w",
            padx=12,
            pady=(12, 5)
        )

        self.latest_text_label = ctk.CTkLabel(
            self.latest_frame,
            text="No message analyzed yet.",
            text_color=TEXT_SECONDARY,
            justify="left",
            wraplength=240
        )

        self.latest_text_label.pack(
            anchor="w",
            padx=12,
            pady=(0, 12)
        )

        self.latest_result_label = ctk.CTkLabel(
            self.latest_frame,
            text="",
            text_color=TEXT_ACCENT,
            font=ctk.CTkFont(size=13, weight="bold")
        )

        self.latest_result_label.pack(
            anchor="w",
            padx=12,
            pady=(0, 12)
        )

        # =========================
        # Training statistics
        # =========================

        self.training_frame = ctk.CTkFrame(
            self.dev_panel,
            fg_color=BG_FRAME,
            corner_radius=10,
            border_width=1,
            border_color=BORDER
        )

        self.training_frame.grid(
            row=5,
            column=0,
            padx=12,
            pady=(0, 12),
            sticky="ew"
        )

        self.training_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.training_frame,
            text="Neural Network",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=TEXT_PRIMARY
        ).grid(
            row=0,
            column=0,
            padx=12,
            pady=(12, 8),
            sticky="w"
        )

        self.training_status_label = ctk.CTkLabel(
            self.training_frame,
            text="Status: Ready",
            text_color=TEXT_SECONDARY,
            anchor="w"
        )

        self.training_status_label.grid(
            row=1,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.epoch_label = ctk.CTkLabel(
            self.training_frame,
            text="Epoch: 0 / 15",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.epoch_label.grid(
            row=2,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.train_loss_label = ctk.CTkLabel(
            self.training_frame,
            text="Training loss: —",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.train_loss_label.grid(
            row=3,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.val_loss_label = ctk.CTkLabel(
            self.training_frame,
            text="Validation loss: —",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.val_loss_label.grid(
            row=4,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.train_accuracy_label = ctk.CTkLabel(
            self.training_frame,
            text="Training accuracy: —",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.train_accuracy_label.grid(
            row=5,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.val_accuracy_label = ctk.CTkLabel(
            self.training_frame,
            text="Validation accuracy: —",
            text_color=TEXT_PRIMARY,
            anchor="w"
        )

        self.val_accuracy_label.grid(
            row=6,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.training_progress = ctk.CTkProgressBar(
            self.training_frame,
            progress_color=ACCENT,
            fg_color=BORDER
        )

        self.training_progress.grid(
            row=7,
            column=0,
            padx=12,
            pady=8,
            sticky="ew"
        )

        self.training_progress.set(0)

        self.device_label = ctk.CTkLabel(
            self.training_frame,
            text="Device: CPU (NumPy)",
            text_color=TEXT_SECONDARY,
            anchor="w"
        )

        self.device_label.grid(
            row=8,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.best_epoch_label = ctk.CTkLabel(
            self.training_frame,
            text="Best epoch: —",
            text_color=TEXT_SECONDARY,
            anchor="w"
        )

        self.best_epoch_label.grid(
            row=9,
            column=0,
            padx=12,
            pady=3,
            sticky="ew"
        )

        self.train_button = ctk.CTkButton(
            self.training_frame,
            text="Start Training",
            height=38,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=ACCENT_TEXT,
            command=self.start_training
        )

        self.train_button.grid(
            row=10,
            column=0,
            padx=12,
            pady=(10, 12),
            sticky="ew"
        )

    def send_message(self, event=None):
        text = self.message_input.get().strip()

        if not text:
            return "break"

        self.message_input.delete(0, "end")

        # Measure inference time.
        start_time = time.perf_counter()

        result = self.detect_language(text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        language = result["language"]
        confidence = result["confidence"]

        # Update statistics.
        self.message_count += 1
        self.language_counts[language] += 1
        self.inference_times.append(elapsed_ms)

        self.chat_status.configure(
            text=f"Analyzed in {elapsed_ms:.2f} ms"
        )

        # Message container.
        message_frame = ctk.CTkFrame(
            self.messages_frame,
            fg_color=BG_FRAME,
            corner_radius=10,
            border_width=1,
            border_color=BORDER
        )

        message_frame.grid(
            row=self.message_row,
            column=0,
            padx=5,
            pady=6,
            sticky="ew"
        )

        message_frame.grid_columnconfigure(0, weight=1)

        # User message.
        ctk.CTkLabel(
            message_frame,
            text=text,
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=14),
            justify="left",
            anchor="w",
            wraplength=500
        ).grid(
            row=0,
            column=0,
            padx=12,
            pady=(12, 6),
            sticky="ew"
        )

        # Detection result.
        if language == "Unknown":
            result_text = "Language: Unknown — insufficient evidence"
        else:
            result_text = (
                f"Language: {language}  ·  "
                f"Confidence: {confidence:.1%}"
            )

        ctk.CTkLabel(
            message_frame,
            text=result_text,
            text_color=TEXT_ACCENT,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        ).grid(
            row=1,
            column=0,
            padx=12,
            pady=(0, 4),
            sticky="ew"
        )

        # Show alternative predictions.
        alternatives = result.get("alternatives", [])

        if alternatives:
            alternative_text = "Alternatives: " + ", ".join(
                f"{item['language']} ({item['confidence']:.1%})"
                for item in alternatives
            )

            ctk.CTkLabel(
                message_frame,
                text=alternative_text,
                text_color=TEXT_SECONDARY,
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=500
            ).grid(
                row=2,
                column=0,
                padx=12,
                pady=(0, 4),
                sticky="ew"
            )

        ctk.CTkLabel(
            message_frame,
            text=f"{len(text)} characters  ·  {elapsed_ms:.2f} ms",
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(size=10),
            anchor="w"
        ).grid(
            row=3,
            column=0,
            padx=12,
            pady=(0, 12),
            sticky="ew"
        )

        self.message_row += 1

        # Update developer panel.
        if self.devmode:
            self.update_devpanel(text, result, elapsed_ms)

        # Scroll to the latest message.
        self.messages_frame._parent_canvas.yview_moveto(1.0)

        return "break"

    def detect_language(self, text):
        if self.language_model is None:
            return {
                "language": "Model not trained",
                "confidence": 0.0,
                "alternatives": [],
                "inference_time_ms": 0.0
            }

        return self.language_model.predict(
            text,
            top_k=3
        )

    # =========================
    # Search functionality
    # =========================

    def search_items(self, event=None):
        query = self.search_bar.get().strip().casefold()

        items = self.chats if self.active_chats else self.projects

        filtered_items = [
            item for item in items
            if query in item.casefold()
        ]

        self.render_items(filtered_items + ["New Chat" if self.active_chats else "New Project"])

    # =========================
    # Render chat/project list
    # =========================

    def render_items(self, items):
        # Remove existing buttons
        for button in self.item_buttons:
            button.destroy()

        self.item_buttons.clear()

        for index, item in enumerate(items):
            selected = item == self.current_selection

            button = ctk.CTkButton(
                self.content_titles,
                text=item,
                height=42,
                anchor="w",
                corner_radius=8,
                fg_color=ACCENT if selected else "transparent",
                hover_color=ACCENT_HOVER if selected else BG_HOVER,
                text_color=ACCENT_TEXT if selected else TEXT_PRIMARY,
                command=lambda name=item: self.select_item(name)
            )

            button.grid(
                row=index,
                column=0,
                padx=3,
                pady=3,
                sticky="ew"
            )

            self.item_buttons.append(button)

    # =========================
    # Select a chat or project
    # =========================

    def select_item(self, name):
        self.current_selection = name

        self.render_items(
            self.chats + ["New Chat"] if self.active_chats else self.projects + ["New Project"]
        )

        print(f"Selected: {name}")

    # =========================
    # Navigation
    # =========================

    def set_chat(self):
        if self.active_chats:
            return

        self.active_chats = True
        self.current_selection = None

        self.btn_chats.configure(
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=ACCENT_TEXT
        )

        self.btn_projects.configure(
            fg_color=BTN_DEFAULT,
            hover_color=BTN_HOVER,
            text_color=TEXT_PRIMARY
        )

        self.search_bar.configure(
            placeholder_text="Search chats..."
        )

        self.search_items()

    def set_projects(self):
        if not self.active_chats:
            return

        self.active_chats = False
        self.current_selection = None

        self.btn_projects.configure(
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=ACCENT_TEXT
        )

        self.btn_chats.configure(
            fg_color=BTN_DEFAULT,
            hover_color=BTN_HOVER,
            text_color=TEXT_PRIMARY
        )

        self.search_bar.configure(
            placeholder_text="Search projects..."
        )

        self.search_items()

    # =========================
    # Collapse / expand sidebar
    # =========================

    def toggle_sidebar(self):
        self.active_sidebar = not self.active_sidebar

        if self.active_sidebar:
            # Show the existing sidebar
            self.sidebar.place(
                x=0,
                y=0,
                relheight=1
            )

            self.btn_collapse.configure(text="‹")
            self.btn_collapse.place(x=315, y=10, anchor="nw")

        else:
            # Hide the sidebar completely
            self.sidebar.place_forget()

            self.btn_collapse.configure(text="›")
            self.btn_collapse.place(x=10, y=10, anchor="nw")

    # =========================
    # Settings window
    # =========================

    def open_settings(self):
        if hasattr(self, "settings_window"):
            if self.settings_window.winfo_exists():
                self.settings_window.focus()
                return

        self.settings_window = ctk.CTkToplevel(self.root)

        self.settings_window.title("Jaimes Settings")
        self.settings_window.geometry("420x350")
        self.settings_window.resizable(False, False)

        self.settings_window.configure(
            fg_color=BG_MAIN
        )

        # Title
        title = ctk.CTkLabel(
            self.settings_window,
            text="Settings",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=TEXT_PRIMARY
        )

        title.pack(
            anchor="w",
            padx=25,
            pady=(25, 20)
        )

        # Appearance
        appearance_label = ctk.CTkLabel(
            self.settings_window,
            text="Appearance",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=14, weight="bold")
        )

        appearance_label.pack(
            anchor="w",
            padx=25,
            pady=(0, 5)
        )

        appearance_menu = ctk.CTkOptionMenu(
            self.settings_window,
            values=["System", "Light", "Dark"],
            fg_color=ACCENT,
            button_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            text_color=ACCENT_TEXT,
            dropdown_fg_color=BG_FRAME,
            dropdown_text_color=TEXT_PRIMARY,
            command=self.change_appearance
        )

        appearance_menu.set(self.current_appearance)

        appearance_menu.pack(
            fill="x",
            padx=25,
            pady=(0, 20)
        )

        # Sidebar option
        sidebar_label = ctk.CTkLabel(
            self.settings_window,
            text="Sidebar",
            text_color=TEXT_PRIMARY,
            font=ctk.CTkFont(size=14, weight="bold")
        )

        sidebar_label.pack(
            anchor="w",
            padx=25,
            pady=(0, 5)
        )

        sidebar_switch = ctk.CTkSwitch(
            self.settings_window,
            text="Show sidebar",
            text_color=TEXT_PRIMARY,
            progress_color=ACCENT,
            button_hover_color=ACCENT_HOVER,
            command=self.toggle_sidebar
        )

        sidebar_switch.pack(
            anchor="w",
            padx=25,
            pady=(0, 20)
        )

        if self.active_sidebar:
            sidebar_switch.select()
        else:
            sidebar_switch.deselect()

        # Close button
        close_button = ctk.CTkButton(
            self.settings_window,
            text="Done",
            height=38,
            fg_color=ACCENT,
            hover_color=ACCENT_HOVER,
            text_color=ACCENT_TEXT,
            command=self.settings_window.destroy
        )

        close_button.pack(
            fill="x",
            padx=25,
            pady=(5, 20)
        )

        self.settings_window.transient(self.root)
        self.settings_window.grab_set()

    # =========================
    # Appearance
    # =========================

    def change_appearance(self, appearance):
        self.current_appearance = appearance

        ctk.set_appearance_mode(
            appearance.lower()
        )

    def start_training(self):
        if (
                self.training_thread is not None
                and self.training_thread.is_alive()
        ):
            return

        if not self.dataset_path.exists():
            self.training_status_label.configure(
                text="Status: Dataset not found"
            )
            return

        self.training_stop_event.clear()
        self.training_history.clear()

        self.train_button.configure(
            state="disabled",
            text="Training..."
        )

        self.training_status_label.configure(
            text="Status: Preparing dataset..."
        )

        self.training_progress.set(0)

        self.training_thread = threading.Thread(
            target=self._training_worker,
            daemon=True
        )

        self.training_thread.start()

    def _training_worker(self):
        try:
            detector = train_language_model(
                csv_path=str(self.dataset_path),
                model_path=str(self.model_path),
                epochs=15,
                batch_size=64,
                progress_callback=self._training_progress_callback,
                stop_event=self.training_stop_event
            )

            self.root.after(
                0,
                lambda: self._training_finished(detector)
            )

        except Exception as error:
            message = str(error)

            self.root.after(
                0,
                lambda: self._training_failed(message)
            )

    def _training_progress_callback(self, stats):
        self.root.after(
            0,
            lambda: self.update_training_stats(stats)
        )

    def update_training_stats(self, stats):
        self.training_stats = stats
        self.training_history.append(stats)

        self.training_status_label.configure(
            text="Status: Training"
        )

        self.epoch_label.configure(
            text=f"Epoch: {stats['epoch']} / {stats['epochs']}"
        )

        self.train_loss_label.configure(
            text=f"Training loss: {stats['train_loss']:.4f}"
        )

        self.val_loss_label.configure(
            text=f"Validation loss: {stats['val_loss']:.4f}"
        )

        self.train_accuracy_label.configure(
            text=f"Training accuracy: {stats['train_accuracy']:.2%}"
        )

        self.val_accuracy_label.configure(
            text=f"Validation accuracy: {stats['val_accuracy']:.2%}"
        )

        self.training_progress.set(
            stats["epoch"] / stats["epochs"]
        )

        self.device_label.configure(
            text=f"Device: {stats['device']}"
        )

        self.best_epoch_label.configure(
            text=f"Best epoch: {stats['best_epoch']}"
        )

    def _training_finished(self, detector):
        self.language_model = detector

        self.training_status_label.configure(
            text="Status: Training completed"
        )

        self.train_button.configure(
            state="normal",
            text="Train Again"
        )

        print(f"Model saved to: {self.model_path}")

    def _training_failed(self, error):
        self.training_status_label.configure(
            text="Status: Training failed"
        )

        self.train_button.configure(
            state="normal",
            text="Retry Training"
        )

        print(f"Training error: {error}")


if __name__ == "__main__":
    App()