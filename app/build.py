import customtkinter as ctk
import os
import json

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
        self.active_chats = True

        self.current_appearance = "System"
        self.current_selection = None
        self.item_buttons = []

        self.chats = os.listdir("/home/james/PycharmProjects/Jaimes/chats")
        self.projects = [el["name"] for el in json.load(open("/home/james/PycharmProjects/Jaimes/projects.json"))["projects"]]

        self.init_sidebar()

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

        self.sidebar.place(
            x=0,
            y=0,
            relheight=1
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


if __name__ == "__main__":
    App()