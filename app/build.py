import customtkinter as ctk

ctk.set_default_color_theme("theme.json")
ctk.set_appearance_mode("system")


class App:
    def __init__(self):
        # General settings
        self.root = ctk.CTk()
        self.root.geometry("1200x900")

        # General cache
        self.active_sidebar = True
        self.active_chats = True

        # Init the app
        self.init_state()

        self.btn_chats.configure(
            fg_color=("#139A43", "#0B5D1E"),
            hover_color=("#0B5D1E", "#139A43")
        )
        self.btn_projects.configure(
            fg_color=("gray95", "gray10"),
            hover_color=("gray97", "gray12")
        )

    def mainloop(self):
        self.root.mainloop()

    def init_state(self):
        self.init_sidebar()

    def init_sidebar(self):
        if self.active_sidebar:
            # Create a vertical expanding Frame for the sidebar, that has the fixed width of 300
            side_bar = ctk.CTkFrame(
                self.root,
                width=300
            )
            side_bar.place(
                x=0,
                y=0,
                relheight=1,
                anchor="nw"
            )

            # Config the sidebar to have three parts
            side_bar.grid_rowconfigure(index=0, weight=0) # Search (fixed height)
            side_bar.grid_rowconfigure(index=1, weight=1) # Content (adjusting height)
            side_bar.grid_rowconfigure(index=2, weight=0) # Settings and Account (fixed height)

            side_bar.grid_columnconfigure(index=0, weight=1) # adjusting width (prefixed to 300px)

            # Search
            search_bar = ctk.CTkEntry(
                side_bar,
                width=280,
                height=50,
                placeholder_text="Search..."
            )
            search_bar.grid(
                row=0,
                padx=10,
                pady=10
            )

            # Content
            list_canvas = ctk.CTkFrame(
                side_bar,
                fg_color="transparent",
                border_color=("#139A43", "#0B5D1E"),
                border_width=2
            )
            list_canvas.grid(
                row=1,
                column=0,
                padx=10,
                pady=0,
                sticky="nsew"
            )

            list_canvas.grid_rowconfigure(0, weight=0)  # Navigation buttons (fixed height)
            list_canvas.grid_rowconfigure(1, weight=1)  # Chat/project list (adjusting height)
            list_canvas.grid_columnconfigure(0, weight=1) # Predefined width
            list_canvas.grid_columnconfigure(1, weight=1)

            self.btn_chats = ctk.CTkButton(
                list_canvas,
                width=125,
                height=50,
                text="CHATS",
                command=self.set_chat
            )
            self.btn_projects = ctk.CTkButton(
                list_canvas,
                width=125,
                height=50,
                text="PROJECTS",
                command=self.set_projects
            )
            self.btn_chats.grid(
                row=0,
                column=0,
                padx=5,
                pady=5
            )
            self.btn_projects.grid(
                row=0,
                column=1,
                padx=5,
                pady=5
            )

            content_titles = ctk.CTkScrollableFrame(list_canvas)
            content_titles.grid(
                row=1,
                column=0,
                columnspan=2,
                padx=5,
                pady=5,
                sticky="nsew"
            )

            example = ["James", "Potter"]
            buttons = []

            if self.active_chats:
                for i, chat_title in enumerate(example):
                    buttons.append(ctk.CTkButton(
                        content_titles,
                        height=50,
                        text=chat_title,
                        anchor="w")
                    )
                    buttons[-1].configure(
                        fg_color="transparent",
                        text_color=("#139A43", "#0B5D1E"),
                        border_width=0
                    )
                    buttons[-1].grid(
                        row=i,
                        column=0,
                        padx=5,
                        pady=5
                    )

            bottom = ctk.CTkFrame(
                side_bar,
                width=280,
                height=120,
                fg_color="transparent",
                border_width=0
            )
            bottom.grid(
                row=2,
                padx=10,
                pady=10
            )

            # Settings and account
            settings = ctk.CTkButton(
                bottom,
                width=280,
                height=50,
                text="Settings"
            )
            settings.place(
                x=0,
                y=10,
                anchor="nw"
            )

            self.account = ctk.CTkFrame(
                bottom,
                width=260,
                height=50,
                border_width=0
            )
            self.account.place(
                x=0,
                y=60,
                anchor="nw"
            )

            self.name = ctk.CTkLabel(
                self.account,
                width=260,
                height=30,
                text="James Potter",
                anchor="sw",
                text_color=("#0B5D1E", "#139A43"),
                fg_color="transparent",
                bg_color="transparent"
            )
            self.tokens = ctk.CTkLabel(
                self.account,
                width=260,
                height=20,
                text="123K used",
                anchor="nw",
                text_color=("#139A43", "#0B5D1E"),
                fg_color="transparent",
                bg_color="transparent"
            )
            self.name.place(
                x=10,
                y=0,
                anchor="nw"
            )
            self.tokens.place(
                x=10,
                y=30,
                anchor="nw"
            )

    def set_chat(self):
        self.active_chats = True
        self.btn_chats.configure(
            fg_color=("#139A43", "#0B5D1E"),
            hover_color=("#0B5D1E", "#139A43")
        )
        self.btn_projects.configure(
            fg_color=("gray95", "gray10"),
            hover_color=("gray97", "gray12")
        )

    def set_projects(self):
        self.active_chats = False
        self.btn_chats.configure(
            fg_color=("gray95", "gray10"),
            hover_color=("gray97", "gray12")
        )
        self.btn_projects.configure(
            fg_color=("#139A43", "#0B5D1E"),
            hover_color=("#0B5D1E", "#139A43")
        )


if __name__ == "__main__":
    app = App()
    app.mainloop()
