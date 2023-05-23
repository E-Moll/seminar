import typing
import math
import random
import tkinter as tk
import customtkinter as ctk
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest


class Application:
    # Constants
    WINDOW_GEOMETRY = "1750x950+70+30"
    NUM_DATAPOINTS_START = "1"
    STEP_SIZE_DP = "100"
    NORMAL_PERCENTAGE_START = "80"
    EPSILON_START = "2.5"
    MINPTS_START = "5"
    T_START = "100"
    PSI_START = "256"
    ANOMALY_SCORE_THRESHOLD_START = "0.5"
    EDITABLE_MEASURES = ["Data Point Nr.", "Min (for eq. dist.)", "Max (for eq. dist.)", "µ", "σ"]
    INITIAL_VALUES = [["km/h", "# of cars"], [-10.0, 10.0, -10.0], [-10.0, 10.0, 10.0], [-10.0, 10.0, 1.5],
                      [0.0, 10.0, 4.0]]  # [<MIN>, <MAX>, <INITIAL>]
    X_LIM = [-10.0, 10.0]
    Y_LIM = [-7.5, 7.5]
    STEP_SIZE_PERCENT = 2
    VALID_POINT_COLOR = "#00007f"
    EPSILON_RANGE_LINEWIDTH = 0.45
    EPSILON_RANGE_RED_COLOR = "#ff0000"
    EPSILON_RANGE_GRAY_3_COLOR = "#999999"
    EPSILON_RANGE_GRAY_2_COLOR = "#bbbbbb"
    EPSILON_RANGE_GRAY_1_COLOR = "#eeeeee"
    OUTLIER_TEXT_COLOR = "#ff9507"
    OUTLIER_POINT_COLOR = OUTLIER_TEXT_COLOR  # "#ffcc33"
    LABEL_COLOR = "#00007f"
    ALL_SIZE = 25
    OUTLIER_SIZE = 10
    MAX_ROWS_VALUES = 35
    NON_EDITABLE_MEASURES = ["Mean", "Standard Deviation", "% Marked As Outliers"]
    NUM_OF_MEASURES = len(EDITABLE_MEASURES) + len(NON_EDITABLE_MEASURES)
    FIRST_COLUMN_FIXED_STRINGS = EDITABLE_MEASURES + NON_EDITABLE_MEASURES + \
                                 [f"{i + 1}" for i in range(MAX_ROWS_VALUES)]

    def refill_coordinate_lists(self):
        self.outlier_x = [self.all_x[i] for i in self.outliers_indices]
        self.outlier_y = [self.all_y[i] for i in self.outliers_indices]

    def _range_query(self, db: np.ndarray, dist: typing.Callable[[tuple[float, float], tuple[float, float]], float], p: tuple[float, float], epsilon: float) -> list:
        """Finds all neighbors in db of a point p in its epsilon neighborhood using dist as a measuring function.
        p itself is included as the first element in the returned list."""
        neighbors = [p]
        for i in range(len(db)):
            if tuple(db[i]) != tuple(p) and dist(p, db[i]) <= epsilon:  # Check if the current two points are neighbors
                neighbors += [db[i]]
        return neighbors

    def dbscan_from_pseudocode(self):
        db = self.all_data
        epsilon = float(self.spinbox_msp_epsilon.get())  # 0.8
        min_pts = int(self.spinbox_msp_minpts.get())  # 5
        dist = lambda p, q: math.sqrt(pow(p[0] - q[0], 2) + pow(p[1] - q[1], 2))
        label = {tuple(p): "undefined" for p in db}

        for p in db:  # Iterate over every point
            if label[tuple(p)] != "undefined":  # Skip processed points
                continue
            n = self._range_query(db, dist, p, epsilon)  # Find initial neighbors
            if len(n) < min_pts:  # Non-core points are noise
                label[tuple(p)] = "noise"
                continue
            c = random.random() * 0xffffff  # Start a new cluster / generate random new label that serves as color code
            label[tuple(p)] = str(c)
            s = n[1:]  # Expand neighborhood
            for q in s:
                if label[tuple(q)] == "noise":
                    label[tuple(q)] = str(c)
                if label[tuple(q)] != "undefined":
                    continue
                n = self._range_query(db, dist, q, epsilon)
                label[tuple(q)] = str(c)
                if len(n) < min_pts:  # Core-point check
                    continue
                s += n

        # Gather outliers
        self.outlier_x = []
        self.outlier_y = []
        for [x, y], v in label.items():
            if v == "noise":
                self.outlier_x += [x]
                self.outlier_y += [y]

    def iforest_from_sklearn(self):
        """https://scikit-learn.org/stable/auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-examples-ensemble-plot-isolation-forest-py"""
        t = int(self.spinbox_msp_t.get())  # 100
        psi = int(self.spinbox_msp_psi.get())  # 256
        anomaly_score_threshold = float(self.spinbox_msp_anomaly_score_threshold.get())  # 0.6

        # Build the tree
        clf = IsolationForest(n_estimators=t, max_samples=psi, random_state=0)
        clf.fit(self.all_data)

        # Evaluate data points
        anomaly_indicator = clf.score_samples(self.all_data) + anomaly_score_threshold
        self.outliers_indices = [i for i in range(len(self.all_x)) if anomaly_indicator[i] < 0.0]
        self.refill_coordinate_lists()

    OUTLIER_METHODS = {"DBSCAN": dbscan_from_pseudocode, "Isolation Forest": iforest_from_sklearn}  # "Outlier Detection Using Custom DBSCAN": outlier_detection_using_custom_dbscan,

    # Some important variables
    num_of_outliers = None
    outliers_indices = None
    all_data = None
    all_x = None
    all_y = None
    outlier_x = None
    outlier_y = None
    selected_method = list(OUTLIER_METHODS.keys())[0]
    current_circle = None
    last_circle = None
    second_to_last_circle = None

    def replace(self, widget: tk.Widget, str_: str):
        state = widget.cget("state")
        widget.config(state="normal")
        widget.delete(0, "end")
        widget.insert(0, str_)
        widget.config(state=str(state))

    def __init__(self):
        """With the help of: https://www.youtube.com/watch?v=iM3kjbbKHQU"""

        ## Graph ##
        # Create an array of x values
        self.all_x = np.linspace(-5, 5, 101)

        # Calculate y values for a parabolic function
        self.all_y = self.all_x ** 2

        # Create a figure and axes
        self.fig, self.ax = plt.subplots()
        self.scatter_plot_all = self.ax.scatter(x=self.all_x, y=self.all_y, s=self.ALL_SIZE, c=self.VALID_POINT_COLOR)
        self.scatter_plot_outlier = self.ax.scatter(x=self.all_x, y=self.all_y, s=self.OUTLIER_SIZE,
                                                    c=self.OUTLIER_POINT_COLOR)
        self.ax.set_xlim(self.X_LIM[0], self.X_LIM[1])
        self.ax.set_ylim(self.Y_LIM[0], self.Y_LIM[1])
        self.ax.grid(True)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        ## UI general settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.geometry(self.WINDOW_GEOMETRY)
        self.root.title("Comparison of Data Mining Methods for Outlier Detection")

        ### Elements
        ## Frames
        self.frame_graph = tk.Frame(master=self.root)
        self.frame_graph.grid(row=0, column=0, pady=20, padx=60)

        self.frame_table = tk.Frame(master=self.root)
        self.frame_table.grid(row=0, column=1, rowspan=4, pady=20, padx=60)

        self.frame_settings = tk.Frame(master=self.root)
        self.frame_settings.grid(row=1, column=0, pady=20, padx=60)

        ## Graph
        self.canvas_graph = tkagg.FigureCanvasTkAgg(self.fig, master=self.frame_graph)
        def on_graph_click(event):
            x = event.xdata
            y = event.ydata
            if x is not None and y is not None:
                self.all_data = np.append(self.all_data, [[x, y]], axis=0)
                self.all_x = np.append(self.all_x, x)
                self.all_y = np.append(self.all_y, y)
                if self.second_to_last_circle is not None:
                    self.second_to_last_circle.set_color(self.EPSILON_RANGE_GRAY_1_COLOR)
                if self.last_circle is not None:
                    self.last_circle.set_color(self.EPSILON_RANGE_GRAY_2_COLOR)
                    self.second_to_last_circle = self.last_circle
                if self.current_circle is not None:
                    self.current_circle.set_color(self.EPSILON_RANGE_GRAY_3_COLOR)
                    self.last_circle = self.current_circle
                if self.selected_method == list(self.OUTLIER_METHODS.keys())[0]:
                    radius = float(self.spinbox_msp_epsilon.get())
                else:
                    radius = 0
                self.current_circle = plt.Circle((x, y), radius=radius, color=self.EPSILON_RANGE_RED_COLOR, fill=False, linewidth=self.EPSILON_RANGE_LINEWIDTH * radius)
                plt.gca().add_artist(self.current_circle)
                self.update()
        self.canvas_graph.mpl_connect("button_press_event", on_graph_click)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack()

        ## Table
        self.table = []

        def on_click(event):
            self.update()

        def on_click_gen_new(event):
            self.all_x, self.all_y, self.all_data = self.get_generated_values()
            self.update()

        for row, str_ in enumerate(self.FIRST_COLUMN_FIXED_STRINGS):
            sub_table = []
            for col in range(3):
                font = tk.font.Font()
                if row == 0:
                    font = tk.font.Font(weight="bold", size=12)

                if col == 0:
                    c = tk.Entry(master=self.frame_table, justify=tk.CENTER, font=font)
                    c.insert(0, str_)
                    c.config(state="readonly")
                    c.config(foreground=self.LABEL_COLOR)
                elif row == 0:
                    c = tk.Entry(master=self.frame_table, justify=tk.CENTER, font=font)
                    self.replace(c, self.INITIAL_VALUES[row][col - 1])
                elif row in range(1, len(self.EDITABLE_MEASURES)):
                    c = tk.Spinbox(master=self.frame_table, from_=self.INITIAL_VALUES[row][0],
                                   to=self.INITIAL_VALUES[row][1], increment=0.05 * self.INITIAL_VALUES[row][1],
                                   justify=tk.CENTER, font=font)
                    self.replace(c, self.INITIAL_VALUES[row][2])
                    c.bind("<Button-1>", on_click_gen_new)
                else:
                    c = tk.Entry(master=self.frame_table, justify=tk.CENTER, font=font)
                    c.insert(0, "---")
                    c.config(state="readonly")
                c.grid(row=row, column=col)
                sub_table.append(c)

            self.table.append(sub_table)

        ## Settings
        self.label_num_dp = ctk.CTkLabel(master=self.frame_settings, text="# of data points:", justify=tk.LEFT,
                                         text_color=self.LABEL_COLOR)
        self.spinbox_num_dp = tk.Spinbox(master=self.frame_settings, from_=0, to=50_000, increment=float(self.STEP_SIZE_DP),
                                         justify=tk.CENTER)
        self.spinbox_num_dp.bind("<Button-1>", on_click_gen_new)
        self.replace(self.spinbox_num_dp, self.NUM_DATAPOINTS_START)

        self.label_dr = ctk.CTkLabel(master=self.frame_settings, text="Distribution ratio:", justify=tk.LEFT,
                                     text_color=self.LABEL_COLOR)

        self.label_normal_pc = ctk.CTkLabel(master=self.frame_settings, text="Normal:", justify=tk.LEFT,
                                            text_color=self.LABEL_COLOR)

        def on_normal_pc_change():
            self.replace(self.spinbox_equal_pc, str(100.0 - float(self.spinbox_normal_pc.get())))

        self.spinbox_normal_pc = tk.Spinbox(master=self.frame_settings, from_=0, to=100,
                                            increment=self.STEP_SIZE_PERCENT,
                                            justify=tk.CENTER, command=on_normal_pc_change)
        self.spinbox_normal_pc.bind("<Button-1>", on_click_gen_new)
        self.replace(self.spinbox_normal_pc, self.NORMAL_PERCENTAGE_START)
        self.label_percent_1 = ctk.CTkLabel(master=self.frame_settings, text="%", justify=tk.LEFT,
                                            text_color=self.LABEL_COLOR)

        self.label_equal_pc = ctk.CTkLabel(master=self.frame_settings, text="Equal:", justify=tk.LEFT,
                                           text_color=self.LABEL_COLOR)

        def on_equal_pc_change():
            self.replace(self.spinbox_normal_pc, str(100.0 - float(self.spinbox_equal_pc.get())))

        self.spinbox_equal_pc = tk.Spinbox(master=self.frame_settings, from_=0, to=100,
                                           increment=self.STEP_SIZE_PERCENT,
                                           justify=tk.CENTER, command=on_equal_pc_change)
        self.spinbox_equal_pc.bind("<Button-1>", on_click_gen_new)
        self.replace(self.spinbox_equal_pc, str(100 - int(self.NORMAL_PERCENTAGE_START)))
        self.label_percent_2 = ctk.CTkLabel(master=self.frame_settings, text="%", justify=tk.LEFT,
                                            text_color=self.LABEL_COLOR)

        self.label_method = tk.Label(master=self.frame_settings,
                                     text="Outlier Detection Method: " + self.selected_method, justify=tk.LEFT,
                                     fg=self.LABEL_COLOR)

        self.listbox_method = tk.Listbox(master=self.frame_settings)
        methods = list(self.OUTLIER_METHODS.keys())
        self.listbox_method.insert(0, *methods)

        # Method specific parameter settings
        self.label_msp_epsilon = ctk.CTkLabel(master=self.frame_settings, text="Epsilon:",
                                              justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_epsilon = tk.Spinbox(master=self.frame_settings, from_=0, to=20, increment=0.1,
                                              justify=tk.CENTER)
        self.spinbox_msp_epsilon.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_epsilon, self.EPSILON_START)

        self.label_msp_minpts = ctk.CTkLabel(master=self.frame_settings, text="MinPts:",
                                             justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_minpts = tk.Spinbox(master=self.frame_settings, from_=1, to=1000, increment=3,
                                             justify=tk.CENTER)
        self.spinbox_msp_minpts.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_minpts, self.MINPTS_START)


        self.label_msp_t = ctk.CTkLabel(master=self.frame_settings, text="T:",
                                             justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_t = tk.Spinbox(master=self.frame_settings, from_=1, to=1000, increment=5,
                                             justify=tk.CENTER)
        self.spinbox_msp_t.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_t, self.T_START)

        self.label_msp_psi = ctk.CTkLabel(master=self.frame_settings, text="Psi:",
                                             justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_psi = tk.Spinbox(master=self.frame_settings, from_=1, to=10_000, increment=50,
                                             justify=tk.CENTER)
        self.spinbox_msp_psi.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_psi, self.PSI_START)

        self.label_msp_anomaly_score_threshold = ctk.CTkLabel(master=self.frame_settings, text="Anomaly Score Threshold:",
                                             justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_anomaly_score_threshold = tk.Spinbox(master=self.frame_settings, from_=0, to=1, increment=0.1,
                                             justify=tk.CENTER)
        self.spinbox_msp_anomaly_score_threshold.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_anomaly_score_threshold, self.ANOMALY_SCORE_THRESHOLD_START)

        def on_select(event):
            a = self.listbox_method.selection_get()
            if a == "0":
                return
            self.selected_method = a

            self.label_msp_epsilon.grid_forget()
            self.spinbox_msp_epsilon.grid_forget()

            self.label_msp_minpts.grid_forget()
            self.spinbox_msp_minpts.grid_forget()

            self.label_msp_t.grid_forget()
            self.spinbox_msp_t.grid_forget()

            self.label_msp_psi.grid_forget()
            self.spinbox_msp_psi.grid_forget()

            self.label_msp_anomaly_score_threshold.grid_forget()
            self.spinbox_msp_anomaly_score_threshold.grid_forget()


            self.label_method.config(text="Outlier Detection Method: " + self.selected_method)
            if self.selected_method == methods[0]:  # DBSCAN
                self.label_msp_epsilon.grid(row=6, column=0)
                self.spinbox_msp_epsilon.grid(row=6, column=1)
                self.label_msp_minpts.grid(row=7, column=0)
                self.spinbox_msp_minpts.grid(row=7, column=1)
                self.update()
            if self.selected_method == methods[1]:  # Isolation Forest
                self.label_msp_t.grid(row=6, column=0)
                self.spinbox_msp_t.grid(row=6, column=1)
                self.label_msp_psi.grid(row=7, column=0)
                self.spinbox_msp_psi.grid(row=7, column=1)
                self.label_msp_anomaly_score_threshold.grid(row=8, column=0)
                self.spinbox_msp_anomaly_score_threshold.grid(row=8, column=1)
                self.update()


        self.listbox_method.bind("<<ListboxSelect>>", on_select)

        self.label_num_dp.grid(row=0, column=0)
        self.spinbox_num_dp.grid(row=0, column=1)

        self.label_dr.grid(row=1, column=0)

        self.label_normal_pc.grid(row=2, column=0)
        self.spinbox_normal_pc.grid(row=2, column=1)
        self.label_percent_1.grid(row=2, column=2)

        self.label_equal_pc.grid(row=3, column=0)
        self.spinbox_equal_pc.grid(row=3, column=1)
        self.label_percent_2.grid(row=3, column=2)

        self.label_method.grid(row=4, column=0)

        self.listbox_method.grid(row=5, column=0, columnspan=2)

        self.init_values()
        self.root.mainloop()

    def update(self):
        ## Update mean
        self.replace(self.table[len(self.EDITABLE_MEASURES)][1], f"{np.mean(self.all_x):.4f}")
        self.replace(self.table[len(self.EDITABLE_MEASURES)][2], f"{np.mean(self.all_y):.4f}")

        # Update standard deviation
        self.replace(self.table[len(self.EDITABLE_MEASURES) + 1][1], f"{np.std(self.all_x):.4f}")
        self.replace(self.table[len(self.EDITABLE_MEASURES) + 1][2], f"{np.std(self.all_y):.4f}")

        # Calculate outliers
        self.OUTLIER_METHODS.get(self.selected_method)(self)

        # Update outlier percentage
        percentage = 100.0 * len(self.outlier_x) / int(self.spinbox_num_dp.get())
        self.replace(self.table[self.NUM_OF_MEASURES - 1][1], f"{percentage:.1f}")
        self.replace(self.table[self.NUM_OF_MEASURES - 1][2], f"{percentage:.1f}")

        # Filling table
        for i in range(self.NUM_OF_MEASURES, min(len(self.all_x), self.NUM_OF_MEASURES + self.MAX_ROWS_VALUES)):
            self.replace(self.table[i][1], f"{self.all_x[i]:.2f}")
            self.replace(self.table[i][2], f"{self.all_y[i]:.2f}")
            if self.all_x[i] in self.outlier_x and self.all_y[i] in self.outlier_y:
                color = self.OUTLIER_TEXT_COLOR
            else:
                color = self.VALID_POINT_COLOR
            self.table[i][1].config(fg=color)
            self.table[i][2].config(fg=color)

        ## Update graph
        # Redraw points
        self.scatter_plot_all.set_offsets(np.c_[self.all_x, self.all_y])
        self.scatter_plot_outlier.set_offsets(np.c_[self.outlier_x, self.outlier_y])
        self.ax.set_xlabel(self.table[0][1].get())
        self.ax.set_ylabel(self.table[0][2].get())
        self.canvas_graph.draw()

    def get_generated_values(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.concatenate([
            np.random.normal(loc=float(self.table[3][1].get()),
                             scale=float(self.table[4][1].get()),
                             size=
                             int(float(self.spinbox_num_dp.get()) * float(self.spinbox_normal_pc.get()) * 0.01)),
            np.random.uniform(low=float(self.table[1][1].get()),
                              high=float(self.table[2][1].get()),
                              size=int(
                                  float(self.spinbox_num_dp.get()) * float(self.spinbox_equal_pc.get()) * 0.01))])
        y = np.concatenate([
            np.random.normal(loc=float(self.table[3][2].get()),
                             scale=float(self.table[4][2].get()),
                             size=int(float(self.spinbox_num_dp.get()) * float(self.spinbox_normal_pc.get()) * 0.01)),
            np.random.uniform(low=float(self.table[1][2].get()),
                              high=float(self.table[2][2].get()),
                              size=int(
                                  float(self.spinbox_num_dp.get()) * float(self.spinbox_equal_pc.get()) * 0.01))])
        return x, y, np.concatenate([np.array([x]).T, np.array([y]).T], axis=1)

    def init_values(self):
        self.all_x, self.all_y, self.all_data = self.get_generated_values()
        self.update()


if __name__ == "__main__":
    app = Application()
