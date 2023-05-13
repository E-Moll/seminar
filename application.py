import math
import tkinter as tk
import customtkinter as ctk
import matplotlib.backends.backend_tkagg as tkagg
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn import *


class Application:
    # Constants
    NUM_DATAPOINTS_START = "30"
    STEP_SIZE_DP = "100"
    NORMAL_PERCENTAGE_START = "80"
    EDITABLE_MEASURES = ["Data Point Nr.", "Min (for eq. dist.)", "Max (for eq. dist.)", "µ", "σ"]
    INITIAL_VALUES = [["km/h", "# of cars"], [-10.0, 10.0, -10.0], [-10.0, 10.0, 10.0], [-10.0, 10.0, 1.5], [0.0, 10.0, 4.0]]  # [<MIN>, <MAX>, <INITIAL>]
    X_LIM = [-10.0, 10.0]
    Y_LIM = [-10.0, 10.0]
    STEP_SIZE_PERCENT = 2
    VALID_POINT_COLOR = "#00007f"
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

    def dbscan(self):
        eps = 0.8
        min_pts = 5

        # Record all distances between any two points that are at the most epsilon apart from each other
        distances = {}
        for i in range(len(self.all_data)):
            for j in range(i + 1, len(self.all_data)):
                # Calculate Distance
                euclidean_distance = math.sqrt(pow(self.all_x[j] - self.all_x[i], 2) + pow(self.all_y[j] - self.all_y[i], 2))
                # Check if the current two points are neighbors
                if euclidean_distance <= eps:
                    # If not already, save neighbors for each of the two points
                    for k, l in zip([i, j], [j, i]):
                        if tuple(self.all_data[k]) in distances:
                            distances[tuple(self.all_data[k])].append(self.all_data[l])
                        else:
                            distances.update({tuple(self.all_data[k]): [self.all_data[l]]})

        # Mark all those points as outliers that have less than min_pts neighbors TODO: outlier detection is currently inverted
        self.outlier_x = []
        self.outlier_y = []
        self.num_of_outliers = 0
        for item in distances.items():
            if len(item[1]) < min_pts:  # item[1] is a list of all neighbors in the epsilon range
                self.outlier_x.append(item[0][0])  # item[0][0] is the datapoint's x dimension
                self.outlier_y.append(item[0][1])  # item[0][1] is the datapoint's y dimension
                self.num_of_outliers += 1

    def other(self):
        # TODO
        self.outliers_indices = [i for i in range(len(self.all_x)) if abs(self.all_x[i] + self.all_y[i]) > 1.0]
        self.refill_coordinate_lists()

    OUTLIER_METHODS = {"DBSCAN": dbscan, "Other procedure": other}

    # Some important variables
    num_of_outliers = None
    outliers_indices = None
    all_data = None
    all_x = None
    all_y = None
    outlier_x = None
    outlier_y = None
    selected_method = list(OUTLIER_METHODS.keys())[0]

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
        self.scatter_plot_outlier = self.ax.scatter(x=self.all_x, y=self.all_y, s=self.OUTLIER_SIZE, c=self.OUTLIER_POINT_COLOR)
        self.ax.set_xlim(self.X_LIM[0], self.X_LIM[1])
        self.ax.set_ylim(self.Y_LIM[0], self.Y_LIM[1])
        self.ax.grid(True)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        ## UI general settings
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.geometry("1750x950+70+30")
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
                    c = tk.Spinbox(master=self.frame_table, from_=self.INITIAL_VALUES[row][0], to=self.INITIAL_VALUES[row][1], increment=0.05 * self.INITIAL_VALUES[row][1], justify=tk.CENTER, font=font)
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
        self.spinbox_num_dp = tk.Spinbox(master=self.frame_settings, from_=0, to=50_000, increment=self.STEP_SIZE_DP, justify=tk.CENTER)
        self.spinbox_num_dp.bind("<Button-1>", on_click_gen_new)
        self.replace(self.spinbox_num_dp, self.NUM_DATAPOINTS_START)

        self.label_dr = ctk.CTkLabel(master=self.frame_settings, text="Distribution ratio:", justify=tk.LEFT,
                                     text_color=self.LABEL_COLOR)

        self.label_normal_pc = ctk.CTkLabel(master=self.frame_settings, text="Normal:", justify=tk.LEFT,
                                            text_color=self.LABEL_COLOR)

        def on_normal_pc_change():
            self.replace(self.spinbox_equal_pc, str(100.0 - float(self.spinbox_normal_pc.get())))

        self.spinbox_normal_pc = tk.Spinbox(master=self.frame_settings, from_=0, to=100, increment=self.STEP_SIZE_PERCENT,
                                            justify=tk.CENTER, command=on_normal_pc_change)
        self.spinbox_normal_pc.bind("<Button-1>", on_click_gen_new)
        self.replace(self.spinbox_normal_pc, self.NORMAL_PERCENTAGE_START)
        self.label_percent_1 = ctk.CTkLabel(master=self.frame_settings, text="%", justify=tk.LEFT,
                                            text_color=self.LABEL_COLOR)

        self.label_equal_pc = ctk.CTkLabel(master=self.frame_settings, text="Equal:", justify=tk.LEFT,
                                           text_color=self.LABEL_COLOR)

        def on_equal_pc_change():
            self.replace(self.spinbox_normal_pc, str(100.0 - float(self.spinbox_equal_pc.get())))

        self.spinbox_equal_pc = tk.Spinbox(master=self.frame_settings, from_=0, to=100, increment=self.STEP_SIZE_PERCENT,
                                           justify=tk.CENTER, command=on_equal_pc_change)
        self.spinbox_equal_pc.bind("<Button-1>", on_click_gen_new)
        self.replace(self.spinbox_equal_pc, str(100 - int(self.NORMAL_PERCENTAGE_START)))
        self.label_percent_2 = ctk.CTkLabel(master=self.frame_settings, text="%", justify=tk.LEFT,
                                            text_color=self.LABEL_COLOR)

        self.label_method = tk.Label(master=self.frame_settings, text="Outlier Detection Method: " + self.selected_method, justify=tk.LEFT,
                                     fg=self.LABEL_COLOR)

        self.label_msp_quantile = ctk.CTkLabel(master=self.frame_settings, text="Quantile:",
                                               justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_quantile = tk.Spinbox(master=self.frame_settings, from_=0, to=1, increment=0.05, justify=tk.CENTER)
        self.spinbox_msp_quantile.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_quantile, "0.975")
        self.label_msp_neighbors = ctk.CTkLabel(master=self.frame_settings, text="K Neighbors:",
                                                justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_neighbors = tk.Spinbox(master=self.frame_settings, from_=1, to=3000, increment=3, justify=tk.CENTER)
        self.spinbox_msp_neighbors.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_neighbors, "5")
        self.label_msp_pc_outliers = ctk.CTkLabel(master=self.frame_settings, text="Percentage of Outliers:",
                                                  justify=tk.LEFT, text_color=self.LABEL_COLOR)
        self.spinbox_msp_pc_outliers = tk.Spinbox(master=self.frame_settings, from_=0, to=100, increment=self.STEP_SIZE_PERCENT, justify=tk.CENTER)
        self.spinbox_msp_pc_outliers.bind("<Button-1>", on_click)
        self.replace(self.spinbox_msp_pc_outliers, "20")

        self.listbox_method = tk.Listbox(master=self.frame_settings)
        methods = list(self.OUTLIER_METHODS.keys())
        self.listbox_method.insert(0, *methods)

        def on_select(event):
            a = self.listbox_method.selection_get()
            if a == "0":
                return
            self.selected_method = a
            self.label_msp_quantile.grid_forget()
            self.spinbox_msp_quantile.grid_forget()
            self.label_msp_neighbors.grid_forget()
            self.spinbox_msp_neighbors.grid_forget()
            self.label_msp_pc_outliers.grid_forget()
            self.spinbox_msp_pc_outliers.grid_forget()
            self.label_method.config(text="Outlier Detection Method: " + self.selected_method)
            if self.selected_method == methods[0]:  # Mahalanobis
                self.label_msp_quantile.grid(row=6, column=0)
                self.spinbox_msp_quantile.grid(row=6, column=1)
                self.update()
            if self.selected_method == methods[1]:  # Robust Distance
                self.label_msp_quantile.grid(row=6, column=0)
                self.spinbox_msp_quantile.grid(row=6, column=1)
                self.update()
            # if self.selected_method == methods[2]:  # Local Outlier Factor
            #     self.label_msp_neighbors.grid(row=6, column=0)
            #     self.spinbox_msp_neighbors.grid(row=6, column=1)
            #     self.update()
            # if self.selected_method == methods[2]:  # K Nearest Neighbor
            #     self.label_msp_neighbors.grid(row=6, column=0)
            #     self.spinbox_msp_neighbors.grid(row=6, column=1)
            #     self.label_msp_pc_outliers.grid(row=7, column=0)
            #     self.spinbox_msp_pc_outliers.grid(row=7, column=1)
            #     self.update()

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
        percentage = 100.0 * self.num_of_outliers / int(self.spinbox_num_dp.get())
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
