import math
import os
import sys
import json
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit, QComboBox, QRadioButton
from networkx import bellman_ford_path_length
from PyQt5 import QtGui, QtCore, QtWidgets
import tabulate

from PersonalModules.Genetic import genetic_algorithm
from PersonalModules.UCB_VND import UCB_VND
from PersonalModules.Upper_Confidence_Bound import plot_histogram
from PersonalModules.VND import Variable_Neighborhood_Descent
from PersonalModules.greedy import greedy_algorithm
from PersonalModules.utilities import bellman_ford, display, get_stat, len_free_slots, len_sinked_relays, sentinel_relay
from PersonalModules.vns import Variable_Neighborhood_Search
from main import calculate_X, calculate_Y, create2, get_ordinal_number, save, save2

def get_Diameter(sentinel_bman, cal_bman, mesh_size):
    sentinel_relays = sentinel_relay(sentinel_bman)
    if 999 in sentinel_bman:
        return cal_bman/mesh_size
    else:
        return max(sentinel_relays)

def createEverything(chosen_grid, sink_location, mesh_size):
    free_slots = []

    grid = chosen_grid * mesh_size

    if sink_location == 'Center':
        if (chosen_grid % 2) == 0:
            sink = ((grid / 2) + 10, (grid / 2) - 10)
        elif (chosen_grid % 2) == 1:
            sink = (((grid - mesh_size) / 2) + 10, ((grid - mesh_size) / 2) + 10)
    elif sink_location == 'Top Left':
        sink = (mesh_size*2 + mesh_size/2, grid - (mesh_size*2 + mesh_size/2))
    elif sink_location == 'Top Right':
        sink = (grid - (mesh_size*2 + mesh_size/2), grid - (mesh_size*2 + mesh_size/2))
    elif sink_location == 'Bottom Left':
        sink = (mesh_size*2 + mesh_size/2, mesh_size*2 + mesh_size/2)       
    elif sink_location == 'Bottom Right':
        sink = (grid - (mesh_size*2 + mesh_size/2), mesh_size*2 + mesh_size/2)            

    mesh_number = ((sink[0] - 10) // 20) + ((sink[1] - 10) // 20) * chosen_grid + 1
    print(f'Mesh number: {mesh_number}')
    print(f'Sink coordinate: {sink}')

    # Create sentinels
    sinkless_sentinels = [(x, 10) for x in range(10, grid + 10, 20)] + \
                        [(x, grid - 10) for x in range(10, grid + 10, 20)] + \
                        [(10, y) for y in range(30, grid - 10, 20)] + \
                        [(grid - 10, y) for y in range(30, grid - 10, 20)]

    # Create the free slots
    for x in range(30, grid - 10, 20):
        for y in range(30, grid - 10, 20):
            if sink != (x, y):
                free_slots.append((x, y))

    return grid, sink, sinkless_sentinels, free_slots

class MyApplication(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def clear_output(self):
        self.output_text.clear()  # Clear the content of the output_text QTextEdit widget

    def init_ui(self):
        self.setWindowTitle('Algorithm VNS')
        self.setGeometry(100, 100, 1000, 800)

        #Window Icon
        self.setWindowIcon(QtGui.QIcon('C:/Users/nouri/OneDrive/Desktop/Papers/LOGO3.png'))

        # Widgets
        self.grid_size_label = QLabel('Grid Size:')
        self.grid_size_input = QLineEdit()
        self.mesh_size_label = QLabel('Mesh Size:')
        self.mesh_size_input = QLineEdit()
        self.mesh_size_input.setPlaceholderText('20')
        self.range_label = QLabel('Communication Range:')
        self.range_input = QLineEdit()
        self.range_input.setPlaceholderText('30')
        self.sens_range_label = QLabel('Sensing Range:')
        self.sens_range_input = QLineEdit()
        self.sens_range_input.setPlaceholderText('15')

        self.execution_type_label = QLabel('Execution Type:')
        
        self.execution_type_radio_button_0 = QRadioButton('One time VNS')
        self.execution_type_radio_button_1 = QRadioButton('Multiple times VNS')
        self.execution_type_radio_button_2 = QRadioButton('Load a Grid')
        self.execution_type_radio_button_0.setChecked(True)  # Set default selection

        self.Number_of_executions_label = QLabel('Number of executions:')
        self.Number_of_executions_input = QLineEdit()
        self.Number_of_executions_input.setPlaceholderText('1')
        
        self.Sink_location_label = QLabel('Do you want the sink to be in the middle: (Yes or write the index of the mesh)')
        self.Sink_location_label.move(50, 50)
        self.Sink_location_input = QComboBox(self)
        self.Sink_location_input.addItems(["Center", "Top Left", "Top Right", "Bottom Left", "Bottom Right"])

        self.alpha_lable = QLabel('Enter the value of alpha (Importance of number of relays nodes deployed):')
        self.alpha_input = QLineEdit()
        self.alpha_input.setPlaceholderText('0.5')

        self.beta_lable = QLabel('Enter the value of Beta (Importance of diameter of the network):')
        self.beta_input = QLineEdit()
        self.beta_input.setPlaceholderText('0.5')

        self.folder_path = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions"
        self.files = os.listdir(self.folder_path)

        self.file_label = QLabel("Choose a file to load:", self)
        self.file_label.move(50, 50)

        self.file_combo = QComboBox(self)
        self.file_combo.addItems(self.files)

        self.run_button = QPushButton('Run')
        self.run_button.clicked.connect(self.run_application)

        self.output_text = QTextEdit()
        self.output_text.setMinimumWidth(550)
        
        self.clear_button = QPushButton('Clear Output')
        self.clear_button.clicked.connect(self.clear_output)

        # Layout
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.grid_size_label)
        input_layout.addWidget(self.grid_size_input)
        input_layout.addWidget(self.mesh_size_label)
        input_layout.addWidget(self.mesh_size_input)
        input_layout.addWidget(self.range_label)
        input_layout.addWidget(self.range_input)
        input_layout.addWidget(self.sens_range_label)
        input_layout.addWidget(self.sens_range_input)
        input_layout.addWidget(self.execution_type_label)
        input_layout.addWidget(self.execution_type_radio_button_0)
        input_layout.addWidget(self.execution_type_radio_button_1)
        input_layout.addWidget(self.execution_type_radio_button_2)
        input_layout.addWidget(self.Number_of_executions_label)
        input_layout.addWidget(self.Number_of_executions_input)
        input_layout.addWidget(self.Sink_location_label)
        input_layout.addWidget(self.Sink_location_input)
        input_layout.addWidget(self.alpha_lable)
        input_layout.addWidget(self.alpha_input)
        input_layout.addWidget(self.beta_lable)
        input_layout.addWidget(self.beta_input)
        input_layout.addWidget(self.file_label)
        input_layout.addWidget(self.file_combo)
        input_layout.addWidget(self.run_button)
        
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel('Output:'))
        output_layout.addWidget(self.output_text)
        output_layout.addWidget(self.clear_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addLayout(output_layout)

        self.setLayout(main_layout)

        # Applying Styles        
        input_fields = [self.grid_size_input, self.mesh_size_input, self.range_input, self.sens_range_input,
                        self.Number_of_executions_input, self.alpha_input, self.beta_input]
        for field in input_fields:
            field.setStyleSheet("background-color: #f0f0f0; color: #333333; font-size: 16px;")
        self.run_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.clear_button.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))


        self.run_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px;")
        self.clear_button.setStyleSheet("background-color: #cccccc; color: black; font-size: 16px;")
        self.output_text.setStyleSheet("background-color: #f0f0f0; color: #333333; font-size: 16px;")
    
    def run_application(self):
        chosen_grid = int(self.grid_size_input.text())
        mesh_size = int(self.mesh_size_input.text() or 20)
        custom_range = int(self.range_input.text() or 30)
        sensing_range = int(self.sens_range_input.text() or 15)
        user_input = int(self.Number_of_executions_input.text() or 1)
        sink_location = self.Sink_location_input.currentText()
        alpha = float(self.alpha_input.text() or 0.5)
        beta = float(self.beta_input.text() or 0.5) 

        get_in = True
        # Create everything
        if get_in:
            grid, sink, sinkless_sentinels, free_slots = createEverything(chosen_grid, sink_location, mesh_size)
            max_hops_number = grid
            sentinel_relay = []
            self.output_text.append("Everything got created Succesfully !\n")

        if self.execution_type_radio_button_0.isChecked():
            self.output_text.append("You chose One time VNS ! \n")

            self.output_text.append("\n   Starting Genetic algorithm...")
            start_time = time.time()

            sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = genetic_algorithm(10, 15, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range, mesh_size)
            #sinked_sentinels, sinked_relays, free_slots, Finished, ERROR = greedy_algorithm(sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range)
            self.output_text.append("   Genetic algorithm finished execution successfully !")

            # Get the performance before VNS, perform VNS then Get the performance after VNS
            self.output_text.append("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                sinked_sentinels)
            performance_before, relays_before, hops_before = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size, alpha, beta) 
            diameter_before = get_Diameter(sentinel_bman, cal_bman, mesh_size) 
            relays_before = len_sinked_relays(sinked_relays)                    
            self.output_text.append("   Calculations are done !")

            self.output_text.append(f'\n Fitness BEFORE: {performance_before}')
            self.output_text.append(f"\n Network diameter BEFORE: {diameter_before}")
            self.output_text.append(f'\n Relays BEFORE: {relays_before}')
            self.output_text.append(f'\n Hops BEFORE: {hops_before}')

            display(grid, sink, sinked_relays, sinked_sentinels, title="Genetic Algorithm")

            self.output_text.append("\n   Starting Variable Neighborhood Descent algorithm...")
            sinked_relays, free_slots = UCB_VND(grid, sink, sinked_sentinels, sinked_relays,
                                                                 free_slots, custom_range, mesh_size, lmax=5, alpha=alpha, beta=beta)
            self.output_text.append("   Variable Neighborhood Descent algorithm finished execution successfully !")

            self.output_text.append("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays, sinked_sentinels)
            performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size, alpha, beta)
            #diameter_after = round(cal_bman / mesh_size)
            diameter_after = get_Diameter(sentinel_bman, cal_bman, mesh_size)
            relays_after = len_sinked_relays(sinked_relays)
            self.output_text.append("   Calculations are done !")
            end_time = time.time()
            total_time = int(end_time - start_time)
            hours, remainder = divmod(total_time, 3600)
            minutes, remainder = divmod(remainder, 60)
            time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

            self.output_text.append(f"\nFitness BEFORE: {performance_before}")
            self.output_text.append(f"Fitness AFTER: {performance_after}\n")

            self.output_text.append(f"Network diameter BEFORE: {diameter_before}")
            self.output_text.append(f"Network diameter AFTER: {diameter_after}\n")

            self.output_text.append(f"Relays BEFORE: {relays_before}")
            self.output_text.append(f"Relays AFTER: {relays_after}\n")

            self.output_text.append(f"Hops Average BEFORE: {hops_before}")
            self.output_text.append(f"Hops Average AFTER: {hops_after}\n")

            self.output_text.append(f'Execution time: {time_string}\n')

            display(grid, sink, sinked_relays, sinked_sentinels, title="VND Algorihtm")

            self.output_text.append("\n Another execution ! \n")

            print(f'\nThe final solution: {len_sinked_relays(sinked_relays)} relays deployed: {sinked_relays}')
            print(f'The final solution: {len_free_slots(grid, sinked_relays)} free slots remaining\n\n')
            print(f'The final sentinel list solution: {sentinel_bman}')

        elif self.execution_type_radio_button_1.isChecked():
            Solutions_Data = []
            executions = 1
            original_free_slots = free_slots[:]
            
            vns_avg_hops = 0
            vns_avg_relays = 0
            vns_avg_performance = 0
            vns_avg_diameter = 0
            ga_avg_hops = 0
            ga_avg_relays = 0
            ga_avg_performance = 0
            ga_avg_diameter = 0

            self.output_text.append("You chose Multiple times VNS !\n")

            # Generate the initial solution using greedy algorithm
            self.output_text.append("\n   Starting Genetic algorithm...")
        
            '''genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots, Finished, ERROR = genetic_algorithm(100, 10, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range)
            Greedy_grid_data = [grid, sink, genetic_sinked_relays, genetic_sinked_sentinels]
            self.output_text.append("   Genetic algorithm finished execution successfully !")

            # Get the performance before VNS, perform VNS then Get the performance after VNS
            self.output_text.append("\n   Please wait until some calculations are finished...")
            distance_bman, sentinel_bman, genetic_cal_bman = bellman_ford(grid, genetic_free_slots, sink, genetic_sinked_relays,
                                                                genetic_sinked_sentinels)
            performance_before, relays_before, hops_before = get_stat(genetic_sinked_relays, sentinel_bman, genetic_cal_bman, grid, genetic_free_slots, sink, genetic_sinked_sentinels, alpha, beta)
            self.output_text.append("   Calculations are done !")

            self.output_text.append(f"\n Network diameter BEFORE: {round(genetic_cal_bman / mesh_size)}")

            display(grid, sink, genetic_sinked_relays, genetic_sinked_sentinels, title="Genetic Algorihtm")'''

            start_time = time.time()

            while executions <= user_input:
                grid, sink, sinkless_sentinels, free_slots = createEverything(chosen_grid, sink_location, mesh_size)
                genetic_sinked_sentinels, genetic_sinked_relays, genetic_free_slots, Finished, ERROR = genetic_algorithm(3, 10, sink, sinkless_sentinels, free_slots, max_hops_number+1, custom_range, mesh_size)
                
                #Greedy_grid_data = [grid, sink, genetic_sinked_relays, genetic_sinked_sentinels]
                self.output_text.append("   Genetic algorithm finished execution successfully !")

                # Get the performance before VNS, perform VNS then Get the performance after VNS
                self.output_text.append("\n   Please wait until some calculations are finished...")
                distance_bman, sentinel_bman, genetic_cal_bman = bellman_ford(grid, genetic_free_slots, sink, genetic_sinked_relays,
                                                                    genetic_sinked_sentinels)
                performance_before, relays_before, hops_before = get_stat(genetic_sinked_relays, sentinel_bman, genetic_cal_bman, grid, genetic_free_slots, sink, genetic_sinked_sentinels, mesh_size, alpha, beta)
                diameter_before = get_Diameter(sentinel_bman, genetic_cal_bman, mesh_size)
                self.output_text.append("   Calculations are done !")

                self.output_text.append(f"\n Network diameter BEFORE: {performance_before}")

                display(grid, sink, genetic_sinked_relays, genetic_sinked_sentinels, title="Genetic Algorihtm")

                sinked_relays = genetic_sinked_relays
                sinked_sentinels = genetic_sinked_sentinels
                free_slots = genetic_free_slots

                ga_diameter = diameter_before
                ga_avg_hops += hops_before
                ga_avg_relays += relays_before
                ga_avg_performance += performance_before
                ga_avg_diameter += ga_diameter

                self.output_text.append(f"\n # This is the {get_ordinal_number(executions)} VNS grid execution.")

                self.output_text.append("\n   Starting Variable Neighborhood Descent algorithm...")
                sinked_relays, free_slots = UCB_VND(grid, sink, sinked_sentinels, sinked_relays,
                                                                        free_slots, custom_range, mesh_size, lmax=5, alpha=alpha, beta=beta)
                VNS_grid_data = [grid, sink, sinked_relays, sinked_sentinels]
                self.output_text.append("   Variable Neighborhood Descent algorithm finished execution successfully !")

                self.output_text.append("\n   Please wait until some calculations are finished...")
                distance_bman, sentinel_bman, cal_bman = bellman_ford(grid, free_slots, sink, sinked_relays,
                                                                    sinked_sentinels)
                performance_after, relays_after, hops_after = get_stat(sinked_relays, sentinel_bman, cal_bman, grid, free_slots, sink, sinked_sentinels, mesh_size, alpha, beta)
                diameter_after = get_Diameter(sentinel_bman, cal_bman, mesh_size)
                relays_after = len_sinked_relays(sinked_relays)
                self.output_text.append("   Calculations are done !")

                display(grid, sink, sinked_relays, sinked_sentinels, title=f"{get_ordinal_number(executions)} VND Algorihtm")

                self.output_text.append(f"\nFitness BEFORE: {performance_before}")
                self.output_text.append(f"Fitness AFTER: {performance_after}\n")
                
                self.output_text.append(f"Relays BEFORE: {relays_before}")
                self.output_text.append(f"Relays AFTER: {relays_after}\n")

                self.output_text.append(f"Network diameter BEFORE: {diameter_before}")
                self.output_text.append(f"Network diameter AFTER: {diameter_after}\n")

                self.output_text.append(f"Hops Average BEFORE: {hops_before}")
                self.output_text.append(f"Hops Average AFTER: {hops_after}\n")

                vns_avg_hops += hops_after
                vns_avg_relays += relays_after
                vns_avg_performance += performance_after
                vns_avg_diameter += diameter_after

                executions = executions + 1

            # Aftermath
            stop = False

            self.output_text.append('\n\nSimulation Results:\n')

            self.output_text.append('GA Results AVERAGE:')

            self.output_text.append(f'Relays AVERAGE: {math.ceil(ga_avg_relays / user_input)}')
            self.output_text.append(f'Hops AVERAGE: {math.ceil(ga_avg_hops / user_input)}')
            self.output_text.append(f'Performance AVERAGE: {ga_avg_performance / user_input}')
            self.output_text.append(f'Diameter AVERAGE: {math.ceil(ga_avg_diameter / user_input)}')
            
            self.output_text.append('\nVNS Results AVERAGE:')

            self.output_text.append(f'Relays AVERAGE: {math.ceil(vns_avg_relays / user_input)}')
            self.output_text.append(f'Hops AVERAGE: {math.ceil(vns_avg_hops / user_input)}')
            self.output_text.append(f'Performance AVERAGE: {vns_avg_performance / user_input}')
            self.output_text.append(f'Diameter AVERAGE: {math.ceil(vns_avg_diameter / user_input)}')

            end_time = time.time()
            total_time = int(end_time - start_time)
            hours, remainder = divmod(total_time, 3600)
            minutes, remainder = divmod(remainder, 60)
            time_string = f"{hours:02.0f}H_{minutes:02.0f}M_{remainder:02.0f}S"

            avg_execution_time = total_time / user_input
            avg_hours, avg_remainder = divmod(avg_execution_time, 3600)
            avg_minutes, avg_remainder = divmod(avg_remainder, 60)
            avg_time_string = f"{avg_hours:02.0f}H_{avg_minutes:02.0f}M_{avg_remainder:02.0f}S"

            self.output_text.append(f'\nExecution time AVERAGE: {avg_time_string}')
            self.output_text.append(f'Total execution time: {time_string}')

        elif self.execution_type_radio_button_2.isChecked():
            '''folder_path = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions"
            files = os.listdir(folder_path)
            i = 0
            for file in files:
                i += 1
                print(f"\nFile {i}: {file}")'''

            user_input = self.file_combo.currentText()

            filepath = "C:/Users/nouri/OneDrive/Desktop/Papers/Solutions/" + user_input

            self.output_text.append(filepath)

            # Load the variable from the file
            with open(filepath, "r") as f:
                Solutions_Data = json.load(f)
            '''stop = False
            while not stop:
                user_input = int(input(
                    "\nENTER:\n   1 to display the grid data table.\n   2 to STOP the program.\n   3 Display a saved grid."))
                if user_input == 1:
                    Solutions = []
                    for i in range(len(Solutions_Data)):
                        Solutions.append(
                            [Solutions_Data[i][0], Solutions_Data[i][1], Solutions_Data[i][2], Solutions_Data[i][3],
                            Solutions_Data[i][4], Solutions_Data[i][5], Solutions_Data[i][6], Solutions_Data[i][7],
                            Solutions_Data[i][8]])
                    headers_var = ["Executions", "Grid", "Time spent", "Greedy performance", "VNS performance",
                                "Greedy total relays", "VNS total relays", "Greedy Average hops", "VNS Average hops"]
                    print(tabulate(Solutions, headers=headers_var, showindex=False, tablefmt="rounded_outline"))

                elif user_input == 2:
                    stop = True
                    print("\n           PROGRAM STOPPED !")
                elif user_input == 3:
                    user_input = int(input("Type The executions number to display it's grid."))
                    user_input2 = int(input("Show Greedy (Type 0) or VNS (Type 1) ?"))
                    does_num_executions_exist = False
                    for i in range(len(Solutions_Data)):
                        if user_input == Solutions_Data[i][0]:
                            Data = Solutions_Data[i][9][user_input2]
                            display(Data[0], Data[1], Data[2], Data[3])
                            does_num_executions_exist = True
                    if not does_num_executions_exist:
                        print("\n   /!\ INVALID INPUT OR NUMBER OF HOPS DOESN'T EXIST PLEASE TRY AGAIN /!\ ")'''


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myapp = MyApplication()
    myapp.show()
    sys.exit(app.exec_())
