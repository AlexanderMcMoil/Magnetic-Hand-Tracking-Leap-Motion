from turtle import distance

import cv2
import time
# import mediapipe as mp
import numpy as np
import struct
import msvcrt
from utils.utils import *
import serial
import matplotlib.pyplot as plt
import leap
import numpy as np
import cv2
import threading
from sklearn import preprocessing, linear_model, neural_network, metrics, pipeline
from enum import Enum, auto


port_fes = "COM4"

latest_hand = None
data_lock = threading.Lock()

preset_clear = [
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
    [0, 0, 0, 0],  
]


def record_data(filename:str, connection:leap.Connection, duration:float=30, ser:serial.Serial=None):
    # cap = VideoCapture(0)
    if ser is None:
        ser = serial.Serial(port=serial_port, baudrate=baud_rate)
    file = open(filename, "w")
    time0 = time.time_ns()
    data:np.ndarray
    record_data = True

    canvas = Canvas()
    tracking_listener = TrackingListener(canvas)
    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    time_start = time.time_ns()
    prev_hand_pos = [0,0,0]
    running = True
    # try:
    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            canvas.set_tracking_mode(leap.TrackingMode.Desktop)

            while running:
                with data_lock:
                    hands = tracking_listener.hands
                if hands is not None and len(hands) > 0:
                    if (np.round(prev_hand_pos, 3) != np.round(list(hands[0].palm.position), 3)).all():
                        magnet_values = get_arduino_values(ser)


                        angles = get_angles(hands[0])
                        data = np.concatenate([magnet_values.flatten(), angles])

                        prev_hand_pos = list(hands[0].palm.position)
                        if record_data and angles[finger_angle_indices["index_mcp"]] != -1:
                            file.write(np.array2string(data.flatten(), max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
                if time.time_ns() - time_start > duration*1000000000:
                    running = False
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        # ser.close()
        file.close()
    finally:
        # ser.close()
        file.close()
    pass

def train_model(specific_neural_networks: dict = None, dataset_name: str = None):
    # thumb_dataset = 0
    
    models = {
        "thumb":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.001, max_iter=5000, validation_fraction=0.125, tol=1e-4))]),
        "index":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.001, max_iter=5000, validation_fraction=0.125, tol=1e-4))]),
        "middle":   pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.001, max_iter=5000, validation_fraction=0.125, tol=1e-4))]),
        "ring":     pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.001, max_iter=5000, validation_fraction=0.125, tol=1e-4))]),
        "pinky":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.001, max_iter=5000, validation_fraction=0.125, tol=1e-4))]),
        "wrist":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.001, max_iter=5000, validation_fraction=0.125, tol=1e-4))]),
    }
    fingers = ["thumb","index","middle","ring", "pinky"]
    thumb_dataset = np.genfromtxt("./datasets/thumb2.txt", delimiter=',')
    finger_dataset = np.genfromtxt("./datasets/all_fingers_10.txt", delimiter=',')
    wrist_dataset = np.genfromtxt("./datasets/wrist.txt", delimiter=',')
    x_finger = finger_dataset[:,:24]
    y_finger = finger_dataset[:,24:]
    x_wrist = wrist_dataset[:,:24]
    y_wrist = wrist_dataset[:,24:]
    
    y_finger = np.radians(y_finger)
    y_wrist = np.radians(y_wrist)

    model = Generic_Hand_Model(models)
    if specific_neural_networks is None:
        specific_neural_networks = {
            "thumb": "thumb2",
            "index": "index4",
            "middle": "middle6",
            "ring": "ring5",
            "pinky": "pinky4",
            "wrist": "wrist",
        }
    
    for finger, name in specific_neural_networks.items():
        if name == "train":
            if finger == "wrist":
                model.fit_finger(x_wrist, y_wrist, finger)
            elif finger == "thumb":
                model.fit_finger(thumb_dataset[:,:data_num], np.radians(thumb_dataset[:,data_num:]), finger)
            else:
                model.fit_finger(x_finger, y_finger, finger)
        else:
            model.models[finger] = load_model(name)
    return model
    pass

def send_fes_command(ser, channel, parameters):
    command = pack_serial_data(channel, parameters[0], parameters[1], parameters[2], parameters[3])
    ser.write(command)
    return

def pack_serial_data(channel, pulse_width, frequency, leading_amplitude, lagging_amplitude):
    """
    Packs 5 parameters into a 4-byte array for serial transmission.
    
    Bit layout (32 bits total):
    [31:24] : frequency (8 bits)
    [23:23] : Padding (1 bits) - set to 0
    [22:18] : lagging_amplitude (5 bits)
    [17:8]  : pulse_width (10 bits)
    [7:3]   : leading_amplitude (5 bits)
    [2:0]   : channel (3 bits)
    """
    
    # 1. Mask inputs to ensure they strictly fit within their bit limits
    channel &= 0x07             # 3 bits (max 7)
    leading_amplitude &= 0x1F   # 5 bits (max 31)
    pulse_width &= 0x3FF        # 10 bits (max 1023)
    frequency &= 0xFF           # 6 bits (max 63)
    lagging_amplitude &= 0x1F   # 5 bits (max 31)
    
    # 2. Shift and combine using bitwise OR
    packed_data = (channel) | \
                  (leading_amplitude << 3) | \
                  (pulse_width << 8) | \
                  (lagging_amplitude << 18) | \
                  (frequency << 24)
                  
    return packed_data.to_bytes(4, byteorder='big')

def pack_serial_data_multiground(channel, pulse_width, frequency, leading_amplitude, lagging_amplitude, signal_address, ground_address):
    """
    Packs 5 parameters into a 4-byte array for serial transmission.
    
    Bit layout (32 bits total):
    [33:32] : ground_address (2 bits)
    [31:29] : signal_address (3 bits)
    [29:24] : lagging_amplitude (5 bits)
    [23:18] : frequency (6 bits)
    [17:8]  : pulse_width (10 bits)
    [7:3]   : leading_amplitude (5 bits)
    [2:0]   : channel (3 bits)
    """

    # 1. Mask inputs to ensure they strictly fit within their bit limits
    channel &= 0x07             # 3 bits (max 7)
    leading_amplitude &= 0x1F   # 5 bits (max 31)
    pulse_width &= 0x3FF        # 10 bits (max 1023)
    frequency &= 0x3F           # 6 bits (max 63)
    lagging_amplitude &= 0x1F   # 5 bits (max 31)
    signal_address &= 0x07      # 3 bits (max 7)
    ground_address &= 0x03      # 2 bits (max 3)

    packed_data = (channel) | \
                  (leading_amplitude << 3) | \
                  (pulse_width << 8) | \
                  (frequency << 18) | \
                  (lagging_amplitude << 24) | \
                  (signal_address << 29) | \
                  (ground_address << 32)
    return packed_data.to_bytes(5, byteorder='big')

def send_fes_command_multiground(ser, parameters):
    command = pack_serial_data(*parameters)
    ser.write(command)
    return


def send_preset(ser, parameters):
    for i in range(len(parameters)):
        command = pack_serial_data(i, parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3])
        ser.write(command)
        # time.sleep(1.5)
    return

def calibrate_goal(ser_hand_tracking: serial.Serial, ser_fes: serial.Serial, model:Generic_Hand_Model, ideal_parameters:np.ndarray):
    send_fes_command(ser_fes, 0, ideal_parameters)
    time.sleep(5)
    magnet_values = get_arduino_values(ser_hand_tracking)
    angles = model.predict(magnet_values.reshape([1,-1])).flatten()
    return angles

def extension_condition(angles, extension_thresholds):
    if angles[finger_angle_indices["wrist"]] < extension_thresholds[finger_angle_indices["wrist"]]:
        return True
    return False

def flexion_condition(angles, flexion_thresholds):
    if angles[finger_angle_indices["wrist"]] > flexion_thresholds[finger_angle_indices["wrist"]] - 10:
        return True
    return False
flexion_angles = [41.21532597, 2.10172005, 1.61832444, 44.30577811, 68.13241936, 59.95652903,
                  45.84537721, 74.6872136, 65.72474797, 54.84501658, 96.02223953, 84.49957079,
                  58.1430314, 76.77909011, 67.5655993, 56.03340535]
def hand_metric(angles, flexion_angles):
    # Determine how extended the hand is
    # The higher the value, the more extended the hand is
    # distance = np.linalg.norm(angles - flexion_angles)
    return -1*angles[finger_angle_indices["wrist"]]
    # return flexion_angles[finger_angle_indices["wrist"]] - angles[finger_angle_indices["wrist"]] + (flexion_angles[finger_angle_indices["index_mcp"]] - angles[finger_angle_indices["index_mcp"]])/2 + (flexion_angles[finger_angle_indices["middle_mcp"]] - angles[finger_angle_indices["middle_mcp"]])/2 + (flexion_angles[finger_angle_indices["ring_mcp"]] - angles[finger_angle_indices["ring_mcp"]])/2 + (flexion_angles[finger_angle_indices["pinky_mcp"]] - angles[finger_angle_indices["pinky_mcp"]])/2 + (flexion_angles[finger_angle_indices["index_pip"]] - angles[finger_angle_indices["index_pip"]])/2 + (flexion_angles[finger_angle_indices["middle_pip"]] - angles[finger_angle_indices["middle_pip"]])/2 + (flexion_angles[finger_angle_indices["ring_pip"]] - angles[finger_angle_indices["ring_pip"]])/2 + (flexion_angles[finger_angle_indices["pinky_pip"]] - angles[finger_angle_indices["pinky_pip"]])/2 + (flexion_angles[finger_angle_indices["thumb_mcp"]] - angles[finger_angle_indices["thumb_mcp"]])
    # return distance

# class FesState(Enum):
#     INIT_WAIT = auto()
#     CALIBRATE = auto()
#     WAITING_EXTENSION = auto()
#     RAMP_EXTENSION = auto()
#     HOLDING_EXTENSION = auto() 
#     WAITING_FLEXION = auto()
#     HOLDING_FLEXION = auto()
#     DONE = auto()

# class FESControllerFSM:
#     def __init__(self, ser_hand_tracking, ser_fes, model, ideal_parameters, extension_thresholds):
#         self.ser_hand_tracking = ser_hand_tracking
#         self.ser_fes = ser_fes
#         self.model = model
#         self.ideal_parameters = ideal_parameters
#         self.extension_thresholds = extension_thresholds
#         self.angles = np.zeros(16)
        
#         # Internal variables
#         self.flexion_thresholds = None
#         self.reps = 0
#         self.max_reps = 1
        
#         # Durations
#         self.duration_wait_extension = 4.0
#         self.duration_full_extension = 3.0
#         self.duration_full_flexion = 3.0
#         self.init_wait_duration = 1.5
#         self.secondary_extension_ramp_time = 1.5
#         self.ramp_number = 1
        
#         # State tracking
#         self.state = FesState.INIT_WAIT
#         self.timer_start = time.time()
        
#         # Trigger the initial command (get hand at full flexion)
#         send_fes_command(self.ser_fes, 0, [0, 0, 0, 0])

#     def update(self):
#         """
#         Call this method continuously in your main loop.
#         It evaluates the current state and handles transitions non-blockingly.
#         """
#         magnet_values = get_arduino_values(self.ser_hand_tracking)
#         self.angles = self.model.predict(magnet_values.reshape([1, -1])).flatten()
#         if self.state == FesState.INIT_WAIT:
#             # Wait for 1.5 seconds to settle
#             if time.time() - self.timer_start >= self.init_wait_duration:
#                 self.state = FesState.WAITING_EXTENSION
#                 magnet_values = get_arduino_values(self.ser_hand_tracking)
#                 self.flexion_thresholds = self.model.predict(magnet_values.reshape([1, -1])).flatten()
#                 send_fes_command(self.ser_fes, 0, self.ideal_parameters)  # Trigger extension

#         elif self.state == FesState.WAITING_EXTENSION:
#             # Poll tracking until extension condition is met
#             # magnet_values = get_arduino_values(self.ser_hand_tracking)
#             # self.angles = self.model.predict(magnet_values.reshape([1, -1])).flatten()
            
#             if extension_condition(self.angles, self.extension_thresholds):
#                 self.timer_start = time.time()  # Reset timer for the hold
#                 self.state = FesState.HOLDING_EXTENSION
#                 print("Starting Extension Hold")
#             if time.time() - self.timer_start >= self.duration_wait_extension:
#                 self.state = FesState.RAMP_EXTENSION
#                 # self.timer_start = time.time()
#                 print("Starting Extension Ramp")
        
#         elif self.state == FesState.RAMP_EXTENSION:
#             # magnet_values = get_arduino_values(self.ser_hand_tracking)
#             # self.angles = self.model.predict(magnet_values.reshape([1, -1])).flatten()
            
#             if time.time() - self.timer_start >= self.secondary_extension_ramp_time:
#                 parameters = self.ideal_parameters
#                 parameters[2] += self.ramp_number
#                 parameters[3] += self.ramp_number
#                 send_fes_command(self.ser_fes, 0, parameters)
#                 self.timer_start = time.time()
#                 print(f"Ramping #{self.ramp_number}")
#                 self.ramp_number += 1
            
#             if extension_condition(self.angles, self.extension_thresholds):
#                 self.timer_start = time.time()  # Reset timer for the hold
#                 self.ramp_number = 1
#                 self.state = FesState.HOLDING_EXTENSION
#                 print("Holding Extension")

#         elif self.state == FesState.HOLDING_EXTENSION:
#             # Wait for duration_full_extension
#             if time.time() - self.timer_start >= self.duration_full_extension:
#                 # Action: Trigger flexion
#                 send_fes_command(self.ser_fes, 0, [0, 0, 0, 0])
#                 self.state = FesState.WAITING_FLEXION
#                 print("Starting Rest")

#         elif self.state == FesState.WAITING_FLEXION:
#             # Poll tracking until flexion condition is met
#             # magnet_values = get_arduino_values(self.ser_hand_tracking)
#             # self.angles = self.model.predict(magnet_values.reshape([1, -1])).flatten()
            
#             if flexion_condition(self.angles, self.flexion_thresholds):
#                 self.timer_start = time.time()  # Reset timer for the hold
#                 self.state = FesState.HOLDING_FLEXION

#         elif self.state == FesState.HOLDING_FLEXION:
#             # Wait for duration_full_flexion
#             if time.time() - self.timer_start >= self.duration_full_flexion:
#                 self.reps += 1
#                 print(f"Completed Rep {self.reps}")
                
#                 if self.reps < self.max_reps:
#                     # Action: Trigger next extension
#                     send_fes_command(self.ser_fes, 0, self.ideal_parameters)
#                     self.state = FesState.WAITING_EXTENSION
#                     self.timer_start = time.time()
#                     print("Starting Extension")
#                 else:
#                     self.state = FesState.DONE

#         elif self.state == FesState.DONE:
#             # Controller is finished, nothing left to do
#             pass

# def control_loop(ser_hand_tracking: serial.Serial, ser_fes: serial.Serial, model:Generic_Hand_Model, ideal_parameters:np.ndarray, extension_thresholds:np.ndarray):
#     # control loop will output to channel 0
#     # get the hand at full flexion
#     send_fes_command(ser_fes, 0, [0,0,0,0])
#     time.sleep(1.5)
#     magnet_values = get_arduino_values(ser_hand_tracking)
#     flexion_thresholds = model.predict(magnet_values.reshape([1,-1])).flatten()


#     duration_full_extension = 3
#     duration_full_flexion = 3
#     repititions = 5
#     reps = 0
#     controller = FESControllerFSM(ser_hand_tracking, ser_fes, model, ideal_parameters, extension_thresholds)
#     wrist_angles = []
#     state = []
#     time_start = time.time()
#     times = []
#     while controller.state != FesState.DONE:
#         controller.update()
#         wrist_angles.append(np.degrees(-controller.angles[finger_angle_indices["wrist"]]))
#         state.append(controller.state.value)
#         times.append(time.time() - time_start)
#         time.sleep(0.025)  # Small sleep to prevent CPU overuse
#     fig,ax = plt.subplots()
#     ax.plot(times, wrist_angles)
#     ax.set(xlabel="Time (s)", ylabel='Wrist Angle (deg.)')
#     print(state)
#     state = np.array(state)
#     plt.axhline(np.degrees(-extension_thresholds[finger_angle_indices["wrist"]]), color='r', label="Extension Threshold")
#     ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state==FesState.WAITING_EXTENSION.value), alpha=0.3, label="Initial Extension")
#     ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state==FesState.RAMP_EXTENSION.value), alpha=0.5, label="Extension Ramping")
#     ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state==FesState.HOLDING_EXTENSION.value), alpha=0.3, label="Extension Hold")
#     plt.legend()
#     plt.show()
#     # for i in range(repititions):
#     #     send_fes_command(ser_fes, 0, ideal_parameters)
#     #     while True:
#     #         magnet_values = get_arduino_values(ser_hand_tracking)
#     #         angles = model.predict(magnet_values.reshape([1,-1])).flatten()
#     #         if extension_condition(angles, extension_thresholds):
#     #             break
#     #     time.sleep(duration_full_extension)
#     #     send_fes_command(ser_fes, 0, [0,0,0,0])
#     #     while True:
#     #         magnet_values = get_arduino_values(ser_hand_tracking)
#     #         angles = model.predict(magnet_values.reshape([1,-1])).flatten()
#     #         if flexion_condition(angles, flexion_thresholds):
#     #             break
#     #     print(f"Completed Rep {i+1}")
#     #     time.sleep(duration_full_flexion)
#     pass
class FesState(Enum):
    INIT_WAIT = auto()
    INITIAL_RAMP_UP = auto()   # Ramps from 0 to ideal_parameters
    WAITING_EXTENSION = auto() # Waits for extension at ideal_parameters
    KICK_RAMP_UP = auto()      # Kicks parameters higher if extension fails
    HOLDING_EXTENSION = auto() 
    RAMP_DOWN = auto()         # Ramps down to 0
    WAITING_FLEXION = auto()
    HOLDING_FLEXION = auto()
    DONE = auto()

class FESControllerFSM:
    def __init__(self, ser_hand_tracking, ser_fes, model, ideal_parameters, extension_thresholds):
        self.ser_hand_tracking = ser_hand_tracking
        self.ser_fes = ser_fes
        self.model = model
        self.ideal_parameters = np.array(ideal_parameters, dtype=float)
        self.extension_thresholds = extension_thresholds
        self.angles = np.zeros(16)
        
        # Internal variables
        self.flexion_thresholds = None
        self.reps = 0
        self.max_reps = 5
        self.current_parameters = np.zeros(4)
        
        # Durations
        self.init_wait_duration = 1.5
        self.duration_wait_extension = 3.0
        self.duration_full_extension = 3.0
        self.duration_full_flexion = 3.0
        
        # Initial Ramp Up Settings (0 to ideal_parameters)
        self.initial_ramp_duration = 5
        self.initial_ramp_step_time = 0.25
        self.initial_ramp_total_steps = self.initial_ramp_duration / self.initial_ramp_step_time
        self.initial_ramp_step_size = np.array([0, 0, self.ideal_parameters[2]/self.initial_ramp_total_steps, self.ideal_parameters[3]/self.initial_ramp_total_steps])
        self.initial_ramp_steps_taken = 0
        
        # Kick Ramp Up Settings (Beyond ideal_parameters)
        self.kick_ramp_step_time = 1.5
        self.kick_ramp_increment = 1.0
        
        # Ramp Down Settings (current_parameters to 0)
        self.ramp_down_duration = 5
        self.ramp_down_step_time = 0.25
        self.ramp_down_total_steps = self.ramp_down_duration / self.ramp_down_step_time
        self.ramp_down_step_size = np.array([0, 0, self.ideal_parameters[2]/self.initial_ramp_total_steps, self.ideal_parameters[3]/self.initial_ramp_total_steps])
        self.down_steps_taken = 0
        
        # State tracking
        self.state = FesState.INIT_WAIT
        self.timer_start = time.time()
        
        # Trigger the initial command (get hand at full flexion)
        send_fes_command(self.ser_fes, 0, [0, 0, 0, 0])

    def update(self):
        # Update tracking angles
        magnet_values = get_arduino_values(self.ser_hand_tracking)
        self.angles = self.model.predict(magnet_values.reshape([1, -1])).flatten()
        
        if self.state == FesState.INIT_WAIT:
            if time.time() - self.timer_start >= self.init_wait_duration:
                self.flexion_thresholds = self.angles.copy()
                
                # Start initial ramp up
                self.current_parameters = np.zeros(4)
                self.initial_ramp_steps_taken = 0
                self.state = FesState.INITIAL_RAMP_UP
                self.timer_start = time.time()
                print("Starting Initial Ramp Up")

        elif self.state == FesState.INITIAL_RAMP_UP:
            # Catch early extension during the ramp
            if extension_condition(self.angles, self.extension_thresholds):
                self.timer_start = time.time()
                self.state = FesState.HOLDING_EXTENSION
                print("Extension reached early. Holding Extension")
                
            elif time.time() - self.timer_start >= self.initial_ramp_step_time:
                self.current_parameters = np.round(self.current_parameters + self.initial_ramp_step_size)
                self.initial_ramp_steps_taken += 1
                
                # Cap at ideal_parameters during this phase
                self.current_parameters = np.minimum(self.current_parameters, self.ideal_parameters)
                send_fes_command(self.ser_fes, 0, self.current_parameters)
                self.timer_start = time.time()
                print(f"Initial Ramp Step: {self.current_parameters}")
                
                if self.initial_ramp_steps_taken >= self.initial_ramp_total_steps:
                    self.state = FesState.WAITING_EXTENSION
                    self.timer_start = time.time()
                    print("At Ideal Parameters. Waiting for Extension")

        elif self.state == FesState.WAITING_EXTENSION:
            if extension_condition(self.angles, self.extension_thresholds):
                self.timer_start = time.time()
                self.state = FesState.HOLDING_EXTENSION
                print("Starting Extension Hold")
                
            elif time.time() - self.timer_start >= self.duration_wait_extension:
                self.state = FesState.KICK_RAMP_UP
                self.timer_start = time.time()
                print("Timeout. Starting Kick Ramp Up")
        
        elif self.state == FesState.KICK_RAMP_UP:
            if time.time() - self.timer_start >= self.kick_ramp_step_time:
                # Target specific channels for the kick (based on your previous code)
                self.current_parameters[2] += self.kick_ramp_increment
                self.current_parameters[3] += self.kick_ramp_increment
                send_fes_command(self.ser_fes, 0, self.current_parameters)
                
                self.timer_start = time.time()
                print(f"Kick Ramped Up to: {self.current_parameters}")
            
            if extension_condition(self.angles, self.extension_thresholds):
                self.timer_start = time.time()
                self.state = FesState.HOLDING_EXTENSION
                print("Holding Extension")

        elif self.state == FesState.HOLDING_EXTENSION:
            if time.time() - self.timer_start >= self.duration_full_extension:
                self.state = FesState.RAMP_DOWN
                self.timer_start = time.time()
                self.down_steps_taken = 0
                
                # Calculate dynamically based on wherever parameters ended up
                self.ramp_down_step_size = self.current_parameters / self.ramp_down_total_steps
                print("Starting Ramp Down")

        elif self.state == FesState.RAMP_DOWN:
            if time.time() - self.timer_start >= self.ramp_down_step_time:
                self.current_parameters -= self.ramp_down_step_size
                self.current_parameters = np.maximum(0, self.current_parameters)
                self.down_steps_taken += 1
                
                send_fes_command(self.ser_fes, 0, self.current_parameters)
                self.timer_start = time.time()
                
                if self.down_steps_taken >= self.ramp_down_total_steps:
                    send_fes_command(self.ser_fes, 0, [0, 0, 0, 0])
                    self.state = FesState.WAITING_FLEXION
                    print("Starting Rest (Waiting for Flexion)")

        elif self.state == FesState.WAITING_FLEXION:
            if flexion_condition(self.angles, self.flexion_thresholds):
                self.timer_start = time.time()
                self.state = FesState.HOLDING_FLEXION

        elif self.state == FesState.HOLDING_FLEXION:
            if time.time() - self.timer_start >= self.duration_full_flexion:
                self.reps += 1
                print(f"Completed Rep {self.reps}")
                
                if self.reps < self.max_reps:
                    self.current_parameters = np.zeros(4)
                    self.initial_ramp_steps_taken = 0
                    self.state = FesState.INITIAL_RAMP_UP
                    self.timer_start = time.time()
                    print("Starting Next Rep: Initial Ramp Up")
                else:
                    self.state = FesState.DONE

        elif self.state == FesState.DONE:
            pass


def control_loop(ser_hand_tracking, ser_fes, model, ideal_parameters, extension_thresholds):
    controller = FESControllerFSM(ser_hand_tracking, ser_fes, model, ideal_parameters, extension_thresholds)
    
    wrist_angles = []
    state_history = []
    times = []
    time_start = time.time()
    
    while controller.state != FesState.DONE:
        controller.update()
        
        # Assuming finger_angle_indices is defined in your broader scope
        wrist_angles.append(np.degrees(-controller.angles[finger_angle_indices["wrist"]]))
        state_history.append(controller.state.value)
        times.append(time.time() - time_start)
        
        time.sleep(0.025)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, wrist_angles, color='black', linewidth=1.5)
    ax.set(xlabel="Time (s)", ylabel='Wrist Angle (deg.)')
    
    state_history = np.array(state_history)
    plt.axhline(np.degrees(-extension_thresholds[finger_angle_indices["wrist"]]), color='r', linestyle='--', label="Extension Threshold")
    
    # Shade backgrounds for distinct phases
    ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state_history==FesState.INITIAL_RAMP_UP.value), alpha=0.3, color='cyan', label="Initial Ramp Up")
    ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state_history==FesState.WAITING_EXTENSION.value), alpha=0.3, color='blue', label="Wait Ext @ Ideal")
    ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state_history==FesState.KICK_RAMP_UP.value), alpha=0.5, color='purple', label="Kick Ramp Up")
    ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state_history==FesState.HOLDING_EXTENSION.value), alpha=0.3, color='green', label="Hold Ext")
    ax.fill_between(times, min(wrist_angles), max(wrist_angles), where=(state_history==FesState.RAMP_DOWN.value), alpha=0.5, color='orange', label="Ramp Down")
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.0))
    plt.tight_layout()
    plt.show()

def freq_tuning(ser_hand_tracking: serial.Serial, ser_fes: serial.Serial, model:Generic_Hand_Model, ideal_parameters:np.ndarray):
    freq = 10
    hand_metric_values = []
    lookback_window = 10
    for i in range(54):
        ideal_parameters[1] = freq
        send_fes_command(ser_fes, 0, ideal_parameters)
        time.sleep(0.3)
        magnet_values = get_arduino_values(ser_hand_tracking)
        angles = model.predict(magnet_values.reshape([1,-1])).flatten()
        print(f"Frequency: {freq} Hz, Wrist Angle: {np.degrees(angles[finger_angle_indices['wrist']]):.2f} degrees")
        hand_metric_value = hand_metric(np.degrees(angles), flexion_angles)
        hand_metric_values.append(hand_metric_value)
        
        if len(hand_metric_values) > lookback_window:
            if np.all(np.array(hand_metric_values[-lookback_window+1:]) - np.array(hand_metric_values[-lookback_window:-1]) < 0.5):
                print(np.array(hand_metric_values[-lookback_window+1:]) - np.array(hand_metric_values[-lookback_window:-1]))
                print(f"Frequency {freq} Hz is too high, stopping tuning.")
                break
            # print(f"Frequency {freq} Hz is too high, stopping tuning.")
            # break
        freq += 1
    print(f"Optimal Frequency: {freq - lookback_window} Hz")
    ideal_parameters[1] = freq - lookback_window

    send_fes_command(ser_fes, 0, [0,0,0,0])
    send_fes_command(ser_fes,0, ideal_parameters)
    # fig,ax = plt.subplots()
    # ax.plot(range(10, len(hand_metric_values)+10), hand_metric_values)
    # plt.axvline(x = freq-lookback_window, color = 'r', label = 'Optimal Frequency')
    # ax.set(xlabel="Freq (Hz)", ylabel="Wrist Angle (deg.)")
    # plt.legend()
    # plt.show()
    return freq - lookback_window

def amplitude_tuning(ser_hand_tracking: serial.Serial, ser_fes: serial.Serial, model:Generic_Hand_Model, ideal_parameters:np.ndarray):
    max_amplitude = ideal_parameters[2]
    max_freq = 255
    for i in range(1, max_amplitude):
        pass
    pass

if __name__ == "__main__":
    flexion_angles = [41.21532597, 2.10172005, 1.61832444, 44.30577811, 68.13241936, 59.95652903,
                       45.84537721, 74.6872136, 65.72474797, 54.84501658, 96.02223953, 84.49957079,
                       58.1430314, 76.77909011, 67.5655993, 56.03340535]
    # specific_neural_networks = {
    #     "thumb": "train",
    #     "index": "train",
    #     "middle": "train",
    #     "ring": "train",
    #     "pinky": "train",
    #     "wrist": "train",
    # }
    specific_neural_networks = {
        "thumb": "thumb2",
        "index": "index4",
        "middle": "middle6",
        "ring": "ring5",
        "pinky": "pinky4",
        "wrist": "wrist",
    }
    ser_hand_tracking = serial.Serial(port=arduino_port, baudrate=baud_rate)
    ser_fes = serial.Serial(port=port_fes, baudrate=baud_rate)
    ideal_parameters = [400, 20, 15, 15]
    model = train_model(specific_neural_networks)
    print("Capturing flexion angles")
    magnet_values = get_arduino_values(ser_hand_tracking)
    angles = model.predict(magnet_values.reshape([1,-1])).flatten()
    print(f"Flexion angles: {np.degrees(angles)}")
    try:
        freq = freq_tuning(ser_hand_tracking, ser_fes, model, ideal_parameters=ideal_parameters)
        ideal_parameters[1] = freq
        extension_angles = calibrate_goal(ser_hand_tracking, ser_fes, model, ideal_parameters=ideal_parameters)
        control_loop(ser_hand_tracking, ser_fes, model, ideal_parameters, extension_angles)
        pass
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # record_data(filename="data_collection/data.csv", connection=leap.Connection(), duration=30, ser=None)
        # time.sleep(3)
        send_preset(ser_fes, preset_clear)
    pass
