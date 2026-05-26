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

port_fes = "COM8"

latest_hand = None
data_lock = threading.Lock()

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
        "thumb":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.01, max_iter=1000))]),
        "index":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.01, max_iter=1000))]),
        "middle":   pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.01, max_iter=1000))]),
        "ring":     pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.01, max_iter=1000))]),
        "pinky":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.01, max_iter=1000))]),
        "wrist":    pipeline.Pipeline([("scaling", preprocessing.MinMaxScaler()),("clf", neural_network.MLPRegressor([200, 200, 200], activation='relu', learning_rate_init=0.01, max_iter=1000))]),
    }
    fingers = ["thumb","index","middle","ring", "pinky"]
    thumb_dataset = np.genfromtxt("./datasets/thumb2.txt", delimiter=',')
    finger_dataset = np.genfromtxt("./datasets/all_fingers_8.txt", delimiter=',')
    wrist_dataset = np.genfromtxt("./datasets/wrist.txt", delimiter=',')
    x_finger = finger_dataset[:,:24]
    y_finger = finger_dataset[:,24:]
    
    y_finger = np.radians(y_finger)

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
            model.fit_finger(x_finger, y_finger, finger)
        else:
            model.models[finger] = load_model(name)
    return model
    pass

def send_fes_command(ser, channel, parameters):
    command = f"{channel} {str(np.int32(parameters))[1:-1]}\n"
    ser.write(command.encode("utf-8"))
    return

def send_preset(ser, parameters):
    for i in range(len(parameters)):
        command = f"preset {str(np.int32(parameters))[1:-1]}\n"
        ser.write(command.encode("utf-8"))
        time.sleep(1.5)
    return

def calibrate_goal(ser_hand_tracking: serial.Serial, ser_fes: serial.Serial, model:Generic_Hand_Model, ideal_parameters:np.ndarray):
    send_fes_command(ser_fes, 0, ideal_parameters)
    time.sleep(5)
    magnet_values = get_arduino_values(ser_hand_tracking)
    angles = model.predict(magnet_values)
    return angles

def extension_condition(angles, extension_thresholds):
    if angles[finger_angle_indices["wrist"]] < extension_thresholds[finger_angle_indices["wrist"]] + 10:
        return True
    return False

def flexion_condition(angles, flexion_thresholds):
    if angles[finger_angle_indices["wrist"]] > flexion_thresholds[finger_angle_indices["wrist"]] - 10:
        return True
    return False



def control_loop(ser_hand_tracking: serial.Serial, ser_fes: serial.Serial, model:Generic_Hand_Model, ideal_parameters:np.ndarray, extension_thresholds:np.ndarray):
    # control loop will output to channel 0
    # get the hand at full flexion
    send_fes_command(ser_fes, 0, [0,0,0,0])
    time.sleep(1.5)
    magnet_values = get_arduino_values(ser_hand_tracking)
    flexion_thresholds = model.predict(magnet_values)


    duration_full_extension = 5
    duration_full_flexion = 5
    repititions = 5
    reps = 0

    for i in range(repititions):
        send_fes_command(ser_fes, 0, ideal_parameters)
        while True:
            magnet_values = get_arduino_values(ser_hand_tracking)
            angles = model.predict(magnet_values)
            if extension_condition(angles, extension_thresholds):
                break
        time.sleep(duration_full_extension)
        send_fes_command(ser_fes, 0, [0,0,0,0])
        while True:
            magnet_values = get_arduino_values(ser_hand_tracking)
            angles = model.predict(magnet_values)
            if flexion_condition(angles, flexion_thresholds):
                break
        print(f"Completed Rep {i+1}")
        time.sleep(duration_full_flexion)
    pass

if __name__ == "__main__":
    specific_neural_networks = {
        "thumb": "thumb2",
        "index": "train",
        "middle": "train",
        "ring": "train",
        "pinky": "train",
        "wrist": "wrist",
    }
    # ser_hand_tracking = serial.Serial(port=arduino_port, baudrate=baud_rate)
    # ser_fes = serial.Serial(port=port_fes, baudrate=baud_rate)
    # ideal_parameters = np.array([500, 50, 5, 5])
    model = train_model(specific_neural_networks)

    # extension_angles = calibrate_goal(ser_hand_tracking, ser_fes, model, ideal_parameters=ideal_parameters)
    # control_loop(ser_hand_tracking, ser_fes, model, ideal_parameters, extension_angles)

    # record_data(filename="data_collection/data.csv", connection=leap.Connection(), duration=30, ser=None)

    pass
