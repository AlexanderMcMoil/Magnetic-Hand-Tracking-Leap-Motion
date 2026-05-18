from utils.utils import *
from sklearn import tree
import serial
import matplotlib.pyplot as plt
# import pybullet as p
# import pybullet_data
from sklearn import preprocessing, linear_model, neural_network, metrics, pipeline

def main():
    # cap = cv2.VideoCapture(0)
    # model = train_model()
    ser = serial.Serial(arduino_port, baudrate=baud_rate)
    # magnet_record = [[],[],[],[]]
    # fig = plt.figure(1)
    time.sleep(2)
    # names = ["index_mcp", "index_pip"]
    finger = "thumb"
    # model = train_model(finger)
    model = train_generic_hand_model()
    while True:
        # __, frame = cap.read()
        magnet_values = get_arduino_values(ser)
        angles = model.predict(magnet_values.reshape([1,-1])).flatten() * rad2deg
        angles_bullet = np.concatenate([[angles[15]], [angles[0]], [0], 2*angles[1:3], [0], angles[3:6], [0], angles[6:9], [0], angles[9:12], [0], angles[12:15]])
    return

def train_model():

    pass

def train_generic_hand_model():
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
    finger_dataset = np.genfromtxt("./datasets/all_fingers_7.txt", delimiter=',')
    wrist_dataset = np.genfromtxt("./datasets/wrist.txt", delimiter=',')
    from scipy.signal import savgol_filter
    print(finger_dataset.shape)
    x_finger = finger_dataset[:,:24]
    y_finger = finger_dataset[:,24:]
    x_wrist = wrist_dataset[:,:24]
    y_wrist = wrist_dataset[:,24:]

    # maximum = np.max(y_finger, axis=0)
    # minimum = np.min(y_finger, axis=0)
    # # print(minimum)
    # for finger in ("index","middle","ring","pinky"):
    #     y_finger[:, finger_angle_indices[f"{finger}_mcp"]] = 90*(y_finger[:, finger_angle_indices[f"{finger}_mcp"]] - minimum[finger_angle_indices[f"{finger}_mcp"]])/maximum[finger_angle_indices[f"{finger}_mcp"]]
    #     y_finger[:, finger_angle_indices[f"{finger}_pip"]] = 110*(y_finger[:, finger_angle_indices[f"{finger}_mcp"]] - minimum[finger_angle_indices[f"{finger}_pip"]])/maximum[finger_angle_indices[f"{finger}_pip"]]
    # file = open("./datasets/fingers_adjusted.txt", "w")
    # finger_dataset_adjusted = np.hstack([x_finger, y_finger])
    # for i in range(finger_dataset_adjusted.shape[0]):
    #     file.write(np.array2string(finger_dataset_adjusted[i,:], max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
    
    y_finger = np.radians(y_finger)
    y_wrist = np.radians(y_wrist)

    model = Generic_Hand_Model(models)
    # y_finger[:, finger_angle_indices["index_mcp"]] = savgol_filter(y_finger[:, finger_angle_indices["index_mcp"]], 60, 2)
    # y_finger[:, finger_angle_indices["index_pip"]] = savgol_filter(y_finger[:, finger_angle_indices["index_pip"]], 60, 2)
    # model.fit_finger(thumb_dataset[:,:data_num], np.radians(thumb_dataset[:,data_num:]), "thumb")
    # model.models["thumb"] = load_model(f"thumb2")

    # model.fit_finger(wrist_dataset[:,:data_num], np.radians(wrist_dataset[:,data_num:]), "wrist")
    # model.models["wrist"] = load_model(f"wrist")

    
    # y_finger[:, finger_angle_indices["index_mcp"]] =  y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["index_pip"]] =  y_finger[:, finger_angle_indices["index_pip"]]
    # y_finger[:, finger_angle_indices["middle_mcp"]] = y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["middle_pip"]] = y_finger[:, finger_angle_indices["index_pip"]]
    # y_finger[:, finger_angle_indices["ring_mcp"]] =   y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["ring_pip"]] =   y_finger[:, finger_angle_indices["index_pip"]]
    # y_finger[:, finger_angle_indices["pinky_mcp"]] =   y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["pinky_pip"]] =   y_finger[:, finger_angle_indices["index_pip"]]
    

    # index_dataset = np.genfromtxt("./datasets/test/index_test.txt", delimiter=',')
    # x_index = index_dataset[:,:20]
    # x_index = x_index[:, [finger_angle_indices["index_mcp"], finger_angle_indices["index_pip"]]]
    # y_index = index_dataset[:,20:]

    
    
    # model.fit_finger(x_finger, y_finger, "index")
    # model.fit_finger(x_finger, y_finger, "middle")
    # model.fit_finger(x_finger, y_finger, "ring")
    # model.fit_finger(x_finger, y_finger, "pinky")
    # model.fit_finger(x_wrist, y_wrist, "wrist")
    # model.fit_finger()
    
    # model.models["index"] = load_model(f"index3")
    # model.models["middle"] = load_model(f"middle6")
    # model.models["ring"] = load_model(f"ring5")
    # model.models["pinky"] = load_model(f"pinky4")
    # model.models["wrist"] = load_model(f"wrist")
    
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


if __name__ == "__main__":
    main()
    pass