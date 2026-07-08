from utils.utils import *
from sklearn import tree
import serial
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from sklearn import preprocessing, linear_model, neural_network, metrics, pipeline
from scipy.signal import savgol_filter

# port = "COM18"
# baud_rate = 115200

data_lock = threading.Lock()


def display_hand_angles(robot_id, angles):
    # angles_bullet = np.concatenate([[0], [0.5, 0], angles[1:3], [0], angles[3:6], [0], angles[6:9], [0], angles[9:12], [0], angles[12:]])
    # print(angles_bullet.__len__())
    angles_bullet = np.radians(angles)
    p.setJointMotorControlArray(robot_id, jointIndices=range(21), controlMode=p.POSITION_CONTROL, targetPositions=angles_bullet)
    for i in range(50):
        p.stepSimulation()
    return

def main(save_data=False, filename="./datasets/comparison_data.txt"):
    # cap = cv2.VideoCapture(0)
    # model = train_model()
    ser = serial.Serial(arduino_port, baudrate=baud_rate)
    start_time = 0
    duraton = 0
    # magnet_record = [[],[],[],[]]
    # fig = plt.figure(1)
    time.sleep(2)
    names = ["index_mcp", "index_pip"]
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical mode
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_path = "./modelling/human_hand-master/human_hand-master/model/meshes/human_hand_scaled.urdf"
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    robot_id_2 = p.loadURDF(urdf_path, [-1, 0, 0], useFixedBase=1)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-48, cameraTargetPosition=[-0.5,0,0])
    finger = "thumb"
    # model = train_model(finger)
    model = train_generic_hand_model()
    plot = Fast_Magnet_Display()
    # try:
    predicted_angles = []
    ground_truth = []
    record_angles = False

    canvas = Canvas()

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)
    t0 = time.time_ns()
    prev_hand_pos = [0,0,0]
    running = True
    if save_data:
        file = open(filename, "w")
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
                        
                        ground_truth_angles = get_angles(hands[0])

                        angles = model.predict(magnet_values.reshape([1,-1])).flatten() * rad2deg
                        data = np.concatenate([magnet_values.flatten(), angles])
                        data_comparison = np.concatenate([ground_truth_angles, angles])
                        angles_bullet = np.concatenate([[angles[15]], [angles[0]], [0], 2*angles[1:3], [0], angles[3:6], [0], angles[6:9], [0], angles[9:12], [0], angles[12:15]])
                        angles_bullet_2 = np.concatenate([[ground_truth_angles[15]], [ground_truth_angles[0]], [0], 2*ground_truth_angles[1:3], [0], ground_truth_angles[3:6], [0], ground_truth_angles[6:9], [0], ground_truth_angles[9:12], [0], ground_truth_angles[12:15]])
                        display_hand_angles(robot_id=robot_id, angles=angles_bullet)
                        display_hand_angles(robot_id=robot_id_2, angles=angles_bullet_2)
                        prev_hand_pos = list(hands[0].palm.position)
                        if record_angles:
                            # ground_truth[0].append(ground_truth_angles[finger_angle_indices["wrist"]])
                            ground_truth.append(ground_truth_angles)
                            predicted_angles.append(angles)
                            # predicted_angles[0].append(angles[finger_angle_indices["wrist"]])
                            if save_data:
                                file.write(np.array2string(data_comparison.flatten(), max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
                            # ground_truth[1].append(ground_truth_angles[finger_angle_indices["middle_pip"]])
                            # predicted_angles[1].append(angles[finger_angle_indices["middle_pip"]])
                        
                cv2.imshow(canvas.name, canvas.output_image)
                
                key = cv2.waitKey(35)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord(' '):
                    record_angles = not record_angles
                    print(f"{'Not ' if not record_angles else ''}Recording")
                    if record_angles:
                        start_time = time.time()
                    else:
                        duration = time.time() - start_time
                        print(f"Recording duration: {duration:.2f} seconds")
                        means = np.mean(np.abs(np.array(ground_truth) - np.array(predicted_angles)), axis=0)
                        print(means)
                        print(np.mean(means[[finger_angle_indices["index_mcp"], finger_angle_indices["index_pip"], finger_angle_indices["middle_mcp"], finger_angle_indices["middle_pip"], finger_angle_indices["ring_mcp"], finger_angle_indices["ring_pip"], finger_angle_indices["pinky_mcp"], finger_angle_indices["pinky_pip"]]]))


    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        ser.close()
        # if save_data:
        #     file.close()
    finally:
        ser.close()
        if save_data:
            file.close()    
    # p.disconnect()
    # mse = metrics.mean_squared_error(ground_truth[0], predicted_angles[0])
    # print(f"MSE: {mse}")
    ser.close()
    cv2.destroyAllWindows()
    plt.close()
    ground_truth = np.array(ground_truth)
    predicted_angles = np.array(predicted_angles)
    testing_data = np.hstack([ground_truth, predicted_angles])
    # file.write(np.array2string(testing_data, max_line_width=100000, separator=",").replace(" ", ""))
    fig = plt.figure(2)
    figure_titles = ["mcp", "pip"]
    for i in range(2):
        ax = fig.add_subplot(1,2,i+1)
        # ground_truth_norm = savgol_filter(ground_truth[:,finger_angle_indices[f"index_{figure_titles[i]}"]], 10, 3)
        ground_truth_norm = ground_truth[:,finger_angle_indices[f"index_{figure_titles[i]}"]]
        # ground_truth_norm = ground_truth[i]
        # predicted_angles_norm = savgol_filter(predicted_angles[i], 10, 3)
        predicted_angles_norm = predicted_angles[:,finger_angle_indices[f"index_{figure_titles[i]}"]]
        ax.plot(np.linspace(0, duration,len(ground_truth[:,finger_angle_indices[f"index_{figure_titles[i]}"]])), ground_truth_norm, c='Blue',label="Ground Truth")
        ax.plot(np.linspace(0,duration,len(predicted_angles[:,finger_angle_indices[f"index_{figure_titles[i]}"]])), predicted_angles_norm, c='Red', label="Predicted Angle")
        # ax.set_title(figure_titles[i])
        ax.set_ylabel(f"Wrist angle (deg.)")
        plt.margins(x=0)
        plt.legend(loc='upper left')
        if i == 0:
            ax.set_xlabel('Time (s)')
    # plt.legend( )
    plt.show()
    # except Exception as e:
    #     print(e)
    # finally:
    #     p.disconnect()
    #     ser.close()

    return

def test_speeds(save_data=False, filename="./datasets/comparison_data.txt"):
    # cap = cv2.VideoCapture(0)
    # model = train_model()
    ser = serial.Serial(arduino_port, baudrate=baud_rate)
    start_time = 0
    duraton = 0
    # magnet_record = [[],[],[],[]]
    # fig = plt.figure(1)
    time.sleep(2)
    names = ["index_mcp", "index_pip"]
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical mode
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_path = "./modelling/human_hand-master/human_hand-master/model/meshes/human_hand_scaled.urdf"
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    robot_id_2 = p.loadURDF(urdf_path, [-1, 0, 0], useFixedBase=1)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-48, cameraTargetPosition=[-0.5,0,0])
    finger = "thumb"
    # model = train_model(finger)
    model = train_generic_hand_model()
    plot = Fast_Magnet_Display()
    # try:
    predicted_angles = []
    ground_truth = []
    record_angles = False

    canvas = Canvas()

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)
    t0 = time.time_ns()
    prev_hand_pos = [0,0,0]
    running = True
    if save_data:
        file = open(filename, "w")
    tracked_errors = []
    try:
        with connection.open():
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            canvas.set_tracking_mode(leap.TrackingMode.Desktop)

            while running:
                # with data_lock:
                hands = tracking_listener.hands
                if hands is not None and len(hands) > 0:
                    if (np.round(prev_hand_pos, 3) != np.round(list(hands[0].palm.position), 3)).all():
                        magnet_values = get_arduino_values(ser)
                        
                        ground_truth_angles = get_angles(hands[0])

                        angles = model.predict(magnet_values.reshape([1,-1])).flatten() * rad2deg
                        data = np.concatenate([magnet_values.flatten(), angles])
                        data_comparison = np.concatenate([ground_truth_angles, angles])
                        angles_bullet = np.concatenate([[angles[15]], [angles[0]], [0], 2*angles[1:3], [0], angles[3:6], [0], angles[6:9], [0], angles[9:12], [0], angles[12:15]])
                        angles_bullet_2 = np.concatenate([[ground_truth_angles[15]], [ground_truth_angles[0]], [0], 2*ground_truth_angles[1:3], [0], ground_truth_angles[3:6], [0], ground_truth_angles[6:9], [0], ground_truth_angles[9:12], [0], ground_truth_angles[12:15]])
                        display_hand_angles(robot_id=robot_id, angles=angles_bullet)
                        display_hand_angles(robot_id=robot_id_2, angles=angles_bullet_2)
                        prev_hand_pos = list(hands[0].palm.position)
                        if record_angles:
                            # ground_truth[0].append(ground_truth_angles[finger_angle_indices["wrist"]])
                            ground_truth.append(ground_truth_angles)
                            predicted_angles.append(angles)
                            # predicted_angles[0].append(angles[finger_angle_indices["wrist"]])
                            if save_data:
                                file.write(np.array2string(data_comparison.flatten(), max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
                            # ground_truth[1].append(ground_truth_angles[finger_angle_indices["middle_pip"]])
                            # predicted_angles[1].append(angles[finger_angle_indices["middle_pip"]])
                if record_angles:
                    duration = time.time() - start_time
                    # cv2.putText(canvas.output_image, f"{duration:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                    cv2.putText(
                        canvas.output_image,
                        f"{duration:2f}s",
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 44),
                        1,
                    )
                cv2.imshow(canvas.name, canvas.output_image)
                
                key = cv2.waitKey(35)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord(' '):
                    record_angles = not record_angles
                    print(f"{'Not ' if not record_angles else ''}Recording")
                    if record_angles:
                        start_time = time.time()
                    else:
                        duration = time.time() - start_time
                        print(f"Recording duration: {duration:.2f} seconds")
                        means = np.mean(np.abs(np.array(ground_truth) - np.array(predicted_angles)), axis=0)
                        # print(means)
                        means = means[[finger_angle_indices["thumb_mcp"], finger_angle_indices["thumb_pip"],finger_angle_indices["index_mcp"], finger_angle_indices["index_pip"], finger_angle_indices["middle_mcp"], finger_angle_indices["middle_pip"], finger_angle_indices["ring_mcp"], finger_angle_indices["ring_pip"], finger_angle_indices["pinky_mcp"], finger_angle_indices["pinky_pip"], finger_angle_indices["wrist"]]]
                        # print(np.mean(means[[finger_angle_indices["thumb_mcp"], finger_angle_indices["thumb_pip"],finger_angle_indices["index_mcp"], finger_angle_indices["index_pip"], finger_angle_indices["middle_mcp"], finger_angle_indices["middle_pip"], finger_angle_indices["ring_mcp"], finger_angle_indices["ring_pip"], finger_angle_indices["pinky_mcp"], finger_angle_indices["pinky_pip"], finger_angle_indices["wrist"]]]))
                        tracked_errors.append((duration, means))
                        ground_truth = []
                        predicted_angles = []
                        



    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        ser.close()
        # if save_data:
        #     file.close()
    finally:
        ser.close()
        if save_data:
            file.close()    
    # p.disconnect()
    # mse = metrics.mean_squared_error(ground_truth[0], predicted_angles[0])
    # print(f"MSE: {mse}")
    ser.close()
    cv2.destroyAllWindows()
    plt.close()
    ground_truth = np.array(ground_truth)
    predicted_angles = np.array(predicted_angles)
    testing_data = np.hstack([ground_truth, predicted_angles])
    
    print(tracked_errors)
    for i in range(len(tracked_errors)):
        # plt.plot([x[0] for x in tracked_errors], [x[1][i] for x in tracked_errors], label=list(finger_angle_indices.keys())[i])
        errors_rounded= [ '%.2f' % elem for elem in tracked_errors[i][1]]
        print(f"{tracked_errors[i][0]:.2f}s: {errors_rounded}")
    return

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
    thumb_dataset = np.genfromtxt("./datasets/thumb3.txt", delimiter=',')
    finger_dataset = np.genfromtxt("./datasets/all_fingers_11.txt", delimiter=',')
    wrist_dataset = np.genfromtxt("./datasets/wrist.txt", delimiter=',')
    # from scipy.signal import savgol_filter
    x_finger = finger_dataset[:,:ring_num*receiver_num]
    y_finger = finger_dataset[:,ring_num*receiver_num:]
    x_wrist = wrist_dataset[:,:24]
    y_wrist = wrist_dataset[:,24:]

    model = Generic_Hand_Model(models)
    
    # y_finger[:, finger_angle_indices["index_mcp"]] = savgol_filter(y_finger[:, finger_angle_indices["index_mcp"]], 60, 2)
    # y_finger[:, finger_angle_indices["index_pip"]] = savgol_filter(y_finger[:, finger_angle_indices["index_pip"]], 60, 2)
    # model.fit_finger(thumb_dataset[:,:20], thumb_dataset[:,20:], "thumb")
    # model.models["thumb"] = load_model(f"thumb2")

    # y_finger[:, finger_angle_indices["index_mcp"]] =  y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["index_pip"]] =  y_finger[:, finger_angle_indices["index_pip"]]
    # y_finger[:, finger_angle_indices["middle_mcp"]] = y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["middle_pip"]] = y_finger[:, finger_angle_indices["index_pip"]]
    # y_finger[:, finger_angle_indices["ring_mcp"]] =   y_finger[:, finger_angle_indices["index_mcp"]]
    # y_finger[:, finger_angle_indices["ring_pip"]] =   y_finger[:, finger_angle_indices["index_pip"]]

    # index_dataset = np.genfromtxt("./datasets/test/index_test.txt", delimiter=',')
    # x_index = index_dataset[:,:20]
    # x_index = x_index[:, [finger_angle_indices["index_mcp"], finger_angle_indices["index_pip"]]]
    # y_index = index_dataset[:,20:]
    
    # model.fit_finger(x_finger, y_finger, "index")
    # model.fit_finger(x_finger, y_finger, "middle")
    # model.fit_finger(x_finger, y_finger, "ring")

    # model.models["index"] = load_model(f"index")
    # model.models["middle"] = load_model(f"middle")
    # model.models["ring"] = load_model(f"ring")
    # model.models["pinky"] = load_model(f"pinky")
    # model.models["wrist"] = load_model(f"wrist")
    specific_neural_networks = {
        "thumb": "thumb",
        "index": "index",
        "middle": "middle",
        "ring": "ring",
        "pinky": "pinky",
        "wrist": "wrist",
    }
    # specific_neural_networks = {
    #     "thumb": "train",
    #     "index": "train",
    #     "middle": "train",
    #     "ring": "train",
    #     "pinky": "train",
    #     "wrist": "train",
    # }
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
    return model


def evaluate_recording(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    ground_truth = dataset[:,:16]
    predicted_angles = dataset[:,16:]
    # predicted_angles = model.predict(x) * rad2deg
    # ground_truth = y * rad2deg
    absolute_error = np.abs(ground_truth - predicted_angles)
    confidence = np.sort(absolute_error, axis=0)[int(0.95*absolute_error.shape[0]),:]
    mean_absolute_error = np.mean(np.abs(ground_truth - predicted_angles), axis=0)
    mse = metrics.mean_squared_error(ground_truth, predicted_angles)
    for name, index in finger_angle_indices.items():
        print(f"{name.ljust(10)}: {mean_absolute_error[index]:.2f}\t {confidence[index]:.2f} deg")
        # print(f"{angle_names[index]}: Mean Absolute Error: {mean_absolute_error[index]:.2f} deg, 95th Percentile Confidence: {confidence[index]:.2f} deg, MSE: {mse:.2f}")
    print(f"Mean Absolute Error: {mean_absolute_error}")
    print(f"95th Percentile Confidence: {confidence}")
    print(f"MSE: {mse}")

def evaluate_dataset(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    
    x = dataset[:,:data_num]
    y = dataset[:,data_num:]
    # predicted_angles = model.predict(x) * rad2deg
    # ground_truth = y * rad2deg
    model = train_generic_hand_model()
    predicted = model.predict(x) * rad2deg
    absolute_error = np.abs(y - predicted)
    confidence = np.sort(absolute_error, axis=0)[int(0.95*absolute_error.shape[0]),:]
    mean_absolute_error = np.mean(np.abs(y - predicted), axis=0)
    mse = metrics.mean_squared_error(y, predicted)
    for name, index in finger_angle_indices.items():
        print(f"{name.ljust(10)}: {mean_absolute_error[index]:.2f}\t {confidence[index]:.2f} deg")
        # print(f"{angle_names[index]}: Mean Absolute Error: {mean_absolute_error[index]:.2f} deg, 95th Percentile Confidence: {confidence[index]:.2f} deg, MSE: {mse:.2f}")
    print(f"Mean Absolute Error: {mean_absolute_error}")
    print(f"95th Percentile Confidence: {confidence}")
    print(f"MSE: {mse}")

def parse_speeds(angles:str):
    angles = angles.split("\n")
    angles = [x for x in angles if len(x) > 0]
    speeds = []
    for angle in angles:
        speed, values = angle.split("s: ")
        speed = float(speed)
        values = values[1:-1].split(", ")
        values = [float(x[1:-1]) for x in values]
        # print(values)
        # values = [float(x) for x in values]
        speeds.append((np.array([speed] + values)))
    speeds = np.array(speeds)
    fast = speeds[speeds[:, 0] < 1.5]
    med = speeds[(speeds[:, 0] >= 1.5) & (speeds[:, 0] < 6)]
    slow = speeds[speeds[:, 0] >= 6]
    fast_mean = np.mean(fast, axis=0)
    med_mean = np.mean(med, axis=0)
    slow_mean = np.mean(slow, axis=0)
    print(fast)
    print(med)
    print(slow)
    print(f"Fast mean: {fast_mean}")
    print(f"Medium mean: {med_mean}")
    print(f"Slow mean: {slow_mean}")
    joint_names = ["thumb_mcp", "thumb_pip", "index_mcp", "index_pip", "middle_mcp", "middle_pip", "ring_mcp", "ring_pip", "pinky_mcp", "pinky_pip", "wrist"]
    print("\tSlow\tMedium\tFast")
    results = ""
    for i in range(1, speeds.shape[1]):
        print(f"{joint_names[i-1]:<15}: {slow_mean[i]:.2f}\t{med_mean[i]:.2f}\t{fast_mean[i]:.2f}")
        results += f"{joint_names[i-1]}: {slow_mean[i]:.2f}\t{med_mean[i]:.2f}\t{fast_mean[i]:.2f}\n"

    # print(speeds)
    return results

finger_speeds = """1.72s: ['2.27', '3.25', '4.50', '5.21', '5.60', '6.81', '7.14', '6.29', '8.16', '6.26', '13.07']
    1.53s: ['3.28', '1.64', '4.91', '5.15', '6.18', '8.13', '8.15', '6.40', '7.94', '6.90', '12.28']
    1.74s: ['2.95', '1.57', '5.39', '5.30', '6.53', '6.95', '7.95', '6.77', '6.59', '5.44', '12.57']
    1.62s: ['3.15', '1.02', '4.82', '4.06', '7.06', '8.12', '8.59', '6.13', '8.19', '6.32', '12.64']
    1.54s: ['3.79', '1.18', '4.86', '4.63', '6.40', '7.23', '7.83', '6.28', '7.93', '6.74', '11.89']
    1.15s: ['3.81', '1.22', '6.72', '6.71', '8.16', '9.87', '10.20', '8.31', '9.00', '8.36', '11.80']
    1.15s: ['3.78', '1.66', '7.81', '7.29', '8.56', '9.75', '10.80', '9.22', '8.90', '7.54', '12.32']
    1.31s: ['4.57', '2.98', '6.42', '6.70', '8.39', '9.92', '9.69', '8.91', '7.61', '7.46', '12.30']
    1.42s: ['4.78', '1.95', '5.66', '5.73', '7.57', '9.87', '9.13', '8.27', '8.47', '7.28', '12.08']
    1.27s: ['5.36', '2.75', '6.33', '7.03', '7.87', '9.31', '9.01', '7.98', '8.47', '7.81', '12.11']
    1.18s: ['4.62', '2.19', '6.60', '7.78', '9.40', '11.39', '10.65', '8.69', '9.37', '8.84', '12.19']
    1.36s: ['5.29', '3.01', '6.09', '7.29', '7.51', '10.67', '9.53', '8.05', '9.30', '8.44', '11.59']
    1.34s: ['6.01', '2.77', '6.39', '7.66', '8.27', '10.76', '10.72', '9.00', '9.00', '8.70', '11.12']
    1.37s: ['5.03', '1.93', '5.82', '5.90', '6.83', '9.20', '9.20', '8.56', '9.59', '8.63', '10.81']
    1.37s: ['6.12', '2.87', '5.38', '6.16', '7.29', '9.90', '9.86', '7.99', '7.50', '6.90', '10.57']
    1.34s: ['6.33', '3.10', '5.59', '7.03', '7.67', '9.93', '10.30', '8.27', '9.04', '8.15', '10.64']
    1.24s: ['9.05', '3.16', '5.75', '7.69', '7.73', '10.74', '9.85', '8.80', '8.99', '8.03', '10.88']
    1.23s: ['7.63', '2.48', '5.83', '6.40', '8.09', '12.08', '10.62', '9.06', '9.81', '9.08', '10.86']
    1.25s: ['7.70', '3.26', '6.09', '7.39', '7.76', '11.00', '9.87', '8.75', '7.76', '7.09', '10.16']
    1.25s: ['7.59', '3.96', '5.54', '7.38', '7.78', '10.58', '9.95', '7.47', '8.26', '6.92', '10.61']
    1.13s: ['7.40', '3.48', '7.83', '11.26', '9.51', '13.26', '11.67', '10.93', '9.99', '8.97', '10.24']
    1.23s: ['7.57', '3.09', '5.90', '6.45', '8.23', '11.25', '9.81', '8.51', '9.17', '7.78', '10.06']
    2.74s: ['6.97', '3.62', '4.42', '5.16', '4.82', '7.58', '5.99', '5.57', '5.86', '5.31', '9.13']
    2.77s: ['6.68', '3.05', '3.85', '4.94', '4.31', '6.75', '5.91', '4.77', '4.97', '4.08', '9.97']
    2.78s: ['6.31', '2.43', '3.99', '4.47', '4.46', '5.95', '6.31', '5.48', '5.49', '4.81', '9.77']
    2.67s: ['6.34', '3.21', '4.30', '4.42', '4.86', '6.00', '5.75', '5.30', '4.69', '4.08', '9.78']
    4.59s: ['6.64', '2.47', '3.92', '3.90', '3.08', '4.80', '4.50', '4.10', '3.56', '2.82', '8.90']
    2.64s: ['6.58', '2.08', '3.78', '4.04', '4.80', '6.33', '6.05', '5.43', '5.71', '4.68', '10.36']
    3.18s: ['6.50', '1.83', '4.93', '5.19', '4.41', '6.16', '5.88', '5.55', '3.60', '3.12', '9.34']
    3.52s: ['6.75', '1.13', '3.41', '3.64', '4.31', '5.54', '4.68', '4.14', '4.16', '3.87', '9.66']
    2.65s: ['7.02', '2.53', '4.16', '3.34', '5.20', '6.12', '5.52', '4.83', '6.03', '4.96', '9.91']
    3.28s: ['6.83', '1.97', '3.72', '4.58', '3.78', '6.58', '4.59', '4.42', '5.47', '5.07', '9.67']
    2.96s: ['6.99', '1.67', '4.05', '5.10', '3.85', '5.52', '5.66', '5.36', '5.11', '4.92', '8.59']
    3.61s: ['7.14', '1.38', '3.56', '4.76', '3.85', '5.33', '4.29', '5.00', '5.31', '4.92', '8.40']
    2.82s: ['7.25', '2.34', '4.12', '4.42', '4.51', '5.98', '5.76', '4.62', '5.42', '4.59', '9.32']
    2.89s: ['7.18', '1.64', '4.01', '4.71', '4.35', '6.46', '5.52', '4.90', '5.41', '4.82', '8.94']
    3.45s: ['7.62', '1.76', '3.83', '4.47', '4.07', '6.14', '4.79', '4.65', '5.25', '5.10', '9.62']
    3.23s: ['7.80', '1.61', '4.16', '5.35', '3.71', '5.99', '4.93', '5.00', '5.51', '4.83', '9.54']
    3.53s: ['6.58', '1.95', '3.71', '3.98', '3.97', '6.74', '6.00', '5.94', '5.79', '5.51', '9.68']
    2.40s: ['6.74', '1.94', '4.37', '7.01', '4.91', '6.98', '6.48', '5.38', '5.16', '5.56', '9.15']
    3.31s: ['7.35', '1.33', '3.87', '6.32', '3.69', '6.77', '4.31', '4.14', '3.87', '4.33', '9.53']
    7.84s: ['6.78', '1.46', '3.88', '7.22', '2.97', '4.72', '5.36', '4.81', '3.27', '2.97', '7.98']
    10.64s: ['7.13', '1.77', '3.44', '5.95', '3.19', '4.97', '5.81', '5.48', '5.16', '4.64', '9.47']
    16.82s: ['7.07', '2.22', '2.98', '4.03', '3.52', '4.81', '3.42', '2.98', '3.25', '3.19', '10.93']
    7.96s: ['6.85', '2.17', '2.81', '4.83', '3.05', '3.96', '3.91', '3.57', '2.47', '2.22', '9.55']
    8.94s: ['6.26', '2.47', '3.42', '4.60', '3.50', '3.68', '3.77', '3.01', '3.32', '2.76', '8.53']
    6.07s: ['5.37', '2.72', '2.39', '2.05', '2.74', '2.60', '3.48', '2.35', '1.97', '2.82', '8.94']
    8.66s: ['5.65', '3.59', '3.62', '4.51', '3.94', '3.53', '2.87', '3.54', '3.92', '2.69', '6.90']
    9.93s: ['6.10', '3.82', '3.34', '5.18', '4.29', '4.59', '3.75', '4.57', '4.62', '2.87', '8.30']
    7.03s: ['5.32', '4.50', '4.00', '5.66', '3.69', '3.02', '3.19', '3.00', '4.78', '2.33', '8.56']
    6.53s: ['5.29', '2.87', '3.75', '4.38', '3.68', '5.24', '3.40', '3.67', '3.94', '3.44', '9.80']
    6.00s: ['6.13', '2.07', '3.82', '4.72', '3.90', '5.25', '4.20', '5.11', '5.01', '4.91', '7.54']
    8.79s: ['5.75', '2.98', '4.22', '5.28', '2.64', '4.13', '3.98', '3.86', '3.29', '3.90', '9.24']"""

thumb_speeds = """2.39s: ['5.33', '1.55', '3.24', '6.23', '12.18', '4.64', '12.64', '5.42', '2.10', '9.18', '13.38']
    1.22s: ['6.46', '3.17', '3.57', '7.40', '11.69', '5.70', '11.60', '5.83', '1.30', '10.22', '14.40']
    0.97s: ['6.57', '1.68', '4.69', '9.69', '10.88', '4.93', '9.78', '3.67', '1.94', '10.93', '15.08']
    5.61s: ['6.43', '2.34', '4.16', '6.29', '8.14', '5.51', '7.52', '2.71', '0.94', '10.55', '16.43']
    7.12s: ['7.56', '1.49', '3.92', '3.26', '8.29', '5.33', '7.21', '1.82', '1.12', '10.33', '16.74']
    2.71s: ['7.43', '1.61', '6.90', '1.61', '8.20', '5.21', '6.27', '1.64', '1.97', '11.64', '17.05']
    3.21s: ['3.64', '2.00', '2.61', '3.56', '7.80', '4.62', '7.38', '2.95', '4.72', '10.88', '12.90']
    2.93s: ['4.37', '2.92', '1.83', '3.35', '7.98', '6.17', '7.01', '3.56', '4.34', '12.09', '12.81']
    3.06s: ['3.83', '2.40', '2.21', '2.71', '8.94', '4.84', '6.45', '2.44', '4.33', '11.49', '12.75']
    3.85s: ['4.89', '4.10', '1.76', '4.09', '8.70', '6.43', '6.05', '3.04', '3.51', '12.35', '12.67']
    2.91s: ['3.63', '2.95', '1.54', '3.16', '9.16', '5.48', '5.07', '1.66', '3.25', '11.58', '13.12']
    2.99s: ['4.62', '4.12', '1.81', '2.76', '8.87', '6.84', '4.49', '2.27', '3.90', '13.53', '12.93']
    3.76s: ['4.71', '2.98', '2.07', '2.04', '8.13', '4.02', '4.57', '1.33', '4.70', '11.06', '12.97']
    2.79s: ['4.16', '3.02', '1.90', '2.02', '8.75', '4.95', '5.31', '1.32', '3.77', '10.41', '12.52']
    2.46s: ['4.41', '2.49', '2.01', '2.58', '8.88', '4.95', '5.32', '1.88', '3.42', '10.13', '13.40']
    3.43s: ['4.45', '3.10', '2.24', '2.74', '9.68', '5.71', '6.18', '1.52', '2.75', '10.81', '13.56']
    2.17s: ['3.67', '2.83', '2.58', '2.09', '10.23', '5.24', '6.68', '1.75', '2.25', '9.58', '13.93']
    1.44s: ['4.89', '2.29', '2.76', '3.17', '9.72', '7.27', '7.21', '2.73', '1.59', '10.86', '14.12']
    1.25s: ['5.02', '2.87', '2.18', '4.45', '9.41', '6.56', '5.88', '2.60', '2.04', '10.20', '13.60']
    0.83s: ['6.07', '2.67', '1.98', '3.64', '10.44', '5.48', '6.59', '2.46', '1.65', '8.91', '13.62']
    1.42s: ['5.29', '3.20', '2.35', '4.02', '9.89', '6.44', '6.91', '2.90', '1.56', '9.97', '13.81']
    0.94s: ['5.64', '2.80', '2.62', '2.66', '10.80', '5.14', '6.19', '2.51', '1.91', '9.09', '14.80']
    1.30s: ['4.46', '3.10', '2.39', '3.07', '9.28', '5.31', '5.74', '1.92', '1.55', '9.35', '14.39']
    4.02s: ['4.72', '3.56', '3.28', '2.10', '10.01', '4.88', '6.14', '1.37', '2.23', '11.71', '15.28']
    3.67s: ['4.51', '3.20', '4.51', '2.05', '10.45', '4.27', '6.20', '1.93', '1.61', '11.04', '16.33']
    3.65s: ['4.50', '3.67', '4.28', '2.12', '10.15', '3.49', '6.33', '1.20', '2.08', '11.61', '15.66']
    6.71s: ['4.41', '2.99', '3.92', '2.63', '10.73', '3.88', '7.64', '1.34', '1.95', '11.27', '15.31']
    23.72s: ['4.64', '4.60', '3.54', '3.47', '10.34', '3.95', '6.39', '3.13', '3.89', '10.21', '15.11']
    2.68s: ['4.03', '4.05', '2.86', '4.56', '9.20', '4.45', '7.40', '1.74', '3.62', '10.33', '14.59']
    1.71s: ['3.59', '4.83', '1.89', '2.66', '9.37', '3.40', '6.62', '1.65', '3.83', '10.43', '14.70']
    2.26s: ['4.00', '4.55', '2.26', '1.77', '10.33', '4.23', '6.84', '1.52', '3.24', '10.98', '14.95']
    2.33s: ['4.25', '4.95', '2.73', '2.32', '9.54', '4.16', '7.07', '1.25', '3.10', '10.12', '14.65']
    2.20s: ['3.95', '5.03', '2.65', '2.60', '9.43', '4.14', '7.18', '1.07', '3.05', '10.35', '14.77']
    2.24s: ['3.74', '3.99', '2.51', '2.47', '9.43', '4.03', '7.55', '0.97', '3.22', '10.42', '14.90']
    2.43s: ['3.70', '3.36', '2.90', '3.20', '9.41', '3.61', '7.37', '1.50', '3.35', '10.55', '15.14']
    2.57s: ['4.11', '2.56', '2.62', '3.08', '10.32', '3.98', '7.80', '1.80', '2.81', '9.78', '15.28']
    1.59s: ['5.39', '1.63', '1.22', '4.03', '6.05', '4.92', '4.10', '1.55', '2.76', '11.66', '16.62']
    3.19s: ['3.96', '2.28', '2.05', '6.06', '7.11', '8.31', '7.06', '1.46', '2.92', '10.82', '16.30']
    2.47s: ['3.30', '1.76', '3.97', '8.20', '6.92', '8.05', '7.20', '2.37', '2.78', '10.28', '17.25']
    4.07s: ['3.10', '2.07', '4.11', '7.39', '3.32', '9.17', '8.36', '1.74', '2.34', '7.96', '17.09']
    2.08s: ['2.75', '1.71', '5.90', '9.25', '3.36', '10.14', '7.60', '2.00', '2.62', '7.16', '16.83']
    2.84s: ['4.43', '3.69', '3.00', '8.13', '2.86', '11.10', '6.83', '1.60', '2.86', '7.95', '15.83']
    2.65s: ['5.02', '3.65', '4.11', '10.14', '2.99', '11.48', '6.55', '0.99', '2.83', '8.11', '16.26']
    3.99s: ['3.93', '4.31', '3.83', '5.98', '4.60', '9.43', '6.72', '1.47', '2.35', '6.25', '16.68']
    3.17s: ['4.67', '4.42', '2.32', '2.63', '3.78', '8.89', '6.19', '1.10', '2.36', '5.64', '16.14']
    4.00s: ['4.43', '4.02', '2.10', '2.52', '4.32', '8.22', '6.51', '1.23', '2.17', '3.94', '16.62']
    3.93s: ['4.92', '3.68', '1.59', '2.37', '3.85', '8.28', '5.89', '1.07', '1.18', '2.98', '16.22']
    4.72s: ['4.88', '3.15', '2.50', '4.43', '3.14', '8.03', '5.34', '1.58', '1.30', '1.67', '16.11']
    6.49s: ['5.03', '3.85', '2.96', '5.10', '3.75', '8.10', '4.99', '1.73', '0.93', '1.37', '16.11']
    7.16s: ['5.05', '4.30', '2.79', '4.45', '3.47', '8.44', '5.19', '2.04', '0.90', '1.38', '16.35']"""

wrist_speeds="""1.43s: ['3.72', '3.75', '7.61', '16.10', '5.50', '6.33', '5.74', '7.50', '3.45', '12.09', '11.02']
    1.23s: ['4.43', '3.91', '4.57', '13.45', '5.70', '7.25', '6.24', '8.45', '5.05', '9.68', '10.93']
    1.32s: ['4.32', '4.36', '4.79', '13.08', '6.51', '7.03', '5.84', '8.48', '4.41', '12.92', '10.16']
    1.52s: ['4.27', '4.39', '7.65', '17.07', '4.22', '7.68', '3.80', '9.16', '3.26', '12.69', '8.72']
    1.42s: ['4.53', '3.94', '7.78', '17.33', '4.93', '8.03', '4.99', '9.54', '3.48', '11.84', '9.66']
    4.57s: ['5.15', '4.82', '7.23', '15.60', '2.65', '9.00', '5.25', '10.51', '2.27', '7.85', '4.34']
    10.27s: ['4.55', '5.62', '4.84', '14.60', '3.53', '8.26', '8.09', '7.96', '2.97', '8.65', '4.24']
    4.54s: ['4.33', '5.35', '4.33', '14.33', '3.31', '9.02', '6.14', '8.48', '2.61', '6.87', '5.87']
    2.85s: ['3.24', '3.31', '5.07', '14.40', '2.96', '8.13', '5.96', '7.15', '4.28', '7.60', '7.47']
    1.17s: ['2.93', '2.55', '4.02', '16.29', '4.06', '6.24', '6.31', '7.96', '3.22', '9.91', '10.30']
    1.29s: ['3.82', '2.19', '3.87', '13.18', '4.06', '6.89', '5.59', '8.33', '3.41', '7.93', '9.73']
    1.43s: ['4.07', '3.01', '3.65', '16.14', '4.21', '8.14', '3.59', '9.90', '3.45', '11.04', '9.00']
    1.18s: ['4.46', '3.25', '4.10', '15.05', '4.76', '7.21', '4.06', '8.79', '2.91', '7.13', '11.24']
    1.25s: ['4.42', '3.18', '2.68', '15.37', '4.68', '7.39', '5.64', '9.88', '3.60', '9.62', '11.64']
    3.35s: ['3.76', '5.05', '2.99', '13.36', '2.95', '6.67', '6.05', '8.32', '2.68', '9.45', '6.38']
    3.18s: ['3.92', '3.47', '3.82', '14.98', '2.27', '7.46', '6.14', '8.55', '1.88', '8.87', '6.48']
    3.29s: ['3.17', '1.82', '4.56', '15.91', '2.80', '9.08', '5.18', '8.60', '3.23', '11.10', '5.86']
    3.40s: ['3.12', '1.88', '3.88', '16.36', '2.92', '8.45', '4.57', '8.56', '3.13', '10.06', '7.36']
    2.76s: ['2.89', '2.41', '4.12', '14.76', '4.31', '8.10', '4.85', '7.89', '4.12', '12.72', '6.80']
    3.42s: ['2.85', '2.52', '4.79', '14.87', '3.70', '7.84', '4.22', '7.89', '3.99', '13.13', '5.50']
    3.18s: ['2.69', '2.15', '5.08', '16.31', '4.18', '8.14', '4.28', '8.22', '3.47', '12.61', '6.09']
    1.21s: ['2.29', '2.40', '5.88', '17.14', '5.87', '8.26', '4.88', '8.82', '4.57', '11.21', '12.56']
    1.08s: ['3.61', '2.64', '5.18', '16.31', '5.31', '7.42', '4.61', '9.28', '4.60', '11.71', '12.33']
    1.22s: ['3.63', '3.06', '5.24', '15.30', '5.84', '7.52', '3.85', '9.26', '5.22', '10.77', '12.91']
    1.23s: ['4.43', '3.59', '4.62', '14.48', '4.55', '7.82', '4.50', '9.66', '5.04', '10.54', '10.63']
    9.45s: ['3.29', '2.69', '4.05', '15.81', '3.20', '7.70', '5.00', '9.09', '2.80', '12.71', '4.20']
    9.26s: ['3.20', '2.14', '4.62', '15.49', '3.05', '7.25', '4.85', '9.15', '2.63', '10.26', '4.76']
    7.91s: ['3.61', '2.15', '5.06', '16.62', '3.36', '8.34', '4.04', '8.44', '2.44', '12.45', '4.39']
    8.40s: ['4.02', '2.26', '4.06', '15.71', '2.70', '7.63', '4.42', '8.96', '2.52', '10.83', '4.65']"""

finger_yun = """2.41s: ['7.09', '5.31', '3.87', '5.28', '3.93', '5.18', '6.24', '6.57', '6.63', '5.90', '5.69']
2.40s: ['6.00', '6.23', '3.44', '5.18', '3.07', '4.78', '4.36', '5.32', '5.69', '5.09', '10.47']
1.27s: ['5.94', '5.56', '5.13', '9.24', '7.11', '7.57', '5.75', '8.54', '10.45', '8.48', '17.26']
1.33s: ['5.73', '4.36', '5.62', '8.49', '6.77', '7.17', '6.15', '6.97', '8.56', '6.50', '18.70']
1.41s: ['5.23', '5.88', '5.49', '9.26', '5.99', '8.53', '6.96', '9.31', '7.73', '8.19', '19.90']
1.37s: ['5.59', '5.34', '4.53', '6.62', '6.65', '6.36', '6.67', '6.92', '8.30', '6.36', '22.95']
1.47s: ['5.62', '5.02', '4.52', '6.76', '5.85', '5.53', '6.69', '6.31', '7.86', '5.14', '23.14']
1.78s: ['5.92', '5.67', '4.35', '7.12', '4.43', '5.43', '6.98', '5.70', '5.96', '5.71', '23.11']
1.48s: ['5.53', '5.23', '4.86', '6.51', '5.89', '5.77', '6.21', '7.41', '7.81', '5.88', '24.38']
1.64s: ['5.96', '4.99', '3.54', '6.44', '4.90', '5.32', '5.70', '6.34', '7.84', '6.43', '24.74']
1.89s: ['4.67', '3.54', '5.86', '7.81', '5.63', '6.77', '5.15', '6.28', '8.38', '5.64', '24.41']
2.66s: ['5.56', '5.01', '3.97', '7.82', '4.25', '6.14', '4.59', '6.64', '6.60', '5.82', '22.97']
2.55s: ['7.52', '2.42', '4.31', '5.55', '4.29', '4.77', '5.84', '5.16', '4.85', '4.48', '22.18']
3.37s: ['2.06', '7.51', '3.35', '5.04', '2.95', '3.88', '4.03', '3.67', '5.02', '3.69', '24.31']
3.36s: ['2.56', '6.66', '3.07', '4.90', '3.82', '3.80', '3.58', '5.63', '6.15', '4.54', '26.42']
3.42s: ['2.98', '4.56', '2.47', '4.55', '4.34', '5.09', '4.49', '4.93', '5.76', '3.80', '27.41']
2.96s: ['1.96', '6.70', '3.67', '5.98', '3.92', '5.96', '4.90', '6.60', '6.25', '5.47', '28.26']
2.61s: ['2.71', '5.20', '4.55', '6.83', '4.05', '5.40', '5.37', '5.96', '6.04', '4.99', '29.03']
2.63s: ['1.57', '6.30', '3.71', '6.15', '4.35', '4.82', '5.18', '6.76', '7.11', '6.48', '30.27']
3.21s: ['1.17', '6.10', '3.27', '6.68', '4.04', '5.09', '3.89', '5.83', '6.31', '4.73', '31.33']
3.05s: ['2.02', '4.29', '3.62', '5.73', '3.40', '5.37', '3.90', '5.44', '6.64', '4.72', '31.21']
3.26s: ['1.91', '4.80', '2.23', '5.16', '3.89', '4.71', '5.10', '6.41', '5.94', '4.87', '31.29']
3.28s: ['1.51', '5.67', '2.36', '4.86', '3.51', '3.95', '3.73', '6.30', '6.53', '4.69', '31.58']
4.71s: ['4.44', '11.25', '4.51', '8.20', '3.70', '4.98', '3.98', '5.40', '6.94', '5.45', '24.17']
8.53s: ['9.44', '15.24', '5.84', '6.07', '5.55', '5.22', '4.59', '5.85', '3.14', '4.56', '2.90']
9.12s: ['6.73', '12.01', '6.39', '3.31', '5.39', '5.00', '4.53', '5.50', '3.00', '4.55', '8.02']
8.05s: ['5.30', '11.49', '6.22', '4.01', '4.48', '3.80', '3.16', '5.23', '3.44', '4.87', '14.23']
9.00s: ['4.17', '11.32', '5.64', '4.40', '3.35', '3.98', '3.23', '4.70', '4.12', '3.28', '16.37']
8.92s: ['1.92', '10.31', '4.57', '3.70', '4.05', '4.30', '3.72', '5.22', '3.45', '3.95', '17.09']
7.23s: ['1.39', '9.97', '3.99', '2.32', '3.45', '3.66', '4.39', '4.51', '4.74', '3.76', '16.84']"""

wrist_yun = """2.48s: ['10.34', '10.80', '8.28', '7.28', '10.60', '21.00', '4.62', '12.33', '7.41', '5.15', '7.60']
2.63s: ['9.87', '10.77', '6.85', '10.95', '8.23', '20.81', '6.21', '12.64', '8.15', '5.49', '8.15']
2.65s: ['10.18', '11.57', '5.58', '8.73', '9.27', '21.53', '4.24', '12.12', '6.77', '6.73', '8.55']
3.43s: ['9.91', '10.74', '6.85', '7.37', '8.78', '19.39', '6.51', '12.17', '5.73', '4.35', '7.83']
3.72s: ['11.16', '10.40', '5.64', '10.23', '7.49', '19.87', '9.42', '15.14', '5.70', '7.88', '7.66']
3.30s: ['9.86', '10.31', '8.15', '6.62', '8.12', '18.29', '8.24', '14.07', '6.57', '3.42', '8.61']
1.83s: ['10.21', '10.06', '6.10', '8.67', '7.29', '20.23', '6.59', '12.24', '5.18', '5.45', '6.04']
1.58s: ['9.91', '8.04', '5.94', '9.91', '6.42', '16.71', '8.57', '13.43', '5.80', '5.24', '8.55']
1.52s: ['9.97', '7.91', '4.32', '10.44', '5.44', '15.39', '10.54', '13.38', '3.73', '5.21', '7.83']
1.24s: ['10.88', '5.37', '6.29', '17.73', '6.08', '13.36', '10.27', '9.04', '5.65', '8.85', '9.32']
1.40s: ['10.76', '7.02', '7.09', '13.97', '5.35', '14.91', '8.43', '10.02', '6.00', '6.25', '7.34']
1.26s: ['10.46', '3.53', '4.13', '15.96', '6.85', '9.57', '10.06', '10.72', '4.51', '7.32', '6.64']
1.03s: ['10.07', '5.76', '5.35', '17.07', '6.86', '14.46', '10.37', '9.86', '8.76', '9.40', '12.71']
1.01s: ['11.41', '5.81', '6.98', '19.46', '10.28', '15.67', '10.71', '10.39', '7.51', '9.41', '8.83']
3.53s: ['8.51', '9.00', '4.90', '9.66', '6.29', '11.50', '6.81', '9.54', '5.20', '8.93', '8.11']
9.03s: ['8.37', '8.11', '2.85', '8.54', '6.69', '7.75', '10.02', '11.63', '5.04', '8.36', '6.62']
10.66s: ['6.07', '3.99', '7.10', '7.50', '8.31', '6.77', '17.54', '16.48', '5.94', '10.44', '5.59']"""



thumb_yun = """3.82s: ['5.68', '7.98', '13.15', '50.20', '12.96', '3.34', '19.67', '16.04', '58.81', '47.94', '14.94']
                4.02s: ['3.51', '6.32', '26.81', '67.07', '9.26', '5.26', '34.17', '51.37', '70.53', '67.04', '15.09']
                3.48s: ['2.84', '4.97', '23.76', '65.97', '10.27', '5.18', '31.50', '46.23', '68.18', '63.83', '15.35']
                3.79s: ['3.67', '6.05', '17.59', '60.14', '11.96', '5.34', '28.55', '39.52', '68.94', '63.29', '16.51']
                4.19s: ['3.29', '4.68', '12.35', '48.35', '10.80', '4.69', '25.94', '30.76', '66.46', '59.01', '15.54']
                11.77s: ['3.75', '6.77', '10.81', '47.44', '11.00', '5.66', '26.00', '33.17', '66.69', '60.79', '14.83']
                9.59s: ['2.05', '6.03', '28.82', '63.66', '6.77', '4.91', '32.42', '41.98', '66.43', '65.13', '4.39']
                7.90s: ['2.61', '2.53', '27.34', '59.18', '9.67', '5.54', '37.74', '62.54', '68.38', '70.29', '1.29']
                9.75s: ['3.02', '3.60', '20.24', '61.92', '14.27', '4.61', '35.70', '61.46', '67.85', '69.57', '1.44']
                12.43s: ['2.96', '3.21', '22.57', '62.37', '12.16', '4.12', '29.38', '47.58', '66.87', '69.32', '2.07']
                1.62s: ['6.11', '5.49', '22.83', '56.49', '8.14', '3.88', '34.41', '48.16', '64.92', '63.42', '7.60']
                1.83s: ['5.18', '3.65', '20.72', '60.72', '11.84', '3.28', '35.20', '51.27', '65.96', '67.80', '6.69']
                1.60s: ['5.40', '4.70', '20.69', '59.91', '11.71', '2.87', '34.79', '50.82', '65.11', '66.51', '5.87']
                1.60s: ['4.88', '3.80', '21.02', '62.46', '12.90', '3.90', '35.54', '54.31', '65.94', '67.81', '6.03']
                1.05s: ['6.21', '3.36', '16.17', '54.22', '14.03', '4.90', '34.27', '59.67', '62.07', '67.29', '9.22']
                1.62s: ['3.95', '2.06', '10.80', '43.30', '15.71', '4.27', '30.75', '54.72', '57.12', '61.81', '10.33']
                1.47s: ['5.30', '4.18', '10.38', '46.42', '17.09', '5.82', '30.95', '58.69', '55.53', '63.06', '11.34']
                1.73s: ['4.56', '2.93', '14.25', '51.20', '16.78', '6.82', '28.95', '55.78', '53.51', '61.24', '9.48']
                1.33s: ['4.90', '2.46', '15.95', '46.06', '14.31', '8.08', '28.72', '51.60', '57.85', '63.39', '9.95']
                1.20s: ['6.99', '3.39', '13.21', '45.37', '12.97', '6.19', '29.13', '46.81', '64.05', '65.89', '10.39']"""
wrist_yun_2 = """1.93s: ['7.89', '4.37', '18.34', '17.75', '10.58', '16.01', '17.30', '19.19', '5.94', '11.00', '13.42']
1.40s: ['8.28', '4.68', '15.59', '17.00', '13.50', '18.57', '23.31', '21.86', '8.43', '13.57', '20.46']
0.90s: ['5.64', '3.20', '12.07', '13.90', '9.60', '16.53', '18.67', '20.50', '12.73', '14.21', '17.73']
1.11s: ['5.45', '3.23', '14.20', '17.30', '9.99', '14.09', '18.76', '18.59', '13.06', '13.52', '22.62']
1.06s: ['3.49', '3.14', '15.92', '19.08', '7.48', '11.54', '14.90', '14.69', '9.95', '15.19', '21.11']
2.13s: ['4.96', '5.69', '21.71', '20.90', '12.33', '16.15', '15.79', '18.81', '10.73', '19.57', '20.05']
3.04s: ['4.14', '4.16', '17.83', '20.77', '8.82', '15.63', '14.38', '18.84', '10.76', '16.69', '16.49']
2.78s: ['5.13', '4.33', '22.27', '22.00', '7.07', '9.95', '13.80', '17.45', '15.29', '17.29', '14.53']
2.87s: ['5.53', '4.73', '20.99', '16.03', '8.98', '13.72', '8.58', '13.56', '8.72', '15.61', '14.74']
3.42s: ['5.11', '4.71', '20.83', '17.72', '9.76', '12.30', '9.83', '14.03', '8.92', '14.68', '14.49']
9.82s: ['4.27', '4.63', '23.13', '20.70', '7.96', '11.93', '15.86', '18.50', '11.94', '17.15', '14.86']
8.05s: ['4.39', '5.56', '19.22', '15.07', '11.05', '14.19', '16.07', '15.82', '9.55', '13.14', '14.82']"""

if __name__ == "__main__":
    filename = "./datasets/comparison_data_fingers.txt"
    # main(save_data=False)
    # dataset_name = "./datasets/all_fingers_9.txt"
    # evaluate_dataset(dataset_name)
    # results = test_speeds()
    parse_speeds(wrist_yun)
    pass
# from numpy import array
# tracked_errors = [(1.82218599319458, array([ 1.5816518 , 10.16246404, 11.68427857, 20.80046586,  5.08991394,
#         8.24320537,  6.16165137,  5.26882215,  6.59768815,  4.68910903,
#         2.37538247])), (1.6997966766357422, array([1.33019335, 9.33716961, 5.38914873, 5.10313138, 7.15063302,
#        8.60329407, 6.30108825, 5.09452724, 8.28079657, 5.60907082,
#        2.30161466])), (1.6863410472869873, array([1.81598657, 8.96896743, 5.63173282, 6.38264723, 5.28994281,
#        6.53970414, 6.46520707, 5.34079908, 7.59146368, 5.87072617,
#        2.85398927])), (1.639343023300171, array([1.52462133, 8.81543924, 5.56536017, 5.7632516 , 6.10096156,
#        6.61675035, 6.5309598 , 5.96141105, 8.80275387, 7.0208789 ,
#        3.23492858])), (5.770962715148926, array([2.13875771, 7.74428014, 9.55846641, 9.95569211, 3.82397255,
#        5.5782069 , 4.53600212, 3.98789997, 2.66658842, 3.38886523,
#        2.43201579])), (2.956399440765381, array([ 1.43135401, 10.01242555,  4.26224917,  4.94780224,  5.18504109,
#         5.21505016,  4.93875503,  3.64152878,  5.74481169,  4.05799264,
#         2.62463624])), (1.9608643054962158, array([2.17409305, 8.56148067, 6.74662842, 6.8937585 , 6.36462642,
#        6.94819878, 6.29387786, 5.3117009 , 6.49103326, 4.80324591,
#        2.91217276])), (2.0098884105682373, array([2.39324062, 9.22630065, 7.26172748, 7.89871054, 6.01519886,
#        6.33484217, 6.01911756, 4.46740108, 6.42325382, 4.46857081,
#        2.73992474])), (2.122572898864746, array([1.88839299, 8.91863721, 7.81115823, 8.42207375, 4.83689538,
#        7.06751856, 5.04069081, 5.05186963, 6.61053472, 4.68911336,
#        2.63626843])), (2.1284096240997314, array([2.17111594, 7.99898812, 7.47917374, 8.23773885, 5.36026608,
#        7.10609945, 5.23013846, 4.59944406, 5.96610963, 4.83835734,
#        2.52308969])), (2.142345428466797, array([2.8740159 , 8.12846869, 7.28887645, 6.26674624, 4.19728635,
#        5.30900492, 4.97679021, 5.18783547, 5.19956705, 4.82303193,
#        2.32136162])), (1.0932073593139648, array([3.05664323, 8.60110305, 6.4050084 , 6.94833673, 8.56585712,
#        9.05930415, 9.24182391, 6.96895482, 7.83365193, 6.13336274,
#        2.30142007])), (1.1489381790161133, array([3.47214085, 8.56163749, 5.60714378, 7.24242899, 8.50011974,
#        8.36051894, 9.0749341 , 7.83974952, 8.73748823, 7.05553147,
#        2.56922509])), (1.4560062885284424, array([3.72722768, 8.33385889, 5.65045405, 6.89460204, 7.44665105,
#        7.36562257, 7.98568451, 6.58087656, 7.76524006, 6.85311092,
#        2.71314773])), (1.2949824333190918, array([3.99111775, 8.45377187, 6.30910057, 8.31554021, 8.12039233,
#        8.36462833, 8.77119194, 7.58511318, 7.18744524, 5.80731023,
#        2.09626099])), (7.777473449707031, array([4.15208502, 8.37189324, 8.50371611, 8.89901909, 3.58832247,
#        4.77162796, 2.95931758, 3.83388819, 4.13286367, 5.07168626,
#        1.87843251])), (9.296309232711792, array([ 4.40801404,  7.00541979, 12.19910597,  9.59917036,  3.61844596,
#         5.2657716 ,  3.09658818,  4.14276991,  5.0561206 ,  5.87332723,
#         2.12666972])), (1.673435926437378, array([3.14276786, 8.36554099, 6.46255145, 6.2160059 , 6.08537774,
#        6.03908064, 6.41886808, 6.50198412, 4.11786581, 5.5673795 ,
#        3.22432263])), (2.705352306365967, array([ 3.6552717 , 10.01023334,  4.73341584,  4.91299833,  5.66695743,
#         4.81205509,  4.70451203,  3.70835657,  4.90350556,  4.01289136,
#         3.5784411 ])), (3.1914002895355225, array([ 3.0195026 , 10.073073  ,  4.99786894,  4.37358395,  3.33405542,
#         4.08992342,  4.35524226,  3.91434607,  5.49161841,  4.24197548,
#         3.49328617])), (2.972012996673584, array([ 3.032858  , 10.09228008,  6.40980414,  6.5357371 ,  3.40393176,
#         3.9925663 ,  4.22421123,  3.80427334,  5.81264613,  4.25902487,
#         3.79355221])), (2.6872141361236572, array([2.8762918 , 9.52794393, 4.99685569, 5.06928601, 4.30927977,
#        4.46968567, 4.47020065, 4.0014057 , 6.52322494, 4.36979701,
#        3.90632859])), (6.352766990661621, array([ 1.64535143,  9.68042718, 10.9001274 , 11.64189768,  3.28655149,
#         4.81060773,  3.87271949,  2.64404303,  5.69557269,  5.88100692,
#         2.02394282]))]

# for i in range(len(tracked_errors[0][1])):
#         # plt.plot([x[0] for x in tracked_errors], [x[1][i] for x in tracked_errors], label=list(finger_angle_indices.keys())[i])
#         errors_rounded= [ '%.2f' % elem for elem in tracked_errors[i][1]]
#         print(f"{tracked_errors[i][0]:.2f}s: {errors_rounded}")