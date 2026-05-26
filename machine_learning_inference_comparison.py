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
    thumb_dataset = np.genfromtxt("./datasets/thumb.txt", delimiter=',')
    # finger_dataset = np.genfromtxt("./datasets/all_fingers_3.txt", delimiter=',')
    wrist_dataset = np.genfromtxt("./datasets/wrist.txt", delimiter=',')
    # from scipy.signal import savgol_filter
    # x_finger = finger_dataset[:,:ring_num*receiver_num]
    # y_finger = finger_dataset[:,ring_num*receiver_num:]
    model = Generic_Hand_Model(models)
    
    # y_finger[:, finger_angle_indices["index_mcp"]] = savgol_filter(y_finger[:, finger_angle_indices["index_mcp"]], 60, 2)
    # y_finger[:, finger_angle_indices["index_pip"]] = savgol_filter(y_finger[:, finger_angle_indices["index_pip"]], 60, 2)
    # model.fit_finger(thumb_dataset[:,:20], thumb_dataset[:,20:], "thumb")
    model.models["thumb"] = load_model(f"thumb2")

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

    model.models["index"] = load_model(f"index")
    model.models["middle"] = load_model(f"middle")
    model.models["ring"] = load_model(f"ring")
    model.models["pinky"] = load_model(f"pinky")
    model.models["wrist"] = load_model(f"wrist")
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


if __name__ == "__main__":
    filename = "./datasets/comparison_data_fingers.txt"
    # main(save_data=False, filename=filename)
    dataset_name = "./datasets/all_fingers_9.txt"
    evaluate_dataset(dataset_name)
    pass