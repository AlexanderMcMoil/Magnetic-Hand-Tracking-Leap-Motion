from utils.utils import *
from sklearn import tree
import serial
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data
from sklearn import preprocessing, linear_model, neural_network, metrics, pipeline
import leap

arduino_port = "COM3"
baud_rate = 115200

latest_hand = None
data_lock = threading.Lock()

def display_hand_angles(robot_id, angles):
    # angles_bullet = np.concatenate([[0], [0.5, 0], angles[1:3], [0], angles[3:6], [0], angles[6:9], [0], angles[9:12], [0], angles[12:]])
    # print(angles_bullet.__len__())
    angles_bullet = np.radians(angles)
    p.setJointMotorControlArray(robot_id, jointIndices=range(21), controlMode=p.POSITION_CONTROL, targetPositions=angles_bullet)
    for i in range(50):
        p.stepSimulation()
    return



def main(finger:str=None):
    # cap = VideoCapture(0)
    ser = serial.Serial(port=arduino_port, baudrate=baud_rate)

    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical mode
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_path = "./modelling/human_hand-master/human_hand-master/model/meshes/human_hand_scaled.urdf"
    robot_id = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=1)
    p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0, cameraPitch=-48, cameraTargetPosition=[-0.5,0,0])

    time0 = time.time_ns()
    data:np.ndarray
    record_data = False

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
    file_name = "./datasets/all_fingers_9.txt"
    file = open(file_name, "w")

    connection = leap.Connection()
    
    connection.add_listener(tracking_listener)
    t0 = time.time_ns()
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
                    try:
                        if (np.round(prev_hand_pos, 3) != np.round(list(hands[0].palm.position), 3)).all():
                            magnet_values = get_arduino_values(ser)
                            angles = get_angles(hands[0])

                            data = np.concatenate([magnet_values.flatten(), angles])

                            angles_bullet = np.concatenate([[angles[15]], [angles[0]], [0], 2*angles[1:3], [0], angles[3:6], [0], angles[6:9], [0], angles[9:12], [0], angles[12:15]])
                            display_hand_angles(robot_id=robot_id, angles=angles_bullet)
                            prev_hand_pos = list(hands[0].palm.position)
                            
                            if record_data and angles[finger_angle_indices["index_mcp"]] != -1:
                                print(angles)
                                file.write(np.array2string(data.flatten(), max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
                                pass
                    except:
                        pass
                cv2.imshow(canvas.name, canvas.output_image)
                
                key = cv2.waitKey(35)
                if key & 0xFF == ord('q'):
                    file.close()
                    p.disconnect()
                    cv2.destroyAllWindows()
                    break
                elif key & 0xFF == ord(' '):
                    record_data = not record_data
                    print(f"{'Not ' if not record_data else ''}Recording")
                    # if record_data:
                    #     file.open(file_name, "w")
                    # else:
                    #     file.close()
                
                # print(f"Frame Duration: {time.time_ns() - t0}")
                # t0 = time.time_ns()

    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        # ser.close()
        # file.close()
        pass
    finally:
        # ser.close()
        file.close()
        cv2.destroyAllWindows()
        p.disconnect()
    pass

if __name__ == "__main__":
    main()
    pass