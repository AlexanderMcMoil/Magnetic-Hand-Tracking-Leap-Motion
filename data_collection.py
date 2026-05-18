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

# arduino_port = "COM3"
# baud_rate = 115200

latest_hand = None
data_lock = threading.Lock()

def get_arduino_values(ser:serial.Serial) -> np.ndarray:
    ser.write(b'R\n')
    input = ser.read_until(b"\n").decode("utf-8").strip()
    input = input.split()
    input = list(map(int, input))
    # data = np.reshape(input, (5,4))
    return np.array(input)



def main(finger:str=None):
    # cap = VideoCapture(0)
    ser = serial.Serial(port=arduino_port, baudrate=baud_rate)
    file = open("./datasets/all_fingers_5.txt", "w")
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
                    if (np.round(prev_hand_pos, 3) != np.round(list(hands[0].palm.position), 3)).all():
                        magnet_values = get_arduino_values(ser)


                        angles = get_angles(hands[0])
                        data = np.concatenate([magnet_values.flatten(), angles])

                        # get fps
                        # fps = 1000000000 / (time.time_ns() - time0)
                        # time0 = time.time_ns()

                        # cv2.putText(frame, str(int(fps)), (10, 70), cv2.qqq, 3, (255, 0, 255), 3)
                        prev_hand_pos = list(hands[0].palm.position)
                        if record_data and angles[finger_angle_indices["index_mcp"]] != -1:
                            file.write(np.array2string(data.flatten(), max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
                cv2.imshow(canvas.name, canvas.output_image)

                
                key = cv2.waitKey(35)
                if key & 0xFF == ord('q'):
                    break
                elif key & 0xFF == ord(' '):
                    record_data = not record_data
                    print(f"{'Not ' if not record_data else ''}Recording")
                
                # print(f"Frame Duration: {time.time_ns() - t0}")
                # t0 = time.time_ns()

    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        ser.close()
        file.close()
    finally:
        ser.close()
        file.close()
        cv2.destroyAllWindows()
    pass

if __name__ == "__main__":
    main()
    pass