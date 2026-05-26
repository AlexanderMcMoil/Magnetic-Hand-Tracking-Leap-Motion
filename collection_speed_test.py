import time
import numpy as np
import serial
import numpy as np
import matplotlib.pyplot as plt

arduino_port = "COM3"
baud_rate = 115200


def get_arduino_values(ser:serial.Serial) -> np.ndarray:

    ser.write(b'R\n')
    input = ser.read_until(b"\n").decode("utf-8").strip()
    ser.flush()
    input = input.split()
    input = list(map(int, input))
    # data = np.reshape(input, (5,4))
    return np.array(input)

try:
    # Open serial connection
    ser = serial.Serial(arduino_port, baud_rate)
    times = []
    file = open("./datasets/test.txt", "w")
    t0 = time.time_ns()
    maximum = 0
    minimum = 1e20
    while True:
        # t0 = time.time_ns()
        magnet_values = get_arduino_values(ser)
        # print(magnet_values)
        # file.write(np.array2string(magnet_values.flatten(), max_line_width=100000, separator=",").replace(" ", "")[1:-1] + "\n")
        dif = time.time_ns() - t0
        # print(dif)
        times.append(dif/1000000)

        maximum = max(maximum, dif)
        minimum = min(minimum, dif)
        # if dif == 0:
        #     print(magnet_values)
        t0 = time.time_ns()

            
except serial.SerialException as e:
    print(f"Error: {e}")
except KeyboardInterrupt:
    print("\nExiting program.")
except TimeoutError:
    print("Datasets collected!")
finally:
    print("Done!")
    ser.close()
    file.close()
    print(f"Max: {maximum}\nMin: {minimum}")
    print(f"Average: {np.mean(times)}")
    plt.plot(times)
    # plt.yscale("log")
    plt.show()