import numpy as np
import serial
import time
from math import pi, pow
import os
import struct
import matplotlib.pyplot as plt
import os
# from dataset_collection_constants import *
# from dataset_collection_scripts import *
from utils.utils import *

directory_path = "path/to/your/directory"


num_rings = 6
num_receivers = 4


def send_message(ser, message):
    if isinstance(message, np.ndarray):
        # Convert NumPy array to a space-separated string
        message_str = ",".join(map(str, message.tolist()))  # ✅ Correct (Space-Separated)
    else:
        message_str = str(message)

    # Append newline for Arduino parsing
    message_bytes = (message_str + "\n").encode('utf-8')

    ser.write(message_bytes)  # Send message

    print(f"Sent to Arduino: {message_str}")  # Debugging

def get_arduino_values(ser:serial.Serial) -> np.ndarray:
    ser.write(b'R\n')
    t0 = time.time_ns()

    input = ser.read_until(b"\n").decode("utf-8").strip()
    # print(input)
    input = input.split()
    print((time.time_ns() - t0)/1000000)
    input = list(map(int, input))
    # data = struct.unpack(">40H", input)
    # print(input)
    data = np.reshape(input, (num_rings,num_receivers))
    return data

def main():
    # Create folder structure
    # ring_names = ["Thumb Ring", "Index Finger Ring", "Middle Finger Ring", "Ring Finger Ring", "",""]
    # receiver_names = ["right", "front", "left", "top"]
    dataset_dir = "./datasets/test"
    file = open(dataset_dir + "/write_magnet_values_test.txt", "w")
    # fig = plt.figure(1)
    fig, axes = plt.subplots(1,6, figsize=(12, 8))
    scatters = []
    
    for i, ax in enumerate(axes.flatten()[:6]):
        
        scat = ax.scatter([], [])  # Empty scatter plot
        ax.set_xticks(range(4), receiver_names)
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(0, 1024)
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Amplitude")
        ax.set_title(ring_names[i])
        scatters.append(scat)
    # plt.ion()
    plt.show(block=False)
    # fig.canvas.draw()
    backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes.flatten()]

    # for i in range(5):
        # ax = fig.add_subplot(2, 3, i+1)
    magnet_values_all = np.zeros((0,num_rings*num_receivers))
    print(magnet_values_all)
    start_time = time.time()

    try:
        # Open serial connection
        ser = serial.Serial(arduino_port, baud_rate)
        
        # time.sleep(2)  # Allow Arduino to initialize

        while True:
            t0 = time.time_ns()
            # magnet_values = get_arduino_values(ser)
            magnet_values = ser.read_until(b"\n").decode("utf-8").strip()
            magnet_values = list(map(int, magnet_values.split()))
            magnet_values = np.reshape(magnet_values, (num_rings, num_receivers))
            # print(magnet_values)
            magnet_values_all = np.vstack([magnet_values_all, magnet_values.flatten()])
            
            for i in range(num_rings):
                fig.canvas.restore_region(backgrounds[i])
                scatters[i].set_offsets(np.column_stack([range(4), magnet_values[i,:]]))
                axes.flatten()[i].draw_artist(scatters[i])
                fig.canvas.blit(axes.flatten()[i].bbox)
            # fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # time.sleep(0.05)

            # plt.pause(0.01)
            # print((time.time_ns() - t0)/1000000)
            # print(f"FPS: {1000000000/(time.time_ns() - t0)}")
            time.sleep(0.05)


            if not plt.fignum_exists(1):

                break
            
    except serial.SerialException as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nExiting program.")
    except TimeoutError:
        print("Datasets collected!")
    finally:
        print("Done!")
        ser.close()
        plt.ioff()
        plt.close()
        file.close()
    duration = time.time() - start_time
    
        # angles_file.close()
    print("Done!")
    file.close()
    # print(magnet_values_all)
    max = magnet_values_all.max(axis=0)
    min = magnet_values_all.min(axis=0)
    print(f"Max: {min}")
    print(f"Min: {max}")
    print(f"Range: {(max - min).reshape(ring_num,receiver_num)}")
    
    fig = plt.figure(2)
    for i in range(16):
        ax = fig.add_subplot(num_rings, num_receivers, i+1)
        ax.scatter(range(magnet_values_all.shape[0]), magnet_values_all[:, i])
        ax.set_title(f"{ring_names[i//receiver_num]} at {receiver_names[i%receiver_num]}")
        pass
    fig = plt.figure(3)
    
    print(magnet_value_indices['index'])
    colors = ['red', 'blue','green','orange']
    ax = fig.add_subplot(1,1,1)
    for i in magnet_value_indices["index"]:
        ax.plot(np.linspace(0, duration, magnet_values_all.shape[0]), magnet_values_all[:, i] * 3.3/1024*1000, c=colors[i-16], label=receiver_names[i-16])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        # ax.set_title(f"Sensor {i%receiver_num+1} ({receiver_names[i%receiver_num]})")
        # ax.margins(x=0)
    plt.legend(loc="upper right")
    plt.show()

def main_test():
    plot = Fast_Magnet_Display()
    # Initialize serial connection
    ser = serial.Serial(arduino_port, baud_rate)
    time.sleep(2)  # Allow Arduino to initialize

    # Performance tracking
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Get new data
            magnet_values = get_arduino_values(ser)

            plot.update(magnet_values)
            # Exit if window is closed
            if not plt.fignum_exists(1):
                break

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        plt.close()
        total_time = time.time() - start_time
        print(f"Average FPS: {frame_count/total_time:.1f}")

def main_test_2():
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    # ring_names = ["Thumb Ring 1", "Thumb Ring 2", "Index Finger Ring", "Middle Finger Ring", "Ring Finger Ring"]
    # receiver_names = ["right", "front", "left", "bottom"]
    # dataset_dir = "./datasets/test"
    # file = open(dataset_dir + "/write_magnet_values_test.txt", "a")
    # fig = plt.figure(1)
    scatters = []
    
    
    x = np.arange(0, 4, 1)

    fig, axes = plt.subplots(2,3)
    for i in range(2):
        for j in range(3):
            # scat = ax.scatter([], [])  # Empty scatter plot
            axes[i,j].set_xticks(range(4), receiver_names)
            axes[i,j].set_xlim(-0.2, 3.2)
            axes[i,j].set_ylim(0, 1024)
            axes[i,j].set_xlabel("Receiver")
            axes[i,j].set_ylabel("Amplitude")
            axes[i,j].set_title(ring_names[i])

    fig.show()

    # We need to draw the canvas before we start animating...
    fig.canvas.draw()

    styles = ['r-', 'g-', 'y-', 'm-', 'k-', 'c-']
    def plot(ax, style):
        return ax.scatter([0, 1, 2, 3], [0,0,0,0], style, animated=True)[0]
    # lines = [plot(ax, style) for ax, style in zip(axes.flatten(), styles)]
    scatters = [ax.scatter([0,1,2,3], [0,0,0,0], animated=True) for ax in axes.flatten()]
    # Let's capture the background of the figure
    backgrounds = [fig.canvas.copy_from_bbox(ax.bbox) for ax in axes.flatten()]
    
    tstart = time.time()
    # for i in range(1, 2000):
    #     items = enumerate(zip(lines, axes.flatten(), backgrounds), start=1)
    #     for j, (line, ax, background) in items:
    #         fig.canvas.restore_region(background)
    #         line.set_ydata(np.sin(j*x + i/10.0))
    #         ax.draw_artist(line)
    #         fig.canvas.blit(ax.bbox)
    try:
        ser = serial.Serial(port=arduino_port, baudrate=baud_rate)
        while True:
            magnet_values = get_arduino_values(ser)
            items = enumerate(zip(scatters, axes.flatten(), backgrounds))
            for j, (line, ax, background) in items:
                if j >= 5:
                    continue
                fig.canvas.restore_region(background)
                line.set_offsets(np.column_stack([range(4), magnet_values[j,:]]))
                ax.draw_artist(line)
                fig.canvas.blit(ax.bbox)

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


    print('FPS:' , 2000/(time.time()-tstart))

if __name__ == '__main__':
    main()
    # main_test()
    # x = "\xff"
    # print(x)
    # file = open("./datasets/test/write_magnet_values_test.txt", "a")
    # file.write("Content!")
    # file.close()
    # with open("./datasets/test/write_magnet_value.txt", "a") as file:
        # file.write("Content!")
    # print(os.path.abspath("./datasets/test/write_magnet_values_test.txt"))
    # print(len(os.path.abspath("./datasets/test/write_magnet_values_test.txt")))
    pass