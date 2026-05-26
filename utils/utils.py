import numpy as np
import cv2, queue, threading, time, serial
import serial, struct
import math
import pickle
import inspect
from sklearn import linear_model, metrics, neural_network, preprocessing
import leap
# import requests

deg2rad = np.pi / 180
rad2deg = 180 / np.pi

pip2dip = 0.88
tmcp2ip = 0.77

ring_names = ["Wrist Ring", "Pinky Ring", "Ring Finger Ring","Middle Finger Ring", "Index Finger Ring", "Thumb Ring"]
receiver_names = ["right", "front (left)", "top", "front (right)"]
ring_num = 6
receiver_num = 4
data_num = ring_num*receiver_num


arduino_port = "COM3"
baud_rate = 115200
webcam_ip_address = "10.4.125.93"
webcam_port = "4747"
webcam_url = f"http://{webcam_ip_address}:{webcam_port}/video"

mediapipe_finger_indices = [
   [0,1,2,3,4],
   [0,5,6,7,8],
   [0,9,10,11,12],
   [0,13,14,15,16],
   [0,17,18,19,20]
]

# magnet_value_indices = {
#     "thumb": range(0, 4),
#     "index": range(4, 8),
#     "middle": range(8, 12),
#     "ring": range(12, 16),
# }

magnet_value_indices = {
   "wrist": range(0,4),
   "pinky": range(4,8),
   "ring": range(8,12),
   "middle": range(12,16),
   "index": range(16,20),
   "thumb": range(20,24)
}

finger_angle_indices = {
    "thumb_mcp": 0,
    "thumb_pip": 1,
    "thumb_dip": 2,
    "index_mcp": 3,
    "index_pip": 4,
    "index_dip": 5,
    "middle_mcp": 6,
    "middle_pip": 7,
    "middle_dip": 8,
    "ring_mcp": 9,
    "ring_pip": 10,
    "ring_dip": 11,
    "pinky_mcp": 12,
    "pinky_pip": 13,
    "pinky_dip": 14,
    "wrist": 15,
}

finger_angles = {
    "thumb": [finger_angle_indices["thumb_mcp"], finger_angle_indices["thumb_pip"], finger_angle_indices["thumb_dip"]],
    "index": [finger_angle_indices["index_mcp"], finger_angle_indices["index_pip"], finger_angle_indices["index_dip"]],
    "middle": [finger_angle_indices["middle_mcp"], finger_angle_indices["middle_pip"], finger_angle_indices["middle_dip"]],
    "ring": [finger_angle_indices["ring_mcp"], finger_angle_indices["ring_pip"], finger_angle_indices["ring_dip"]],
    "pinky": [finger_angle_indices["pinky_mcp"], finger_angle_indices["pinky_pip"], finger_angle_indices["pinky_dip"]],
    "wrist": [finger_angle_indices["wrist"]]
}

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}

class Canvas:
    def __init__(self):
        self.name = "Python Gemini Visualiser"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def toggle_hands_format(self):
        self.hands_format = "Dots" if self.hands_format == "Skeleton" else "Skeleton"
        print(f"Set hands format to {self.hands_format}")

    def get_joint_position(self, bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
        else:
            return None

    def render_hands(self, event):
        # Clear the previous image
        self.output_image[:, :] = 0

        cv2.putText(
            self.output_image,
            f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}",
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1,
        )

        if len(event.hands) == 0:
            return

        for i in range(0, len(event.hands)):
            hand = event.hands[i]
            angles = get_angles(hand)
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            angles_str = str([f"{angles[3*i], angles[3*i+1], angles[3*i+2],}\n" for i in range(5)])
            for i in range(5):
                cv2.putText(
                    self.output_image,
                    f"{angles[3*i]:2f}, {angles[3*i+1]:2f}, {angles[3*i+2]:2f}",
                    (10, 15*(i+1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.font_colour,
                    1,
                )
            
            cv2.putText(
                self.output_image,
                f"{angles[-1]}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.font_colour,
                1,
            )
                
            for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]
                    if self.hands_format == "Dots":
                        prev_joint = self.get_joint_position(bone.prev_joint)
                        next_joint = self.get_joint_position(bone.next_joint)
                        if prev_joint:
                            cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)

                        if next_joint:
                            cv2.circle(self.output_image, next_joint, 2, self.hands_colour, -1)

                    if self.hands_format == "Skeleton":
                        wrist = self.get_joint_position(hand.arm.next_joint)
                        elbow = self.get_joint_position(hand.arm.prev_joint)
                        if wrist:
                            cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)

                        if elbow:
                            cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)

                        if wrist and elbow:
                            cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                        bone_start = self.get_joint_position(bone.prev_joint)
                        bone_end = self.get_joint_position(bone.next_joint)

                        if bone_start:
                            cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)

                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)

                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                        if ((index_digit == 0) and (index_bone == 0)) or (
                            (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                        ):
                            index_digit_next = index_digit + 1
                            digit_next = hand.digits[index_digit_next]
                            bone_next = digit_next.bones[index_bone]
                            bone_next_start = self.get_joint_position(bone_next.prev_joint)
                            if bone_start and bone_next_start:
                                cv2.line(
                                    self.output_image,
                                    bone_start,
                                    bone_next_start,
                                    self.hands_colour,
                                    2,
                                )

                        if index_bone == 0 and bone_start and wrist:
                            cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)

def get_angle_between(v1, v2):
    """Calculates the angle in degrees between two vectors."""
    # print(v1)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clip to handle floating point errors
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

def get_angles(hand):
    # hand_type = "Left" if hand.type == leap.HandType.Left else "Right"
    # print(f"\n--- {hand_type} Hand Angles ---")
    # Names for display
    finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    angles = []
    bone_vectors_all = []
    for i, digit in enumerate(hand.digits):
        # We calculate the angle between:
        # 1. Metacarpal & Proximal (Knuckle)
        # 2. Proximal & Intermediate (Middle Joint)
        # 3. Intermediate & Distal (Tip Joint)
        
        bone_vectors = []
        for b in range(4):
            bone = digit.bones[b]
            # Vector direction = next_joint - prev_joint
            v = np.array([
                bone.next_joint.x - bone.prev_joint.x,
                bone.next_joint.y - bone.prev_joint.y,
                bone.next_joint.z - bone.prev_joint.z
            ])
            bone_vectors.append(v)
        bone_vectors_all += bone_vectors
        # Calculate angles between consecutive bones
        knuckle_angle = get_angle_between(bone_vectors[0], bone_vectors[1])
        mid_angle = get_angle_between(bone_vectors[1], bone_vectors[2])
        tip_angle = get_angle_between(bone_vectors[2], bone_vectors[3])
        angles += [knuckle_angle, mid_angle, tip_angle]
        # print(f"{finger_names[i]:<7}: Knuckle: {knuckle_angle:>5.1f}°, Mid: {mid_angle:>5.1f}°, Tip: {tip_angle:>5.1f}°")
    angles[0] = get_angle_between(bone_vectors_all[1], bone_vectors_all[4])
    bone = hand.arm
    arm_vector = np.array([
        bone.next_joint.x - bone.prev_joint.x,
        bone.next_joint.y - bone.prev_joint.y,
        bone.next_joint.z - bone.prev_joint.z
    ])
    cross = np.array([
        hand.index.metacarpal.next_joint.x - hand.ring.metacarpal.next_joint.x,
        hand.index.metacarpal.next_joint.y - hand.ring.metacarpal.next_joint.y,
        hand.index.metacarpal.next_joint.z - hand.ring.metacarpal.next_joint.z
    ])
    palm_direction = np.cross(bone_vectors_all[8], cross)
    # wrist_angle = get_angle_between(arm_vector, bone_vectors_all[8])
    # wrist_angle = get_angle_between(arm_vector, np.array(list(hand.palm.direction)))
    wrist_angle = get_angle_between(arm_vector, palm_direction) - 90
    if str(hand.type) == "HandType.Right":
        wrist_angle = -1* wrist_angle
    angles.append(wrist_angle)
    # print(bone_vectors_all)
    # print(angles)
    return np.array(angles)

class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas
        # self.file = file
        self.hands = None

    def on_connection_event(self, event):
        pass

    def on_tracking_mode_event(self, event):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        self.hands = event.hands
        self.canvas.render_hands(event)


class Generic_Hand_Model():

    joint_ratios = [tmcp2ip, pip2dip, pip2dip, pip2dip, pip2dip]
    fingers = ["thumb", "index", "middle", "ring", "pinky", "wrist"]
    finger_angle_indices = {
        "thumb": range(0,3),
        "index": range(3,6),
        "middle": range(6,9),
        "ring": range(9,12),
        "pinky": range(12,15),
        "wrist": [15],
    }
    def __init__(self, models:dict):
        self.models = models
        pass

    def fit(self, x, y):
        for finger in self.fingers:
            self.models[finger].fit(x[:,magnet_value_indices[finger]],y[:,self.finger_angle_indices[finger]][:,:-1])

    def fit_finger(self, x, y, finger):
        # print(y)
        # print(y[:,[15]])
        random_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(random_state)
        np.random.shuffle(y)
        cutoff = int(x.shape[0]*0.8)
        x_train = x[:cutoff,:]
        y_train = y[:cutoff,:]
        x_test = x[cutoff:,:]
        y_test = y[cutoff:,:]
        y_finger = y[:,self.finger_angle_indices[finger]][:,:]
        if finger == "wrist":
            y_finger = y_train[:,self.finger_angle_indices[finger]][:,:]
        else:
            y_finger = y_train[:,self.finger_angle_indices[finger]][:,:-1]

        self.models[finger].fit(x_train[:,magnet_value_indices[finger]],y_finger)
        predictions = self.models[finger].predict(x_test[:,magnet_value_indices[finger]])
        predictions = np.hstack([predictions, predictions[:, [1]] * self.joint_ratios[self.fingers.index(finger)]]) if finger != "wrist" else predictions
        print(f"{finger} error: {np.abs(y_test[:,self.finger_angle_indices[finger]] - predictions).mean(axis=0)*rad2deg}")
        save_model(self.models[finger], f"{finger}")

    def predict(self, data):
        result = np.array([[]])
        for i, finger in enumerate(self.fingers):
            # finger_x = preprocessing.PolynomialFeatures(2).fit_transform(data[:, magnet_value_indices[self.fingers[i]]])
            
            if self.models[finger] is None:
                prediction = np.array([[0,0,0]])
            else:
                # finger = "ring" if finger == "pinky" else finger
                prediction = self.models[finger].predict(data[:,magnet_value_indices[finger]])
                if finger == "wrist":
                    prediction = prediction.reshape([-1,1])
                else:
                    prediction = np.hstack([prediction, prediction[:, [1]] * self.joint_ratios[i]])
                # prediction[:,0] -= 15 * deg2rad
                if finger == "thumb":
                    # prediction[0][0] = 40*deg2rad
                    # print(prediction)
                    pass
                    
                # print(prediction)
            print(result)
            print(result.shape)
            result = np.hstack([result, prediction])
        return result

class SqrtEncoder:
    def __init__(self, *args, **kwargs):
        # self.encoder = StandardLabelEncoder(*args, **kwargs)
        pass
    def fit(self, X, y=None):
        return self
    def transform(self,X):
        data = X.copy()
        data = np.sqrt(data)
        return data
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()

def get_arduino_values(ser:serial.Serial) -> np.ndarray:
    ser.write(b'R\n')
    input = ser.read_until(b"\n").decode("utf-8").strip()
    input = input.split()
    input = list(map(int, input))
    
    return np.array(input)

# def webcam_setup(flashlight=False, autofocus=False):
#   cap = VideoCapture(webcam_url)
#   info = requests.get(f"http://{webcam_ip_address}:{webcam_port}/v1/camera/info").json()
#   if((info["led_on"] and not flashlight) or (not info["led_on"] and flashlight)):
#     requests.put(f"http://{webcam_ip_address}:{webcam_port}/v1/camera/torch_toggle")
#   if(autofocus and info["focusMode"] == 0):
#     requests.put(f"http://{webcam_ip_address}:{webcam_port}/v1/camera/autofocus_mode/1")
#   elif(not autofocus and info["focusMode"] == 1):
#     requests.put(f"http://{webcam_ip_address}:{webcam_port}/v1/camera/autofocus_mode/0")
#   return cap

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

# def split_data(x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     cutoff = int(x.shape[0]*0.8)
#     state = np.random.get_state()
#     np.random.shuffle(x)
#     np.random.set_state(state)
#     np.random.shuffle(y)
#     # print(x[:cutoff,:].shape)
#     x_train = x[0:cutoff]
#     x_test = x[cutoff:]
#     y_train = y[0:cutoff]
#     y_test = y[cutoff:]
#     return x_train, x_test, y_train, y_test


def save_model(model, filename:str) -> None:
  with open(f'./models/{filename}.pkl', 'wb') as f:
    pickle.dump(model, f)
  return

def load_model(filename:str):
  with open(f'./models/{filename}.pkl', 'rb') as f:
    clf = pickle.load(f)
  return clf

import matplotlib.pyplot as plt
class Fast_Magnet_Display():
  def __init__(self):
      self.fig, self.axes = plt.subplots(2,3)
      self.scatters = []
      
      for i, ax in enumerate(self.axes.flatten()[:ring_num]):
        scat = ax.scatter([], [])  # Empty scatter plot
        ax.set_xticks(range(4), receiver_names)
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(0, 1024)
        ax.set_xlabel("Receiver")
        ax.set_ylabel("Amplitude")
        ax.set_title(ring_names[i])
        self.scatters.append(scat)
      # plt.ion()
      plt.show(block=False)
      self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes.flatten()]
      # def on_resize(self, event):
      #   self.fig.canvas.draw()
      #   self.backgrounds = [self.fig.canvas.copy_from_bbox(ax.bbox) for ax in self.axes.flatten()]
      # self.fig.canvas.mpl_connect('resize_event', on_resize)
      pass
  def update(self, magnet_values: np.ndarray) -> None:
    magnet_values = magnet_values.reshape((ring_num,receiver_num))
    for i in range(ring_num):
      self.fig.canvas.restore_region(self.backgrounds[i])
      self.scatters[i].set_offsets(np.column_stack([range(4), magnet_values[i,:]]))
      self.axes.flatten()[i].draw_artist(self.scatters[i])
      self.fig.canvas.blit(self.axes.flatten()[i].bbox)
    self.fig.canvas.flush_events()


def get_long_side_points(rect):
    
    # Extract the dimensions (width, height)
    (width, height) = rect[1]
    
    # Get the four corner points of the rectangle
    box_points = cv2.boxPoints(rect)
    # Convert points to integers (required for drawing functions like cv2.line)
    box_points = np.int0(box_points)

    # The order of points returned by cv2.boxPoints is typically:
    # P0 --- P1
    # |      |
    # P3 --- P2
    # Where P1-P0 corresponds to the width, and P2-P1 corresponds to the height.
    
    # Determine the longer dimension
    if width > height:
        # The longer side corresponds to the width dimension (P0-P1)
        p1 = box_points[0]
        p2 = box_points[1]
        if p1[0] < p2[0] and p1[1] > p2[1]:
            p1 = box_points[2]
            p2 = box_points[3]
        
    else:
        # The longer side corresponds to the height dimension (P1-P2, or P0-P3)
        # We'll use P1 and P2
        p1 = box_points[1]
        p2 = box_points[2]
        pass

    return p1, p2

yellow_lower = np.array([20, 50, 50])
yellow_upper = np.array([40, 255, 255])

def get_joint_angles(frame:np.ndarray):
    # frame = cv2.flip(frame, 1)

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the yellow color
    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=6)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    v = np.zeros((3,2))
    angles = -1*np.ones((2))

    if len(contours) >= 4:
        # Sort contours by area and keep the top 3
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:4]
        
        # Sort the top 3 contours based on their y-position (top of the screen first)
        sorted_contours.sort(key=lambda c: cv2.boundingRect(c)[1])
        
        distal_ring_contour = sorted_contours[0]
        proximal_ring_contour = sorted_contours[1]
        palm_box_contour = sorted_contours[2]

        cv2.drawContours(frame, [palm_box_contour], -1, (255, 0, 0), 3) # Blue for Palm
        cv2.drawContours(frame, [proximal_ring_contour], -1, (0, 255, 0), 3) # Green for Proximal
        cv2.drawContours(frame, [distal_ring_contour], -1, (0, 0, 255), 3) # Red for Distal

        for i in range(3):
            rect = cv2.minAreaRect(sorted_contours[i])
            box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,0),2)
            
            if i == 0:
                (width, height) = rect[1]
                if width > height:
                    p1 = box[1]
                    p2 = box[2]
                else:
                    p1 = box[2]
                    p2 = box[3]
                if p2[0] < p1[0]:
                    temp = p1
                    p1 = p2
                    p2 = temp
            else:
                p1, p2 = get_long_side_points(rect)
            

            # cv2.circle(frame, p1, 5, (255, 255, 0), -1)
            # cv2.circle(frame, p2, 5, (255, 0, 255), -1)
            cv2.arrowedLine(frame, p1, p2, (255, 255, 0), 2,tipLength=0.1 if i == 0 else 0.3)
            v[i] = p1 - p2
        
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angles = np.arccos(np.einsum('nt,nt->n',
            v[[0, 1],:], 
            v[[1, 2],:])) # [15,]
        cv2.putText(frame, f"Angles: {rad2deg*angles[0]:.1f}, {rad2deg*angles[1]:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return angles


"""
class Generic_Hand_Model():

    joint_ratios = [tmcp2ip, pip2dip, pip2dip, pip2dip, pip2dip]
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    finger_angle_indices = {
        "thumb": range(0,3),
        "index": range(3,6),
        "middle": range(6,9),
        "ring": range(9,12),
        "pinky": range(12,15),
    }
    def __init__(self, models:dict):
        self.models = models
        pass
    def fit(self, x, y):
        for finger in self.fingers:
            self.models[finger].fit(x[:,magnet_value_indices[finger]],y[:,self.finger_angle_indices[finger]][:,:-1])

    def fit_finger(self, x, y, finger):
        # print(y)
        self.models[finger].fit(x[:,magnet_value_indices[finger]],y[:,self.finger_angle_indices[finger]][:,:-1])
        predictions = self.models[finger].predict(x[:,magnet_value_indices[finger]])
        print(f"{finger} error: {metrics.mean_squared_error(y[:,self.finger_angle_indices[finger]][:,:-1], predictions)}")

    def predict(self, data):
        result = np.array([[]])
        for i, finger in enumerate(self.fingers):
            # finger_x = preprocessing.PolynomialFeatures(2).fit_transform(data[:, magnet_value_indices[self.fingers[i]]])
            
            if self.models[finger] is None:
                prediction = np.array([[0,0,0]])
            else:
                finger = "ring" if finger == "pinky" else finger
                prediction = self.models[finger].predict(data[:,magnet_value_indices[finger]])
                prediction[:,0] -= 15 * deg2rad
                prediction = np.hstack([prediction, prediction[:, [1]] * self.joint_ratios[i]])
            result = np.hstack([result, prediction])
        return result
"""

if __name__ == "__main__":
  # requests.put(f"http://{webcam_ip_address}:{webcam_port}/v1/camera/torch_toggle")
  # print(requests.get(f"http://{webcam_ip_address}:{webcam_port}/v1/camera/info").json())
  pass