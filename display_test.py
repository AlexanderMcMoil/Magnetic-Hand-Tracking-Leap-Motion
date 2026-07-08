import leap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading
import time

# Global variable to store the latest hand data
latest_hand = None
data_lock = threading.Lock()

def get_angle_between(v1, v2):
    """Calculates the angle in degrees between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    # Clip to handle floating point errors
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)

class PlotListener(leap.Listener):
    def on_tracking_event(self, event):
        global latest_hand
        if len(event.hands) > 0:
            with data_lock:
                # Store the first detected hand
                latest_hand = event.hands[0]
                

def setup_plot():
    # plt.ion()
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax

def draw_hand(ax, hand):
    ax.cla()
    # Set static limits
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 300)
    
    # Labeling (Note: Swapping Y/Z for visual intuition)
    ax.set_xlabel('X (Left/Right)')
    ax.set_ylabel('Z (Forward/Back)')
    ax.set_zlabel('Y (Height)')

    # 1. Wrist and Palm
    wrist = hand.arm.next_joint
    palm = hand.palm.position
    ax.scatter(palm.x, palm.z, palm.y, color='red', s=60, label='Palm')
    ax.scatter(wrist.x, wrist.z, wrist.y, color='black', s=30)

    # 2. Fingers & Bones
    for digit in hand.digits:
        # Build coordinates for the 4 bones (5 joints)
        x = [digit.bones[0].prev_joint.x] + [b.next_joint.x for b in digit.bones]
        z = [digit.bones[0].prev_joint.z] + [b.next_joint.z for b in digit.bones]
        y = [digit.bones[0].prev_joint.y] + [b.next_joint.y for b in digit.bones]
        
        ax.plot(x, z, y, linewidth=2, alpha=0.7)
        ax.scatter(x, z, y, s=15)

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
    # print(bone_vectors_all)
    # print(angles)
    return np.array(angles)

def main():
    fig, ax = setup_plot()
    
    # Initialize connection and add our custom listener
    my_listener = PlotListener()
    connection = leap.Connection()
    connection.add_listener(my_listener)
    frame_time = time.time()
    

    with connection.open():
        print("Listener active. Visualizing hand data...")
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        try:
            while plt.fignum_exists(fig.number):
                # Safely grab the hand data from the listener thread
                current_hand = None
                with data_lock:
                    current_hand = latest_hand
                
                if current_hand:
                    get_angles(current_hand)
                    # print(current_hand.confidence)
                    draw_hand(ax, current_hand)
                    # plt.pause(0.001) # Small pause to allow the GUI to update
                    print(f"Frame duration: {(time.time() - frame_time):.2f}")
                    frame_time = time.time()
                    pass
                
                
        except KeyboardInterrupt:
            print("Closing...")

if __name__ == "__main__":
    main()