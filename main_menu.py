import cv2
import mediapipe as mp
import os
import subprocess
import sys
import time


# ============================================================
# CONFIGURATION
# ============================================================

CAM_SOURCE = "http://localhost:8080/video"
MODEL_FILE = "hand_landmarker.task"

WINDOW_NAME = "Hand Game Center"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

GAME1_FILE = "hand_ar.py"
GAME2_FILE = "game2.py"

SELECTION_HOLD_TIME = 0.35
CONFIRM_HOLD_TIME = 0.55


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# ============================================================
# COLORS
# ============================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (80, 220, 120)
BLUE = (100, 170, 255)
YELLOW = (0, 220, 255)
RED = (80, 80, 255)
GRAY = (130, 130, 130)
PURPLE = (210, 100, 220)


# ============================================================
# HAND CONNECTIONS
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ============================================================
# DRAWING
# ============================================================

def draw_text(
    frame,
    text,
    position,
    font_scale=0.7,
    thickness=2,
    color=WHITE,
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def centered_text(
    frame,
    text,
    y,
    font_scale=0.7,
    thickness=2,
    color=WHITE,
):
    (width, height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )

    x = (frame.shape[1] - width) // 2

    draw_text(
        frame,
        text,
        (x, y),
        font_scale,
        thickness,
        color,
    )


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]

    points = []

    for lm in landmarks:
        points.append(
            (
                int(lm.x * w),
                int(lm.y * h),
            )
        )

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            BLUE,
            2,
            cv2.LINE_AA,
        )

    for i, point in enumerate(points):
        radius = 6 if i in [4, 8, 12, 16, 20] else 4

        cv2.circle(
            frame,
            point,
            radius,
            GREEN,
            -1,
            cv2.LINE_AA,
        )


# ============================================================
# GEOMETRY
# ============================================================

def distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y

    return (dx * dx + dy * dy) ** 0.5


# ============================================================
# FINGER STATE
# ============================================================

def finger_extended(landmarks, tip, pip, mcp):
    """
    More robust finger extension check.

    Primarily checks the tip against the PIP/MCP region.
    This is more tolerant than checking one Y comparison only.
    """

    tip_lm = landmarks[tip]
    pip_lm = landmarks[pip]
    mcp_lm = landmarks[mcp]

    vertical_extension = (
        tip_lm.y
        < pip_lm.y - 0.015
    )

    palm_distance = distance(
        tip_lm,
        mcp_lm,
    )

    pip_distance = distance(
        pip_lm,
        mcp_lm,
    )

    distance_extension = (
        palm_distance
        > pip_distance * 1.10
    )

    return (
        vertical_extension
        or distance_extension
    )


def finger_folded(landmarks, tip, pip):
    return (
        landmarks[tip].y
        > landmarks[pip].y - 0.01
    )


# ============================================================
# THUMB DETECTION
# ============================================================

def thumb_extended(landmarks):
    """
    The thumb is different from the other fingers.

    We use distance from the thumb tip to the palm
    plus its horizontal displacement.
    """

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_mcp = landmarks[5]
    wrist = landmarks[0]

    thumb_to_index = distance(
        thumb_tip,
        index_mcp,
    )

    ip_to_index = distance(
        thumb_ip,
        index_mcp,
    )

    thumb_far = (
        thumb_to_index
        > ip_to_index * 1.08
    )

    horizontal_extension = (
        abs(
            thumb_tip.x
            - wrist.x
        )
        > 0.10
    )

    return (
        thumb_far
        or horizontal_extension
    )


# ============================================================
# MENU GESTURES
# ============================================================

def is_one_finger(landmarks):
    """
    ☝️ Game 1
    """

    index_up = finger_extended(
        landmarks,
        8,
        6,
        5,
    )

    middle_down = finger_folded(
        landmarks,
        12,
        10,
    )

    ring_down = finger_folded(
        landmarks,
        16,
        14,
    )

    pinky_down = finger_folded(
        landmarks,
        20,
        18,
    )

    return (
        index_up
        and middle_down
        and ring_down
        and pinky_down
    )


def is_two_fingers(landmarks):
    """
    ✌️ Game 2
    """

    index_up = finger_extended(
        landmarks,
        8,
        6,
        5,
    )

    middle_up = finger_extended(
        landmarks,
        12,
        10,
        9,
    )

    ring_down = finger_folded(
        landmarks,
        16,
        14,
    )

    pinky_down = finger_folded(
        landmarks,
        20,
        18,
    )

    return (
        index_up
        and middle_up
        and ring_down
        and pinky_down
    )


def is_ily_gesture(landmarks):
    """
    🤟 I Love You gesture.

    Extended:
        Thumb
        Index
        Pinky

    Folded:
        Middle
        Ring

    The detector intentionally uses several
    measurements instead of a single strict
    Y-coordinate test.
    """

    thumb = thumb_extended(
        landmarks
    )

    index = finger_extended(
        landmarks,
        8,
        6,
        5,
    )

    middle = not finger_extended(
        landmarks,
        12,
        10,
        9,
    )

    ring = not finger_extended(
        landmarks,
        16,
        14,
        13,
    )

    pinky = finger_extended(
        landmarks,
        20,
        18,
        17,
    )

    return (
        thumb
        and index
        and middle
        and ring
        and pinky
    )


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    cap = cv2.VideoCapture(
        CAM_SOURCE
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    return cap


# ============================================================
# LAUNCH
# ============================================================

def launch_game(filename):

    if not os.path.exists(filename):

        print()
        print("ERROR:")
        print(f"Cannot find {filename}")
        print()
        return

    print()
    print(f"Launching {filename}...")
    print()

    cv2.destroyAllWindows()

    try:
        subprocess.run(
            [
                sys.executable,
                filename,
            ],
            check=False,
        )
    except Exception as error:
        print(
            f"Error launching {filename}:"
        )
        print(error)

    print()
    print(
        "Returned to main menu."
    )
    print()


# ============================================================
# DRAW MENU
# ============================================================

def draw_menu(
    frame,
    selected_game,
    detected_gesture,
    confirm_progress,
):

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            FRAME_WIDTH,
            FRAME_HEIGHT,
        ),
        BLACK,
        -1,
    )

    frame = cv2.addWeighted(
        frame,
        0.20,
        overlay,
        0.80,
        0,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    centered_text(
        frame,
        "HAND GAME CENTER",
        55,
        1.05,
        3,
        WHITE,
    )

    centered_text(
        frame,
        "Select a game with your fingers",
        88,
        0.50,
        1,
        GRAY,
    )

    # --------------------------------------------------------
    # Game boxes
    # --------------------------------------------------------

    left = (45, 125, 300, 320)
    right = (340, 125, 595, 320)

    if selected_game == 1:
        left_color = GREEN
        right_color = GRAY
    else:
        left_color = GRAY
        right_color = GREEN

    cv2.rectangle(
        frame,
        (left[0], left[1]),
        (left[2], left[3]),
        left_color,
        5,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        frame,
        (right[0], right[1]),
        (right[2], right[3]),
        right_color,
        5,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Game 1
    # --------------------------------------------------------

    draw_text(
        frame,
        "GAME 1",
        (112, 175),
        0.85,
        3,
        WHITE,
    )

    draw_text(
        frame,
        "HAND ARCADE",
        (76, 220),
        0.55,
        2,
        GREEN,
    )

    draw_text(
        frame,
        "☝  1 FINGER",
        (91, 275),
        0.55,
        2,
        WHITE,
    )

    # --------------------------------------------------------
    # Game 2
    # --------------------------------------------------------

    draw_text(
        frame,
        "GAME 2",
        (407, 175),
        0.85,
        3,
        WHITE,
    )

    draw_text(
        frame,
        "HAND RHYTHM",
        (372, 220),
        0.55,
        2,
        PURPLE,
    )

    draw_text(
        frame,
        "✌  2 FINGERS",
        (382, 275),
        0.55,
        2,
        WHITE,
    )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    centered_text(
        frame,
        "☝  = Game 1       ✌  = Game 2",
        365,
        0.52,
        2,
        WHITE,
    )

    centered_text(
        frame,
        "🤟  = Launch Selected Game",
        400,
        0.52,
        2,
        YELLOW,
    )

    centered_text(
        frame,
        "ESC = Exit",
        440,
        0.45,
        1,
        GRAY,
    )

    # --------------------------------------------------------
    # Selection
    # --------------------------------------------------------

    selected_text = (
        "SELECTED: GAME 1"
        if selected_game == 1
        else "SELECTED: GAME 2"
    )

    centered_text(
        frame,
        selected_text,
        115,
        0.42,
        1,
        GREEN,
    )

    # --------------------------------------------------------
    # Gesture status
    # --------------------------------------------------------

    if detected_gesture == "GAME1":

        draw_text(
            frame,
            "☝ GAME 1",
            (15, 25),
            0.45,
            2,
            GREEN,
        )

    elif detected_gesture == "GAME2":

        draw_text(
            frame,
            "✌ GAME 2",
            (15, 25),
            0.45,
            2,
            PURPLE,
        )

    elif detected_gesture == "CONFIRM":

        draw_text(
            frame,
            "🤟 LAUNCHING...",
            (15, 25),
            0.45,
            2,
            YELLOW,
        )

        # Progress bar

        x1 = 180
        x2 = 460
        y1 = 412
        y2 = 428

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            GRAY,
            2,
        )

        fill = int(
            (x2 - x1)
            * confirm_progress
        )

        if fill > 0:

            cv2.rectangle(
                frame,
                (x1, y1),
                (
                    x1 + fill,
                    y2,
                ),
                GREEN,
                -1,
            )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(
        MODEL_FILE
    ):

        print(
            "ERROR: hand_landmarker.task "
            "was not found."
        )

        return

    cap = open_camera()

    if not cap.isOpened():

        print(
            "ERROR: Could not open camera."
        )

        return

    options = HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_FILE
        ),

        running_mode=(
            VisionRunningMode.VIDEO
        ),

        num_hands=1,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    landmarker = (
        HandLandmarker.create_from_options(
            options
        )
    )

    selected_game = 1

    current_selection_gesture = None
    selection_start = None

    confirm_start = None

    timestamp_ms = 0

    try:

        while True:

            # ------------------------------------------------
            # Drop stale frames
            # ------------------------------------------------

            grabbed = False

            for _ in range(2):

                if cap.grab():
                    grabbed = True

            if not grabbed:
                continue

            success, frame = (
                cap.retrieve()
            )

            if not success:
                continue

            frame = cv2.resize(
                frame,
                (
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                ),
                interpolation=cv2.INTER_LINEAR,
            )

            # ------------------------------------------------
            # MediaPipe
            # ------------------------------------------------

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=(
                    mp.ImageFormat.SRGB
                ),
                data=rgb,
            )

            timestamp_ms += 33

            result = (
                landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms,
                )
            )

            detected_gesture = None

            if result.hand_landmarks:

                landmarks = (
                    result.hand_landmarks[0]
                )

                draw_hand(
                    frame,
                    landmarks,
                )

                # IMPORTANT:
                # Check 🤟 FIRST.
                #
                # This prevents the ILY gesture from
                # accidentally being interpreted as a
                # selection gesture.

                if is_ily_gesture(
                    landmarks
                ):

                    detected_gesture = "CONFIRM"

                elif is_two_fingers(
                    landmarks
                ):

                    detected_gesture = "GAME2"

                elif is_one_finger(
                    landmarks
                ):

                    detected_gesture = "GAME1"

            # ------------------------------------------------
            # Selection gesture
            # ------------------------------------------------

            if detected_gesture in [
                "GAME1",
                "GAME2",
            ]:

                if (
                    detected_gesture
                    != current_selection_gesture
                ):

                    current_selection_gesture = (
                        detected_gesture
                    )

                    selection_start = (
                        time.monotonic()
                    )

                else:

                    if selection_start:

                        elapsed = (
                            time.monotonic()
                            - selection_start
                        )

                        if (
                            elapsed
                            >= SELECTION_HOLD_TIME
                        ):

                            if (
                                detected_gesture
                                == "GAME1"
                            ):

                                selected_game = 1

                            else:

                                selected_game = 2

            else:

                current_selection_gesture = None
                selection_start = None

            # ------------------------------------------------
            # Confirm
            # ------------------------------------------------

            confirm_progress = 0.0

            if detected_gesture == "CONFIRM":

                if confirm_start is None:

                    confirm_start = (
                        time.monotonic()
                    )

                elapsed = (
                    time.monotonic()
                    - confirm_start
                )

                confirm_progress = min(
                    elapsed
                    / CONFIRM_HOLD_TIME,
                    1.0,
                )

                if (
                    elapsed
                    >= CONFIRM_HOLD_TIME
                ):

                    if selected_game == 1:

                        launch_game(
                            GAME1_FILE
                        )

                    else:

                        launch_game(
                            GAME2_FILE
                        )

                    confirm_start = None
                    current_selection_gesture = None
                    selection_start = None

                    # Reopen camera after game

                    cap.release()

                    cap = open_camera()

                    # Give camera a moment

                    time.sleep(0.25)

            else:

                confirm_start = None

            # ------------------------------------------------
            # Draw menu
            # ------------------------------------------------

            frame = draw_menu(
                frame,
                selected_game,
                detected_gesture,
                confirm_progress,
            )

            cv2.imshow(
                WINDOW_NAME,
                frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == 27:
                break

    finally:

        cap.release()

        landmarker.close()

        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()