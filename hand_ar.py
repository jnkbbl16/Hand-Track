import cv2
import mediapipe as mp
import random
import math
import urllib.request
import os
import threading
import time


# ============================================================
# SETTINGS
# ============================================================

CAM_SOURCE = "http://localhost:8080/video"

WINDOW_NAME = "Hand Arcade"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PLAYER_RADIUS = 25

# Starting obstacle settings
BASE_OBSTACLE_WIDTH = 70
BASE_OBSTACLE_GAP = 180
BASE_OBSTACLE_SPEED = 5

BASE_SPAWN_DISTANCE = 350

PINCH_DISTANCE = 0.12

# Difficulty settings
MAX_SPEED = 12
MIN_GAP = 90
MIN_SPAWN_DISTANCE = 220

# Camera
CAMERA_BUFFER_SIZE = 1


# ============================================================
# MEDIAPIPE MODEL
# ============================================================

MODEL_FILE = "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)


def download_model():

    if os.path.exists(MODEL_FILE):
        return

    print("Downloading MediaPipe hand model...")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_FILE
        )

        print("Hand model downloaded.")

    except Exception as e:

        raise RuntimeError(
            f"Could not download MediaPipe model: {e}"
        )


download_model()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_FILE
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.5,

    min_hand_presence_confidence=0.5,

    min_tracking_confidence=0.5
)


landmarker = HandLandmarker.create_from_options(
    options
)


# ============================================================
# LOW-LATENCY CAMERA READER
# ============================================================

class LatestFrameCamera:

    def __init__(self, source):

        self.source = source

        self.cap = cv2.VideoCapture(
            source
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            CAMERA_BUFFER_SIZE
        )

        self.frame = None

        self.running = True

        self.lock = threading.Lock()

        self.thread = threading.Thread(
            target=self._reader,
            daemon=True
        )

        self.thread.start()


    def _reader(self):

        while self.running:

            ok, frame = self.cap.read()

            if not ok:

                time.sleep(0.005)

                continue

            with self.lock:

                # Always replace the old frame.
                # This prevents old frames from
                # building up in memory.
                self.frame = frame


    def read(self):

        with self.lock:

            if self.frame is None:

                return False, None

            return True, self.frame.copy()


    def release(self):

        self.running = False

        if self.thread.is_alive():

            self.thread.join(
                timeout=1
            )

        self.cap.release()


camera = LatestFrameCamera(
    CAM_SOURCE
)


# Give camera a moment to receive first frame

time.sleep(0.5)


ok, test_frame = camera.read()

if not ok:

    camera.release()

    raise RuntimeError(
        "Could not open camera.\n\n"
        "Make sure your phone camera server is running.\n\n"
        "Then run:\n"
        "adb devices\n"
        "adb forward tcp:8080 tcp:8080"
    )


# ============================================================
# HAND FUNCTIONS
# ============================================================

def is_thumbs_up(landmarks):

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    index_tip = landmarks[8]
    index_pip = landmarks[6]

    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]


    thumb_up = (
        thumb_tip.y < thumb_ip.y
    )


    index_folded = (
        index_tip.y > index_pip.y
    )


    middle_folded = (
        middle_tip.y > middle_pip.y
    )


    ring_folded = (
        ring_tip.y > ring_pip.y
    )


    pinky_folded = (
        pinky_tip.y > pinky_pip.y
    )


    return (

        thumb_up

        and index_folded

        and middle_folded

        and ring_folded

        and pinky_folded

    )


def is_pinch(landmarks):

    thumb = landmarks[4]

    index = landmarks[8]


    dx = thumb.x - index.x

    dy = thumb.y - index.y


    distance = math.sqrt(
        dx * dx +
        dy * dy
    )


    return distance < PINCH_DISTANCE


def get_pinch_position(
    landmarks,
    width,
    height
):

    thumb = landmarks[4]

    index = landmarks[8]


    x = int(
        (
            thumb.x +
            index.x
        )
        / 2
        * width
    )


    y = int(
        (
            thumb.y +
            index.y
        )
        / 2
        * height
    )


    return x, y


# ============================================================
# DIFFICULTY
# ============================================================

def get_difficulty(score):

    # Every 3 points increases difficulty.

    level = score // 3


    # Obstacle speed

    speed = (
        BASE_OBSTACLE_SPEED
        +
        level * 0.8
    )


    speed = min(
        speed,
        MAX_SPEED
    )


    # Gap becomes smaller

    gap = (
        BASE_OBSTACLE_GAP
        -
        level * 10
    )


    gap = max(
        gap,
        MIN_GAP
    )


    # Obstacles spawn closer together

    spawn_distance = (
        BASE_SPAWN_DISTANCE
        -
        level * 20
    )


    spawn_distance = max(
        spawn_distance,
        MIN_SPAWN_DISTANCE
    )


    return (
        speed,
        gap,
        spawn_distance
    )


# ============================================================
# OBSTACLES
# ============================================================

def create_obstacle(
    x,
    gap_size
):

    safe_top = 60

    safe_bottom = 60


    max_gap_y = (
        FRAME_HEIGHT
        -
        gap_size
        -
        safe_bottom
    )


    if max_gap_y < safe_top:

        max_gap_y = safe_top


    gap_y = random.randint(
        safe_top,
        max_gap_y
    )


    return {

        "x": x,

        "gap_y": gap_y,

        "gap_size": gap_size,

        "passed": False

    }


# ============================================================
# RESET GAME
# ============================================================

def reset_game():

    player_x = 120

    player_y = FRAME_HEIGHT // 2


    speed, gap_size, spawn_distance = (
        get_difficulty(0)
    )


    obstacles = [

        create_obstacle(
            FRAME_WIDTH + 180,
            gap_size
        )

    ]


    score = 0

    game_over = False


    return (

        player_x,

        player_y,

        obstacles,

        score,

        game_over

    )


# ============================================================
# COLLISION
# ============================================================

def circle_rectangle_collision(

    circle_x,
    circle_y,
    radius,

    rect_x,
    rect_y,
    rect_width,
    rect_height

):

    closest_x = max(

        rect_x,

        min(
            circle_x,
            rect_x + rect_width
        )

    )


    closest_y = max(

        rect_y,

        min(
            circle_y,
            rect_y + rect_height
        )

    )


    dx = (
        circle_x -
        closest_x
    )


    dy = (
        circle_y -
        closest_y
    )


    return (

        dx * dx +
        dy * dy

    ) < (

        radius *
        radius

    )


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    frame,
    landmarks
):

    height, width = (
        frame.shape[:2]
    )


    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17)

    ]


    points = []


    for landmark in landmarks:

        x = int(
            landmark.x *
            width
        )

        y = int(
            landmark.y *
            height
        )


        points.append(
            (x, y)
        )


    for start, end in connections:

        cv2.line(

            frame,

            points[start],

            points[end],

            (0, 255, 0),

            2

        )


    for point in points:

        cv2.circle(

            frame,

            point,

            4,

            (0, 0, 255),

            -1

        )


# ============================================================
# INITIAL STATE
# ============================================================

(
    player_x,
    player_y,
    obstacles,
    score,
    game_over
) = reset_game()


game_started = False

thumbs_up_previous = False

timestamp_ms = 0


# ============================================================
# START MESSAGE
# ============================================================

print()

print(
    "=========================================="
)

print(
    "              HAND ARCADE"
)

print(
    "=========================================="
)

print()

print(
    "Show THUMBS UP to start."
)

print(
    "PINCH to hold the character."
)

print(
    "Move your hand to guide the character."
)

print()

print(
    "Avoid the obstacles."
)

print(
    "Each obstacle passed = 1 point."
)

print()

print(
    "Difficulty increases every 3 points."
)

print()

print(
    "Press Q or ESC to quit."
)

print()

print(
    "=========================================="
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # GET NEWEST CAMERA FRAME
    # --------------------------------------------------------

    ok, frame = camera.read()


    if not ok:

        continue


    # Mirror camera

    frame = cv2.flip(
        frame,
        1
    )


    height, width = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # MEDIAPIPE IMAGE
    # --------------------------------------------------------

    rgb = cv2.cvtColor(

        frame,

        cv2.COLOR_BGR2RGB

    )


    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb

    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    timestamp_ms = int(
        time.monotonic()
        * 1000
    )


    # --------------------------------------------------------
    # HAND DETECTION
    # --------------------------------------------------------

    result = (
        landmarker.detect_for_video(

            mp_image,

            timestamp_ms

        )
    )


    thumbs_up = False

    pinch = False

    pinch_x = player_x

    pinch_y = player_y


    # --------------------------------------------------------
    # HAND PROCESSING
    # --------------------------------------------------------

    if result.hand_landmarks:

        landmarks = (
            result.hand_landmarks[0]
        )


        draw_hand(

            frame,

            landmarks

        )


        thumbs_up = (
            is_thumbs_up(
                landmarks
            )
        )


        pinch = (
            is_pinch(
                landmarks
            )
        )


        if pinch:

            (
                pinch_x,
                pinch_y
            ) = get_pinch_position(

                landmarks,

                width,

                height

            )


    # ========================================================
    # THUMBS-UP START / RESTART
    # ========================================================

    thumbs_up_pressed = (

        thumbs_up

        and

        not thumbs_up_previous

    )


    if thumbs_up_pressed:

        if not game_started:

            game_started = True

            (
                player_x,
                player_y,
                obstacles,
                score,
                game_over
            ) = reset_game()


        elif game_over:

            (
                player_x,
                player_y,
                obstacles,
                score,
                game_over
            ) = reset_game()

            game_started = True


    thumbs_up_previous = thumbs_up


    # ========================================================
    # GAME LOGIC
    # ========================================================

    if (

        game_started

        and

        not game_over

    ):

        # ----------------------------------------------------
        # CURRENT DIFFICULTY
        # ----------------------------------------------------

        (

            obstacle_speed,

            obstacle_gap,

            spawn_distance

        ) = get_difficulty(
            score
        )


        # ----------------------------------------------------
        # PLAYER MOVEMENT
        # ----------------------------------------------------

        if pinch:

            smoothing = 0.55


            player_x = int(

                player_x *
                (1 - smoothing)

                +

                pinch_x *
                smoothing

            )


            player_y = int(

                player_y *
                (1 - smoothing)

                +

                pinch_y *
                smoothing

            )


        # Keep player inside screen

        player_x = max(

            PLAYER_RADIUS,

            min(

                width -
                PLAYER_RADIUS,

                player_x

            )

        )


        player_y = max(

            PLAYER_RADIUS,

            min(

                height -
                PLAYER_RADIUS,

                player_y

            )

        )


        # ----------------------------------------------------
        # MOVE OBSTACLES
        # ----------------------------------------------------

        for obstacle in obstacles:

            obstacle["x"] -= (
                obstacle_speed
            )


        # ----------------------------------------------------
        # CREATE NEW OBSTACLE
        # ----------------------------------------------------

        if obstacles:

            last = obstacles[-1]


            if (

                last["x"]

                <

                width -
                spawn_distance

            ):

                obstacles.append(

                    create_obstacle(

                        width + 80,

                        obstacle_gap

                    )

                )


        # ----------------------------------------------------
        # SCORE
        # ----------------------------------------------------

        for obstacle in obstacles:

            right_edge = (

                obstacle["x"]

                +

                BASE_OBSTACLE_WIDTH

            )


            if (

                not obstacle["passed"]

                and

                right_edge <
                player_x

            ):

                obstacle["passed"] = True

                score += 1


        # ----------------------------------------------------
        # REMOVE OLD OBSTACLES
        # ----------------------------------------------------

        obstacles = [

            obstacle

            for obstacle in obstacles

            if (

                obstacle["x"]

                +

                BASE_OBSTACLE_WIDTH

                > -100

            )

        ]


        # ----------------------------------------------------
        # COLLISION
        # ----------------------------------------------------

        for obstacle in obstacles:

            obstacle_x = (
                obstacle["x"]
            )

            gap_y = (
                obstacle["gap_y"]
            )

            gap_size = (
                obstacle["gap_size"]
            )


            # Top obstacle

            top_collision = (
                circle_rectangle_collision(

                    player_x,

                    player_y,

                    PLAYER_RADIUS,

                    obstacle_x,

                    0,

                    BASE_OBSTACLE_WIDTH,

                    gap_y

                )
            )


            # Bottom obstacle

            bottom_y = (
                gap_y +
                gap_size
            )


            bottom_collision = (
                circle_rectangle_collision(

                    player_x,

                    player_y,

                    PLAYER_RADIUS,

                    obstacle_x,

                    bottom_y,

                    BASE_OBSTACLE_WIDTH,

                    height -
                    bottom_y

                )
            )


            if (

                top_collision

                or

                bottom_collision

            ):

                game_over = True

                break


    # ========================================================
    # DRAW OBSTACLES
    # ========================================================

    if game_started:

        for obstacle in obstacles:

            x = int(
                obstacle["x"]
            )

            gap = int(
                obstacle["gap_y"]
            )

            gap_size = int(
                obstacle["gap_size"]
            )


            # Top obstacle

            cv2.rectangle(

                frame,

                (x, 0),

                (

                    x +
                    BASE_OBSTACLE_WIDTH,

                    gap

                ),

                (40, 180, 40),

                -1

            )


            # Bottom obstacle

            cv2.rectangle(

                frame,

                (

                    x,

                    gap +
                    gap_size

                ),

                (

                    x +
                    BASE_OBSTACLE_WIDTH,

                    height

                ),

                (40, 180, 40),

                -1

            )


            # Borders

            cv2.rectangle(

                frame,

                (x, 0),

                (

                    x +
                    BASE_OBSTACLE_WIDTH,

                    gap

                ),

                (0, 255, 0),

                3

            )


            cv2.rectangle(

                frame,

                (

                    x,

                    gap +
                    gap_size

                ),

                (

                    x +
                    BASE_OBSTACLE_WIDTH,

                    height

                ),

                (0, 255, 0),

                3

            )


    # ========================================================
    # DRAW PLAYER
    # ========================================================

    if game_started:

        cv2.circle(

            frame,

            (

                int(player_x),

                int(player_y)

            ),

            PLAYER_RADIUS + 5,

            (255, 255, 255),

            -1

        )


        cv2.circle(

            frame,

            (

                int(player_x),

                int(player_y)

            ),

            PLAYER_RADIUS,

            (0, 140, 255),

            -1

        )


        cv2.circle(

            frame,

            (

                int(player_x),

                int(player_y)

            ),

            7,

            (255, 255, 255),

            -1

        )


    # ========================================================
    # SCORE
    # ========================================================

    if game_started:

        cv2.putText(

            frame,

            f"SCORE: {score}",

            (20, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.2,

            (255, 255, 255),

            3,

            cv2.LINE_AA

        )


        # Difficulty level

        difficulty_level = (
            score // 3
        ) + 1


        cv2.putText(

            frame,

            f"LEVEL: {difficulty_level}",

            (20, 82),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 255),

            2,

            cv2.LINE_AA

        )


    # ========================================================
    # START SCREEN
    # ========================================================

    if not game_started:

        overlay = frame.copy()


        cv2.rectangle(

            overlay,

            (0, 0),

            (width, height),

            (0, 0, 0),

            -1

        )


        frame = cv2.addWeighted(

            overlay,

            0.60,

            frame,

            0.40,

            0

        )


        cv2.putText(

            frame,

            "HAND ARCADE",

            (

                width // 2 -
                190,

                150

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.5,

            (255, 255, 255),

            4,

            cv2.LINE_AA

        )


        cv2.putText(

            frame,

            "SHOW THUMBS UP TO START",

            (

                width // 2 -
                210,

                225

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.85,

            (0, 255, 255),

            2,

            cv2.LINE_AA

        )


        cv2.putText(

            frame,

            "PINCH TO HOLD AND MOVE",

            (

                width // 2 -
                190,

                275

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (255, 255, 255),

            2,

            cv2.LINE_AA

        )


    # ========================================================
    # GAME OVER
    # ========================================================

    if game_over:

        overlay = frame.copy()


        cv2.rectangle(

            overlay,

            (0, 0),

            (width, height),

            (0, 0, 0),

            -1

        )


        frame = cv2.addWeighted(

            overlay,

            0.65,

            frame,

            0.35,

            0

        )


        cv2.putText(

            frame,

            "YOU LOSE",

            (

                width // 2 -
                150,

                165

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.5,

            (0, 0, 255),

            4,

            cv2.LINE_AA

        )


        cv2.putText(

            frame,

            f"SCORE: {score}",

            (

                width // 2 -
                100,

                225

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (255, 255, 255),

            3,

            cv2.LINE_AA

        )


        cv2.putText(

            frame,

            "SHOW THUMBS UP TO RESTART",

            (

                width // 2 -
                210,

                285

            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.72,

            (0, 255, 255),

            2,

            cv2.LINE_AA

        )


    # ========================================================
    # PINCH STATUS
    # ========================================================

    if (

        game_started

        and

        not game_over

    ):

        if pinch:

            cv2.putText(

                frame,

                "PINCHING",

                (

                    20,

                    height - 25

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2,

                cv2.LINE_AA

            )

        else:

            cv2.putText(

                frame,

                "PINCH TO MOVE",

                (

                    20,

                    height - 25

                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2,

                cv2.LINE_AA

            )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        WINDOW_NAME,

        frame

    )


    key = cv2.waitKey(1) & 0xFF


    if (

        key == ord("q")

        or

        key == 27

    ):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

landmarker.close()

cv2.destroyAllWindows()

print(
    "Hand Arcade closed."
)