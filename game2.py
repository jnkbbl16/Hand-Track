import os
import sys
import time
import math
import ctypes
import threading

import cv2
import numpy as np
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

CAM_SOURCE = "http://localhost:8080/video"

WINDOW_NAME = "Hand Rhythm - Tetris Theme"

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "hand_landmarker.task"
)

MUSIC_FILE = os.path.join(
    BASE_DIR,
    "Original Tetris theme (Tetris Soundtrack).mp3"
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# GAME SETTINGS
# ============================================================

NUM_LANES = 4

HIT_ZONE_X = 125

NOTE_WIDTH = 55
NOTE_HEIGHT = 55

# ------------------------------------------------------------
# NOTE SPEED
# ------------------------------------------------------------
#
# The game starts SLOW.
#
# As the song progresses, notes become faster.
#
# START_BPM:
#     Very comfortable beginning.
#
# MID_BPM:
#     Moderate gameplay.
#
# MAX_BPM:
#     Maximum speed later in the song.
#
# ------------------------------------------------------------

START_BPM = 60.0

MID_BPM = 105.0

MAX_BPM = 125.0

# How quickly difficulty increases.
#
# 0.0 = very slow increase
# 1.0 = reaches maximum speed near the end
#
DIFFICULTY_RAMP = 0.75


# ============================================================
# HIT WINDOWS
# ============================================================

PERFECT_WINDOW = 0.075

GREAT_WINDOW = 0.14

GOOD_WINDOW = 0.21

MISS_WINDOW = 0.28


# ============================================================
# OTHER GAME SETTINGS
# ============================================================

START_LIVES = 3

COUNTDOWN_SECONDS = 3.0

GESTURE_HIT_COOLDOWN = 0.18

FEEDBACK_DURATION = 0.55


# ============================================================
# MUSIC
# ============================================================

# Approximate length of the supplied track.

MUSIC_DURATION = 83.6

MUSIC_ALIAS = "TETRIS_RHYTHM"


# ============================================================
# LANE PATTERN
# ============================================================
#
# ☝️ ONE FINGER  = Lane 1
# ✌️ TWO FINGERS = Lane 2
# 🤟 I LOVE YOU  = Lane 3
# 🖐️ OPEN PALM   = Lane 4
# ✊ CLOSED FIST = Neutral
#
# ============================================================

LANE_PATTERN = [
    0, 1, 2, 3,
    0, 2, 1, 3,
    0, 1, 3, 2,
    1, 0, 2, 3,
    2, 3, 0, 1,
    3, 1, 2, 0
]


# ============================================================
# MEDIAPIPE
# ============================================================

mp_base = mp.tasks.BaseOptions

mp_vision = mp.tasks.vision

HandLandmarker = (
    mp_vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp_vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp_vision.RunningMode
)


# ============================================================
# WINDOWS MP3 PLAYER
# ============================================================

class WindowsMP3Player:

    def __init__(self):

        self.enabled = (
            sys.platform == "win32"
        )

        self.started = False

        self.start_time = None

        self._mci = None

        if self.enabled:

            try:

                self._mci = (
                    ctypes.windll.winmm
                    .mciSendStringW
                )

            except Exception:

                self.enabled = False

    def command(self, command):

        if not self.enabled:

            return False

        error_buffer = (
            ctypes.create_unicode_buffer(
                512
            )
        )

        result = self._mci(
            command,
            error_buffer,
            512,
            0
        )

        if result != 0:

            print(
                "MCI error:",
                error_buffer.value
            )

            return False

        return True

    def open(self):

        if not self.enabled:

            return False

        if not os.path.exists(
            MUSIC_FILE
        ):

            print()
            print(
                "ERROR: Music file not found:"
            )
            print(
                MUSIC_FILE
            )
            print()

            return False

        self.close()

        path = os.path.abspath(
            MUSIC_FILE
        ).replace(
            "\\",
            "/"
        )

        command = (
            f'open "{path}" '
            f'type mpegvideo '
            f'alias {MUSIC_ALIAS}'
        )

        return self.command(
            command
        )

    def play(self):

        if not self.enabled:

            return False

        if not self.open():

            return False

        success = self.command(
            f"play {MUSIC_ALIAS}"
        )

        if success:

            self.started = True

            self.start_time = (
                time.perf_counter()
            )

        return success

    def stop(self):

        if not self.enabled:

            return

        if self.started:

            self.command(
                f"stop {MUSIC_ALIAS}"
            )

        self.started = False

        self.start_time = None

    def close(self):

        if not self.enabled:

            return

        self.command(
            f"close {MUSIC_ALIAS}"
        )

        self.started = False

        self.start_time = None

    def get_elapsed(self):

        if (
            not self.started
            or self.start_time is None
        ):

            return 0.0

        return (
            time.perf_counter()
            -
            self.start_time
        )


# ============================================================
# LOW-LATENCY CAMERA
# ============================================================

class CameraStream:

    def __init__(self, source):

        self.source = source

        self.cap = None

        self.frame = None

        self.running = False

        self.lock = threading.Lock()

        self.thread = None

    def start(self):

        self.cap = cv2.VideoCapture(
            self.source,
            cv2.CAP_FFMPEG
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )

        self.running = True

        self.thread = threading.Thread(
            target=self._update,
            daemon=True
        )

        self.thread.start()

        return self

    def _update(self):

        while self.running:

            if self.cap is None:

                time.sleep(
                    0.01
                )

                continue

            ret, frame = (
                self.cap.read()
            )

            if not ret:

                time.sleep(
                    0.01
                )

                continue

            frame = cv2.resize(
                frame,
                (
                    FRAME_WIDTH,
                    FRAME_HEIGHT
                )
            )

            with self.lock:

                self.frame = frame

    def read(self):

        with self.lock:

            if self.frame is None:

                return None

            return self.frame.copy()

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=1.0
            )

        if self.cap is not None:

            self.cap.release()


# ============================================================
# GEOMETRY
# ============================================================

def distance(a, b):

    dx = a.x - b.x

    dy = a.y - b.y

    return math.sqrt(
        dx * dx
        +
        dy * dy
    )


# ============================================================
# FINGER DETECTION
# ============================================================

def finger_extended(
    landmarks,
    tip,
    pip,
    mcp
):

    tip_point = landmarks[tip]

    pip_point = landmarks[pip]

    mcp_point = landmarks[mcp]

    vertical_extended = (
        tip_point.y
        <
        pip_point.y - 0.015
    )

    tip_distance = distance(
        tip_point,
        landmarks[0]
    )

    pip_distance = distance(
        pip_point,
        landmarks[0]
    )

    distance_extended = (
        tip_distance
        >
        pip_distance * 1.10
    )

    return (
        vertical_extended
        and distance_extended
    )


def finger_folded(
    landmarks,
    tip,
    pip,
    mcp
):

    tip_point = landmarks[tip]

    pip_point = landmarks[pip]

    tip_distance = distance(
        tip_point,
        landmarks[0]
    )

    pip_distance = distance(
        pip_point,
        landmarks[0]
    )

    vertical_folded = (
        tip_point.y
        >=
        pip_point.y - 0.015
    )

    distance_folded = (
        tip_distance
        <=
        pip_distance * 1.18
    )

    return (
        vertical_folded
        or distance_folded
    )


def thumb_extended(landmarks):

    thumb_tip = landmarks[4]

    thumb_ip = landmarks[3]

    thumb_mcp = landmarks[2]

    tip_distance = distance(
        thumb_tip,
        landmarks[0]
    )

    ip_distance = distance(
        thumb_ip,
        landmarks[0]
    )

    horizontal_extension = (
        abs(
            thumb_tip.x
            -
            thumb_mcp.x
        )
        >
        0.075
    )

    distance_extension = (
        tip_distance
        >
        ip_distance * 1.06
    )

    return (
        horizontal_extension
        or distance_extension
    )


# ============================================================
# GESTURE DETECTION
# ============================================================

def get_gesture(landmarks):

    index_up = finger_extended(
        landmarks,
        8,
        6,
        5
    )

    middle_up = finger_extended(
        landmarks,
        12,
        10,
        9
    )

    ring_up = finger_extended(
        landmarks,
        16,
        14,
        13
    )

    pinky_up = finger_extended(
        landmarks,
        20,
        18,
        17
    )

    index_down = finger_folded(
        landmarks,
        8,
        6,
        5
    )

    middle_down = finger_folded(
        landmarks,
        12,
        10,
        9
    )

    ring_down = finger_folded(
        landmarks,
        16,
        14,
        13
    )

    pinky_down = finger_folded(
        landmarks,
        20,
        18,
        17
    )

    thumb_up = thumb_extended(
        landmarks
    )

    # ========================================================
    # 🤟 I LOVE YOU
    # ========================================================

    if (
        thumb_up
        and index_up
        and not middle_up
        and not ring_up
        and pinky_up
    ):

        return "ILY"

    # ========================================================
    # 🖐️ OPEN PALM
    # ========================================================

    if (
        thumb_up
        and index_up
        and middle_up
        and ring_up
        and pinky_up
    ):

        return "PALM"

    # ========================================================
    # ✌️ TWO FINGERS
    # ========================================================

    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):

        return "TWO"

    # ========================================================
    # ☝️ ONE FINGER
    # ========================================================

    if (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):

        return "ONE"

    # ========================================================
    # ✊ CLOSED FIST
    # ========================================================

    if (
        index_down
        and middle_down
        and ring_down
        and pinky_down
        and not thumb_up
    ):

        return "FIST"

    return "NONE"


# ============================================================
# GESTURE → LANE
# ============================================================

def gesture_to_lane(gesture):

    if gesture == "ONE":

        return 0

    if gesture == "TWO":

        return 1

    if gesture == "ILY":

        return 2

    if gesture == "PALM":

        return 3

    return None


# ============================================================
# GESTURE NAME
# ============================================================

def gesture_name(gesture):

    if gesture == "ONE":

        return "☝️ LANE 1"

    if gesture == "TWO":

        return "✌️ LANE 2"

    if gesture == "ILY":

        return "🤟 LANE 3"

    if gesture == "PALM":

        return "🖐️ LANE 4"

    if gesture == "FIST":

        return "✊ NEUTRAL"

    return "NO GESTURE"


# ============================================================
# TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.6,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


def draw_centered_text(
    frame,
    text,
    y,
    scale=1.0,
    thickness=2
):

    size = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )[0]

    x = (
        FRAME_WIDTH
        -
        size[0]
    ) // 2

    draw_text(
        frame,
        text,
        (
            x,
            y
        ),
        scale,
        thickness
    )


# ============================================================
# TRANSPARENT RECTANGLE
# ============================================================

def draw_transparent_rect(
    frame,
    x1,
    y1,
    x2,
    y2,
    alpha=0.35
):

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1 - alpha,
        0,
        frame
    )


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    frame,
    landmarks
):

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    points = []

    for lm in landmarks:

        x = int(
            lm.x
            *
            FRAME_WIDTH
        )

        y = int(
            lm.y
            *
            FRAME_HEIGHT
        )

        points.append(
            (
                x,
                y
            )
        )

        cv2.circle(
            frame,
            (
                x,
                y
            ),
            4,
            (255, 255, 255),
            -1
        )

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 255, 255),
            2
        )


# ============================================================
# DRAW LANES
# ============================================================

def draw_lanes(frame):

    lane_height = (
        FRAME_HEIGHT
        //
        NUM_LANES
    )

    for lane in range(
        NUM_LANES
    ):

        y1 = (
            lane
            *
            lane_height
        )

        y2 = (
            (lane + 1)
            *
            lane_height
        )

        draw_transparent_rect(
            frame,
            0,
            y1,
            FRAME_WIDTH,
            y2,
            0.20
        )

        cv2.line(
            frame,
            (
                0,
                y2
            ),
            (
                FRAME_WIDTH,
                y2
            ),
            (255, 255, 255),
            1
        )

        draw_text(
            frame,
            f"LANE {lane + 1}",
            (
                10,
                y1 + 25
            ),
            0.45,
            1
        )

    # Hit line

    cv2.line(
        frame,
        (
            HIT_ZONE_X,
            0
        ),
        (
            HIT_ZONE_X,
            FRAME_HEIGHT
        ),
        (255, 255, 255),
        4
    )

    draw_text(
        frame,
        "HIT",
        (
            HIT_ZONE_X - 18,
            25
        ),
        0.50,
        2
    )


# ============================================================
# SMOOTH DIFFICULTY FUNCTION
# ============================================================

def get_bpm(elapsed):

    # --------------------------------------------------------
    # Normalize time from 0.0 to 1.0
    # --------------------------------------------------------

    progress = (
        elapsed
        /
        MUSIC_DURATION
    )

    progress = max(
        0.0,
        min(
            1.0,
            progress
        )
    )

    # --------------------------------------------------------
    # Difficulty curve
    #
    # This makes the beginning much slower.
    #
    # smoothstep:
    #
    # 0 -> slow
    # 1 -> fast
    #
    # --------------------------------------------------------

    smooth_progress = (
        progress
        *
        progress
        *
        (
            3.0
            -
            2.0
            *
            progress
        )
    )

    # Apply ramp.

    smooth_progress *= (
        DIFFICULTY_RAMP
        +
        (
            1.0
            -
            DIFFICULTY_RAMP
        )
        *
        progress
    )

    if progress < 0.30:

        # Early section:
        # START_BPM -> MID_BPM

        early_progress = (
            progress
            /
            0.30
        )

        early_progress = (
            early_progress
            *
            early_progress
        )

        bpm = (
            START_BPM
            +
            (
                MID_BPM
                -
                START_BPM
            )
            *
            early_progress
        )

    else:

        # Later section:
        # MID_BPM -> MAX_BPM

        late_progress = (
            (
                progress
                -
                0.30
            )
            /
            0.70
        )

        late_progress = max(
            0.0,
            min(
                1.0,
                late_progress
            )
        )

        late_progress = (
            late_progress
            *
            late_progress
            *
            (
                3.0
                -
                2.0
                *
                late_progress
            )
        )

        bpm = (
            MID_BPM
            +
            (
                MAX_BPM
                -
                MID_BPM
            )
            *
            late_progress
        )

    return bpm


# ============================================================
# CREATE DYNAMIC NOTE MAP
# ============================================================

def create_notes():

    notes = []

    current_time = 0.65

    beat_index = 0

    while current_time < MUSIC_DURATION:

        lane = LANE_PATTERN[
            beat_index
            %
            len(LANE_PATTERN)
        ]

        notes.append(
            {
                "time": current_time,
                "lane": lane,
                "hit": False,
                "missed": False
            }
        )

        # ----------------------------------------------------
        # Calculate BPM at THIS point in the song.
        # ----------------------------------------------------

        bpm = get_bpm(
            current_time
        )

        beat_interval = (
            60.0
            /
            bpm
        )

        # ----------------------------------------------------
        # Small additional spacing at the very beginning.
        #
        # This prevents the opening from feeling crowded.
        # ----------------------------------------------------

        if current_time < 8.0:

            beat_interval *= 1.12

        elif current_time < 15.0:

            beat_interval *= 1.06

        current_time += (
            beat_interval
        )

        beat_index += 1

    return notes


# ============================================================
# DRAW NOTE
# ============================================================

def draw_note(
    frame,
    note,
    elapsed
):

    lane = note["lane"]

    hit_time = note["time"]

    time_until_hit = (
        hit_time
        -
        elapsed
    )

    x = (
        HIT_ZONE_X
        +
        (
            time_until_hit
            /
            2.0
        )
        *
        (
            FRAME_WIDTH
            -
            HIT_ZONE_X
            +
            NOTE_WIDTH
        )
    )

    lane_height = (
        FRAME_HEIGHT
        //
        NUM_LANES
    )

    y_center = (
        lane
        *
        lane_height
        +
        lane_height // 2
    )

    x1 = int(
        x
        -
        NOTE_WIDTH / 2
    )

    y1 = int(
        y_center
        -
        NOTE_HEIGHT / 2
    )

    x2 = int(
        x
        +
        NOTE_WIDTH / 2
    )

    y2 = int(
        y_center
        +
        NOTE_HEIGHT / 2
    )

    if (
        x2 < 0
        or x1 > FRAME_WIDTH
    ):

        return

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        frame,
        (
            x1,
            y1
        ),
        (
            x2,
            y2
        ),
        (255, 255, 255),
        2
    )


# ============================================================
# FEEDBACK
# ============================================================

def add_feedback(
    effects,
    text
):

    effects.append(
        {
            "text": text,
            "start": time.perf_counter()
        }
    )


def draw_feedback(
    frame,
    effects
):

    now = time.perf_counter()

    active = []

    for effect in effects:

        age = (
            now
            -
            effect["start"]
        )

        if age < FEEDBACK_DURATION:

            y = int(
                210
                -
                age * 45
            )

            draw_centered_text(
                frame,
                effect["text"],
                y,
                0.85,
                2
            )

            active.append(
                effect
            )

    effects[:] = active


# ============================================================
# ATTEMPT HIT
# ============================================================

def attempt_hit(
    lane,
    elapsed,
    notes,
    state,
    effects
):

    best_note = None

    best_difference = (
        float("inf")
    )

    for note in notes:

        if note["hit"]:

            continue

        if note["missed"]:

            continue

        if note["lane"] != lane:

            continue

        difference = abs(
            elapsed
            -
            note["time"]
        )

        if (
            difference
            <
            best_difference
        ):

            best_difference = (
                difference
            )

            best_note = note

    if best_note is None:

        return False

    # PERFECT

    if (
        best_difference
        <=
        PERFECT_WINDOW
    ):

        best_note["hit"] = True

        state["score"] += 3

        state["combo"] += 1

        state["max_combo"] = max(
            state["max_combo"],
            state["combo"]
        )

        add_feedback(
            effects,
            "PERFECT! +3"
        )

        return True

    # GREAT

    if (
        best_difference
        <=
        GREAT_WINDOW
    ):

        best_note["hit"] = True

        state["score"] += 2

        state["combo"] += 1

        state["max_combo"] = max(
            state["max_combo"],
            state["combo"]
        )

        add_feedback(
            effects,
            "GREAT! +2"
        )

        return True

    # GOOD

    if (
        best_difference
        <=
        GOOD_WINDOW
    ):

        best_note["hit"] = True

        state["score"] += 1

        state["combo"] += 1

        state["max_combo"] = max(
            state["max_combo"],
            state["combo"]
        )

        add_feedback(
            effects,
            "GOOD! +1"
        )

        return True

    return False


# ============================================================
# MISSES
# ============================================================

def process_misses(
    elapsed,
    notes,
    state,
    effects
):

    for note in notes:

        if note["hit"]:

            continue

        if note["missed"]:

            continue

        if (
            elapsed
            >
            note["time"]
            +
            MISS_WINDOW
        ):

            note["missed"] = True

            state["lives"] -= 1

            state["combo"] = 0

            add_feedback(
                effects,
                "MISS!"
            )

            if (
                state["lives"]
                <=
                0
            ):

                state["game_over"] = True


# ============================================================
# HUD
# ============================================================

def draw_hud(
    frame,
    state,
    gesture,
    music_elapsed
):

    draw_transparent_rect(
        frame,
        0,
        0,
        FRAME_WIDTH,
        70,
        0.60
    )

    draw_text(
        frame,
        f"Score: {state['score']}",
        (
            12,
            27
        ),
        0.62,
        2
    )

    draw_text(
        frame,
        f"Combo: {state['combo']}",
        (
            12,
            55
        ),
        0.52,
        2
    )

    draw_text(
        frame,
        f"Lives: {state['lives']}",
        (
            165,
            27
        ),
        0.62,
        2
    )

    draw_text(
        frame,
        gesture_name(gesture),
        (
            165,
            55
        ),
        0.46,
        2
    )

    # Current BPM

    bpm = get_bpm(
        music_elapsed
    )

    draw_text(
        frame,
        f"BPM: {bpm:.0f}",
        (
            300,
            27
        ),
        0.55,
        2
    )

    draw_text(
        frame,
        f"{music_elapsed:05.1f}s",
        (
            FRAME_WIDTH - 75,
            27
        ),
        0.45,
        1
    )

    draw_text(
        frame,
        "M = Menu",
        (
            FRAME_WIDTH - 75,
            55
        ),
        0.40,
        1
    )


# ============================================================
# COUNTDOWN
# ============================================================

def draw_countdown(
    frame,
    remaining
):

    number = int(
        math.ceil(
            remaining
        )
    )

    if number < 1:

        number = 1

    draw_transparent_rect(
        frame,
        0,
        0,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        0.50
    )

    draw_centered_text(
        frame,
        "GET READY!",
        180,
        1.05,
        2
    )

    draw_centered_text(
        frame,
        str(number),
        295,
        2.5,
        5
    )

    draw_centered_text(
        frame,
        "START SLOW → GET FASTER",
        355,
        0.60,
        2
    )


# ============================================================
# GAME OVER
# ============================================================

def draw_game_over(
    frame,
    state
):

    draw_transparent_rect(
        frame,
        0,
        0,
        FRAME_WIDTH,
        FRAME_HEIGHT,
        0.65
    )

    draw_centered_text(
        frame,
        "YOU LOSE",
        165,
        1.55,
        4
    )

    draw_centered_text(
        frame,
        f"Score: {state['score']}",
        225,
        0.85,
        2
    )

    draw_centered_text(
        frame,
        f"Max Combo: {state['max_combo']}",
        265,
        0.65,
        2
    )

    draw_centered_text(
        frame,
        "Press R to restart",
        335,
        0.65,
        2
    )

    draw_centered_text(
        frame,
        "Press M for menu",
        375,
        0.65,
        2
    )

    draw_centered_text(
        frame,
        "Press ESC to quit",
        415,
        0.55,
        2
    )


# ============================================================
# MAIN GAME
# ============================================================

def run_game():

    print()
    print("=" * 60)
    print("HAND RHYTHM - TETRIS THEME")
    print("=" * 60)
    print()
    print("GESTURES:")
    print()
    print("  ☝️  One finger  -> Lane 1")
    print("  ✌️  Two fingers -> Lane 2")
    print("  🤟 I Love You  -> Lane 3")
    print("  🖐️  Open palm   -> Lane 4")
    print("  ✊ Closed fist -> Neutral")
    print()
    print("DIFFICULTY:")
    print()
    print(f"  Starting BPM : {START_BPM}")
    print(f"  Middle BPM   : {MID_BPM}")
    print(f"  Maximum BPM  : {MAX_BPM}")
    print()
    print("KEYBOARD:")
    print("  M   -> Return to menu")
    print("  R   -> Restart")
    print("  ESC -> Quit")
    print()

    # ========================================================
    # MUSIC
    # ========================================================

    music = WindowsMP3Player()

    if not music.enabled:

        print(
            "WARNING: Windows MP3 playback "
            "is unavailable."
        )

    elif not os.path.exists(
        MUSIC_FILE
    ):

        print()
        print(
            "WARNING: MP3 file not found:"
        )
        print(
            MUSIC_FILE
        )
        print()

    else:

        print(
            "Music file found."
        )

        print(
            "Using Windows Multimedia API."
        )

    # ========================================================
    # CAMERA
    # ========================================================

    camera = CameraStream(
        CAM_SOURCE
    ).start()

    # ========================================================
    # MODEL
    # ========================================================

    if not os.path.exists(
        MODEL_FILE
    ):

        print(
            "ERROR: hand_landmarker.task "
            "was not found."
        )

        camera.stop()

        return False

    # ========================================================
    # MEDIAPIPE OPTIONS
    # ========================================================

    options = HandLandmarkerOptions(

        base_options=mp_base(
            model_asset_path=MODEL_FILE
        ),

        running_mode=(
            VisionRunningMode.VIDEO
        ),

        num_hands=1,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5
    )

    # ========================================================
    # GAME STATE
    # ========================================================

    state = {
        "score": 0,
        "combo": 0,
        "max_combo": 0,
        "lives": START_LIVES,
        "game_over": False
    }

    notes = create_notes()

    effects = []

    previous_gesture = "NONE"

    last_hit_time = 0.0

    countdown_start = (
        time.perf_counter()
    )

    game_started = False

    returned_to_menu = False

    timestamp_ms = 0

    # ========================================================
    # FALLBACK TIMER
    # ========================================================

    fallback_start = None

    # ========================================================
    # LANDMARKER
    # ========================================================

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        try:

            while True:

                # ====================================================
                # CAMERA
                # ====================================================

                frame = camera.read()

                if frame is None:

                    blank = np.zeros(
                        (
                            FRAME_HEIGHT,
                            FRAME_WIDTH,
                            3
                        ),
                        dtype=np.uint8
                    )

                    draw_centered_text(
                        blank,
                        "Waiting for camera...",
                        240,
                        0.75,
                        2
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        blank
                    )

                    key = (
                        cv2.waitKey(1)
                        &
                        0xFF
                    )

                    if key == 27:

                        break

                    continue

                # Mirror camera.

                frame = cv2.flip(
                    frame,
                    1
                )

                # ====================================================
                # MEDIAPIPE
                # ====================================================

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb
                )

                timestamp_ms += 33

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )
                )

                gesture = "NONE"

                if result.hand_landmarks:

                    landmarks = (
                        result.hand_landmarks[0]
                    )

                    draw_hand(
                        frame,
                        landmarks
                    )

                    gesture = get_gesture(
                        landmarks
                    )

                # ====================================================
                # COUNTDOWN
                # ====================================================

                if not game_started:

                    countdown_elapsed = (
                        time.perf_counter()
                        -
                        countdown_start
                    )

                    remaining = (
                        COUNTDOWN_SECONDS
                        -
                        countdown_elapsed
                    )

                    draw_lanes(
                        frame
                    )

                    draw_hud(
                        frame,
                        state,
                        gesture,
                        0.0
                    )

                    if remaining > 0:

                        draw_countdown(
                            frame,
                            remaining
                        )

                    else:

                        # --------------------------------------------
                        # Start music.
                        # --------------------------------------------

                        game_started = True

                        fallback_start = (
                            time.perf_counter()
                        )

                        if os.path.exists(
                            MUSIC_FILE
                        ):

                            music.play()

                    cv2.imshow(
                        WINDOW_NAME,
                        frame
                    )

                    key = (
                        cv2.waitKey(1)
                        &
                        0xFF
                    )

                    if key == 27:

                        break

                    if key in (
                        ord("m"),
                        ord("M")
                    ):

                        returned_to_menu = True

                        break

                    continue

                # ====================================================
                # GAME OVER
                # ====================================================

                if state["game_over"]:

                    music.stop()

                    elapsed = (
                        music.get_elapsed()
                    )

                    draw_lanes(
                        frame
                    )

                    draw_hud(
                        frame,
                        state,
                        gesture,
                        elapsed
                    )

                    draw_game_over(
                        frame,
                        state
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        frame
                    )

                    key = (
                        cv2.waitKey(1)
                        &
                        0xFF
                    )

                    # --------------------------------------------
                    # RESTART
                    # --------------------------------------------

                    if key in (
                        ord("r"),
                        ord("R")
                    ):

                        music.stop()

                        music.close()

                        state = {
                            "score": 0,
                            "combo": 0,
                            "max_combo": 0,
                            "lives": START_LIVES,
                            "game_over": False
                        }

                        notes = create_notes()

                        effects.clear()

                        previous_gesture = "NONE"

                        last_hit_time = 0.0

                        countdown_start = (
                            time.perf_counter()
                        )

                        fallback_start = None

                        game_started = False

                    # --------------------------------------------
                    # MENU
                    # --------------------------------------------

                    elif key in (
                        ord("m"),
                        ord("M")
                    ):

                        returned_to_menu = True

                        break

                    # --------------------------------------------
                    # EXIT
                    # --------------------------------------------

                    elif key == 27:

                        break

                    continue

                # ====================================================
                # MUSIC TIME
                # ====================================================

                elapsed = (
                    music.get_elapsed()
                )

                if elapsed <= 0:

                    if fallback_start is None:

                        fallback_start = (
                            time.perf_counter()
                        )

                    elapsed = (
                        time.perf_counter()
                        -
                        fallback_start
                    )

                # ====================================================
                # END OF SONG
                # ====================================================

                if (
                    elapsed
                    >=
                    MUSIC_DURATION
                ):

                    music.stop()

                    state["game_over"] = True

                    continue

                # ====================================================
                # DRAW LANES
                # ====================================================

                draw_lanes(
                    frame
                )

                # ====================================================
                # DRAW NOTES
                # ====================================================

                for note in notes:

                    if note["hit"]:

                        continue

                    if note["missed"]:

                        continue

                    if (
                        elapsed
                        >=
                        note["time"]
                        -
                        2.0
                    ):

                        draw_note(
                            frame,
                            note,
                            elapsed
                        )

                # ====================================================
                # GESTURE HIT
                # ====================================================

                gesture_changed = (
                    gesture
                    !=
                    previous_gesture
                )

                if (
                    gesture_changed
                    and gesture not in (
                        "NONE",
                        "FIST"
                    )
                ):

                    now = (
                        time.perf_counter()
                    )

                    if (
                        now
                        -
                        last_hit_time
                        >=
                        GESTURE_HIT_COOLDOWN
                    ):

                        lane = (
                            gesture_to_lane(
                                gesture
                            )
                        )

                        if lane is not None:

                            hit = attempt_hit(
                                lane,
                                elapsed,
                                notes,
                                state,
                                effects
                            )

                            if hit:

                                last_hit_time = (
                                    now
                                )

                previous_gesture = gesture

                # ====================================================
                # MISSES
                # ====================================================

                process_misses(
                    elapsed,
                    notes,
                    state,
                    effects
                )

                # ====================================================
                # HUD
                # ====================================================

                draw_hud(
                    frame,
                    state,
                    gesture,
                    elapsed
                )

                # ====================================================
                # FEEDBACK
                # ====================================================

                draw_feedback(
                    frame,
                    effects
                )

                # ====================================================
                # INSTRUCTIONS
                # ====================================================

                draw_transparent_rect(
                    frame,
                    0,
                    FRAME_HEIGHT - 35,
                    FRAME_WIDTH,
                    FRAME_HEIGHT,
                    0.65
                )

                draw_centered_text(
                    frame,
                    "ONE   TWO   ILY   PALM   |   FIST = NEUTRAL",
                    FRAME_HEIGHT - 11,
                    0.39,
                    1
                )

                # ====================================================
                # CAMERA LABEL
                # ====================================================

                draw_text(
                    frame,
                    "LIVE CAMERA",
                    (
                        FRAME_WIDTH - 105,
                        92
                    ),
                    0.38,
                    1
                )

                # ====================================================
                # DISPLAY
                # ====================================================

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )

                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )

                # ====================================================
                # ESC
                # ====================================================

                if key == 27:

                    break

                # ====================================================
                # MENU
                # ====================================================

                if key in (
                    ord("m"),
                    ord("M")
                ):

                    returned_to_menu = True

                    break

        finally:

            music.stop()

            music.close()

            camera.stop()

            cv2.destroyAllWindows()

    return returned_to_menu


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    run_game()