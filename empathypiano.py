# BEFORE RUNNING MAKE SURE ALL THE LIBRARIES IN REQUIREMENTS.TXT ARE INSTALLED ON YOUR PYTHON VERSION

import pygame # used for the gui
import numpy as np # used for changing audio
import threading # allows us to do emotion detection and piano at same time
import cv2 # computer vision to detect camera input
import wave # reads .wav sound files
import io
from scipy.signal import resample # used for pitch shifting
import keyboard
import librosa
import time
import os
import serial

pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

current_emotion = "neutral" # the default is neutral so it doesn't start as happy or sad
stop_camera = False # stop camera will be true when program is turned off
update_emotion_enabled = False
current_octave = 0
camera_ready = False
gesture_state = None
emotion_mode = "hold"

note_frequencies = {}
cached_sounds = {}

# each note is assigned a value for shifting the pitch
# F is the base note in the middle, E is shifted down by 1, F# shifted up, etc.
NOTE_MAP = {
    'C': -5, 'C#': -4, 'D': -3, 'D#': -2, 'E': -1,
    'F': 0, 'F#': 1, 'G': 2, 'G#': 3, 'A': 4, 'A#': 5, 'B': 6
}

# every key on the keyboard is assigned a note on the piano
KEY_BINDINGS = {
    pygame.K_a: 'C', pygame.K_w: 'C#', pygame.K_s: 'D',
    pygame.K_e: 'D#', pygame.K_d: 'E', pygame.K_f: 'F',
    pygame.K_t: 'F#', pygame.K_g: 'G', pygame.K_y: 'G#',
    pygame.K_h: 'A', pygame.K_u: 'A#', pygame.K_j: 'B'
}

# corresponding keyboard key to piano note for key labels in the gui
KEY_LABELS = {
    'C': 'A', 'C#': 'W', 'D': 'S', 'D#': 'E', 'E': 'D',
    'F': 'F', 'F#': 'T', 'G': 'G', 'G#': 'Y',
    'A': 'H', 'A#': 'U', 'B': 'J'
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["SDL_AUDIODRIVER"] = "dummy"
os.environ["SDL_VIDEODRIVER"] = ""

def emotion_thread():
    global current_emotion, stop_camera, gesture_state, update_emotion_enabled, camera_ready
    from deepface import DeepFace
    import mediapipe as mp
    import tensorflow as tf

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    camera_ready = True
    current_emotion = "neutral"

    def is_thumbs_up(hand_landmarks):
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        return thumb_tip.y < index_mcp.y and thumb_tip.y < thumb_ip.y

    def is_thumbs_down(hand_landmarks):
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
        index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        return thumb_tip.y > index_mcp.y and thumb_tip.y > thumb_ip.y

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if update_emotion_enabled:
            for (x, y, w, h) in faces:
                face_roi = rgb[y:y + h, x:x + w]
                try:
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    current_emotion = result[0]['dominant_emotion']
                except Exception:
                    pass
                break

        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if is_thumbs_up(hand_landmarks):
                    gesture_state = "thumbs_up"
                elif is_thumbs_down(hand_landmarks):
                    gesture_state = "thumbs_down"
                else:
                    gesture_state = None
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            gesture_state = None
            
        # Uncomment this for camera window
        # cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera = True
            break

    cap.release()
    cv2.destroyAllWindows()

def arduino_thread():
    global pressed_keys

    ports_to_try = ["COM3", "COM4", "COM5"]
    arduino = None

    for port_name in ports_to_try:
        try:
            arduino = serial.Serial(port_name, 9600, timeout=1)
            print(f"Successfully connected to Arduino on {port_name}")
            break
        except Exception:
            continue

    if arduino is None:
        print("Arduino connection error: Not found")
        return

    button_note_map = {
        0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E',
        5: 'F', 6: 'F#', 7: 'G', 8: 'G#', 9: 'A',
        10: 'A#', 11: 'B'
    }

    try:
        while True:
            if arduino.in_waiting > 0:
                line = arduino.readline().decode(errors="ignore").strip()

                if line.startswith("PRESS"):
                    idx = int(line.split()[1])
                    note = button_note_map.get(idx)
                    if note:
                        pressed_keys.add(note)
                        play_note(note)

                elif line.startswith("RELEASE"):
                    idx = int(line.split()[1])
                    note = button_note_map.get(idx)
                    if note in pressed_keys:
                        pressed_keys.discard(note)

    except Exception as e:
        print(f"Arduino connection error: {e}")


def detect_base_frequency(audio: np.ndarray, rate: int) -> float:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    segment = audio[: min(len(audio), rate * 2)]
    spectrum = np.fft.rfft(segment)
    freqs = np.fft.rfftfreq(len(segment), d=1 / rate)
    peak = freqs[np.argmax(np.abs(spectrum))]
    return peak

def load_wav(filename):
    with wave.open(filename, 'rb') as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels)
        audio /= np.iinfo(np.int16).max
    return audio, rate, channels

def generate_note(note: str, base_audio: np.ndarray, base_rate: int, channels: int, base_freq: float):
    semitones = NOTE_MAP[note]
    if channels > 1:
        audio_mono = np.mean(base_audio, axis=1)
    else:
        audio_mono = base_audio
        
    shifted = librosa.effects.pitch_shift(audio_mono, sr=base_rate, n_steps=semitones)
    fade_len = min(500, len(shifted))
    fade_out = np.linspace(1, 0, fade_len)
    shifted[-fade_len:] *= fade_out

    shifted /= max(1e-9, np.max(np.abs(shifted)))
    shifted = (shifted * 32767).astype(np.int16)

    virtual_wav = io.BytesIO()
    with wave.open(virtual_wav, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(base_rate)
        wf.writeframes(shifted.tobytes())
    virtual_wav.seek(0)
    sound = pygame.mixer.Sound(virtual_wav)
    return sound

def load_sample_for_emotion(emotion_name=None):
    global current_emotion
    emotion_files = {
        "ANGRY": "sounds/angry.wav",
        "DISGUSTED": "sounds/disgusting.wav",
        "FEAR": "sounds/fearful.wav",
        "HAPPY": "sounds/happy.wav",
        "NEUTRAL": "sounds/neutral.wav",
        "SAD": "sounds/sad.wav",
        "SURPRISE": "sounds/surprised.wav"
    }

    emotion_key = (emotion_name or current_emotion).upper()
    base_file = emotion_files.get(emotion_key, "sounds/new.wav")

    if not os.path.exists(base_file):
        base_file = "sounds/new.wav"

    base_audio, base_rate, channels = load_wav(base_file)
    base_freq = detect_base_frequency(base_audio, base_rate)

    sounds = {}
    for n in NOTE_MAP.keys():
        sounds[n] = generate_note(n, base_audio, base_rate, channels, base_freq)
    return sounds

def preload_all_sounds():
    global cached_sounds

    emotions = ["angry", "disgusted", "fear", "happy", "neutral", "sad", "surprise"]
    octaves = [-1, 0, 1]
    total = len(emotions) * len(octaves)
    count = 0
    start_all = time.time()

    for emotion in emotions:
        if emotion.upper() not in cached_sounds:
            cached_sounds[emotion.upper()] = load_sample_for_emotion(emotion)
        base_sounds = cached_sounds[emotion.upper()]

        for octave_shift in octaves:
            cache_key = f"{emotion.upper()}_{octave_shift}"
            if cache_key in cached_sounds:
                count += 1
                continue

            shifted_sounds = {}
            for n, s in base_sounds.items():
                arr = pygame.sndarray.array(s).astype(np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
                arr /= max(1e-9, np.max(np.abs(arr)))

                if octave_shift != 0:
                    try:
                        shifted = librosa.effects.pitch_shift(arr, sr=44100, n_steps=octave_shift * 12)
                    except Exception:
                        shifted = arr
                else:
                    shifted = arr

                shifted = np.clip(shifted, -1.0, 1.0)
                shifted = (shifted * 32767).astype(np.int16)
                virtual_wav = io.BytesIO()
                with wave.open(virtual_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    wf.writeframes(shifted.tobytes())
                virtual_wav.seek(0)
                shifted_sounds[n] = pygame.mixer.Sound(virtual_wav)

            cached_sounds[cache_key] = shifted_sounds
            count += 1

def play_note(note):
    global cached_sounds, current_octave

    try:
        cache_key = f"{current_emotion.upper()}_{current_octave}"

        if cache_key not in cached_sounds:
            if current_emotion.upper() not in cached_sounds:
                cached_sounds[current_emotion.upper()] = load_sample_for_emotion()

            base_sounds = cached_sounds[current_emotion.upper()]
            shifted_sounds = {}
            for n, s in base_sounds.items():
                arr = pygame.sndarray.array(s).astype(np.float32)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
                arr /= max(1e-9, np.max(np.abs(arr)))

                if current_octave != 0:
                    try:
                        shifted = librosa.effects.pitch_shift(arr, sr=44100, n_steps=current_octave * 12)
                    except Exception as e:
                        print("Pitch shift error:", e)
                        shifted = arr
                else:
                    shifted = arr

                shifted = np.clip(shifted, -1.0, 1.0)
                shifted = (shifted * 32767).astype(np.int16)

                virtual_wav = io.BytesIO()
                with wave.open(virtual_wav, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(44100)
                    wf.writeframes(shifted.tobytes())
                virtual_wav.seek(0)
                shifted_sounds[n] = pygame.mixer.Sound(virtual_wav)

            cached_sounds[cache_key] = shifted_sounds

        sounds = cached_sounds[cache_key]
        sound = sounds[note]
        channel = pygame.mixer.find_channel(True)
        if channel:
            channel.play(sound)

    except Exception as e:
        print("play_note error:", e)

def loading_screen(message="Loading..."):
    font = pygame.font.SysFont("Heebo", 90)
    screen.fill((65, 65, 65))
    label = font.render(message, True, (255, 255, 255))
    screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - 50))
    pygame.display.flip()
    pygame.event.pump()

info = pygame.display.Info()
WIDTH, HEIGHT = info.current_w, info.current_h
screen = pygame.display.set_mode((WIDTH, HEIGHT-60))
pygame.display.set_caption("Empathy Piano Digital")

biggest_font = pygame.font.SysFont("Heebo", int(HEIGHT * 0.15))
big_font = pygame.font.SysFont("Heebo", int(HEIGHT * 0.08))
font = pygame.font.SysFont("Heebo", int(HEIGHT * 0.03))

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GREEN = (144, 238, 144)
ANGRY = (135, 14, 14)
DISGUSTED = (49, 135, 59)
FEAR = (147, 104, 166)
HAPPY = (252, 192, 40)
NEUTRAL = (65, 65, 65)
SAD = (0, 0, 139)
SURPRISE = (171, 245, 244)

EMOTION_COLORS = {
    "ANGRY": ANGRY,
    "DISGUSTED": DISGUSTED,
    "FEAR": FEAR,
    "HAPPY": HAPPY,
    "NEUTRAL": NEUTRAL,
    "SAD": SAD,
    "SURPRISE": SURPRISE
}

piano_width = WIDTH * 0.7
piano_height = HEIGHT // 2.05
piano_x = (WIDTH - piano_width) // 2
piano_y = (HEIGHT - piano_height) // 2.2

key_width = piano_width // 7
key_height = piano_height
black_key_width = key_width // 2
black_key_height = piano_height * 0.6

white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
black_keys = ['C#', 'D#', '', 'F#', 'G#', 'A#', '']

white_rects = {}
black_rects = {}
pressed_keys = set()
BUTTON_WIDTH = WIDTH * 0.08
BUTTON_HEIGHT = HEIGHT * 0.05
BUTTON_X = WIDTH * 0.46
BUTTON_Y = HEIGHT * 0.85

emotion_button_rect = pygame.Rect(BUTTON_X, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT)

def draw_piano():
    color = EMOTION_COLORS.get(current_emotion.upper(), NEUTRAL)
    screen.fill((color))
    title_label = biggest_font.render("THE " + current_emotion.upper() + " PIANO", True, WHITE)
    octave_label = big_font.render(f"OCTAVE: {current_octave:+d}", True, WHITE)

    screen.blit(title_label, (WIDTH * 0.5 - title_label.get_width() / 2, HEIGHT * 0.05))
    screen.blit(octave_label, (WIDTH * 0.5 - octave_label.get_width() / 2, HEIGHT * 0.75))


    for i, note in enumerate(white_keys):
        rect = pygame.Rect(piano_x + i * key_width, piano_y, key_width, key_height)
        color = LIGHT_GREEN if note in pressed_keys else WHITE
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        white_rects[note] = rect
        label = font.render(KEY_LABELS[note], True, BLACK)
        screen.blit(label, (rect.centerx - label.get_width() / 2, rect.bottom - HEIGHT * 0.06))

    for i, note in enumerate(black_keys):
        if note != '':
            rect = pygame.Rect(piano_x + i * key_width + key_width * 0.7, piano_y, black_key_width, black_key_height)
            color = LIGHT_GREEN if note in pressed_keys else BLACK
            pygame.draw.rect(screen, color, rect)
            black_rects[note] = rect
            label = font.render(KEY_LABELS[note], True, WHITE)
            screen.blit(label, (rect.centerx - label.get_width() / 2, rect.bottom - HEIGHT * 0.06))

    button_text = "HOLD" if emotion_mode == "hold" else "LIVE"
    button_color = (200, 200, 200) if emotion_mode == "hold" else (255, 0, 0)

    pygame.draw.rect(screen, button_color, emotion_button_rect)
    pygame.draw.rect(screen, BLACK, emotion_button_rect, 2)

    label = font.render(button_text, True, BLACK)
    screen.blit(label,(emotion_button_rect.centerx - label.get_width() / 2,emotion_button_rect.centery - label.get_height() / 2))
    
running = True
camera_thread = None

def main():
    global stop_camera, emotion_mode, current_octave, gesture_state, update_emotion_enabled, camera_ready, current_emotion
    loading_screen("Loading camera...")
    cam_thread = threading.Thread(target=emotion_thread, daemon=True)
    cam_thread.start()
    arduino = threading.Thread(target=arduino_thread, daemon=True)
    arduino.start()
    while not camera_ready and not stop_camera:
        loading_screen("Loading camera...")
        time.sleep(0.1)
    running = True
    last_octave_change = 0
    mode = "hold"
    try:
        while running:
            current_time = time.time()
            if emotion_mode == "active":
                update_emotion_enabled = True
            if gesture_state == "thumbs_up" and current_time - last_octave_change > 1:
                if current_octave < 1:
                    current_octave += 1
                    last_octave_change = current_time
                gesture_state = None

            elif gesture_state == "thumbs_down" and current_time - last_octave_change > 1:
                if current_octave > -1:
                    current_octave -= 1
                    last_octave_change = current_time
                gesture_state = None
            draw_piano()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    stop_camera = True
                    break

                elif event.type == pygame.KEYDOWN:
                    if event.key in KEY_BINDINGS:
                        note = KEY_BINDINGS[event.key]
                        if note not in pressed_keys:
                            pressed_keys.add(note)
                            play_note(note)

                    # elif event.key == pygame.K_5:
                    #     update_emotion_enabled = True 
                    # EMOTION SHORTCUTS FOR DEBUGGING
                    elif event.key == pygame.K_KP1:
                        current_emotion = "angry"
                    elif event.key == pygame.K_KP2:
                        current_emotion = "disgusted"
                    elif event.key == pygame.K_KP3:
                        current_emotion = "fear"
                    elif event.key == pygame.K_KP4:
                        current_emotion = "happy"
                    elif event.key == pygame.K_KP5:
                        current_emotion = "neutral"
                    elif event.key == pygame.K_KP6:
                        current_emotion = "surprise"
                    elif event.key == pygame.K_KP7:
                        current_emotion = "sad"
                    # OCTAVE SHORTCUTS FOR DEBUGGING
                    elif event.key == pygame.K_KP8:
                        if current_octave < 1:
                            current_octave += 1
                    elif event.key == pygame.K_KP9:
                        if current_octave > -1:
                            current_octave -= 1
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if emotion_button_rect.collidepoint(event.pos):
                            if emotion_mode == "hold":
                                emotion_mode = "active"
                                update_emotion_enabled = True
                            else:
                                emotion_mode = "hold"
                                update_emotion_enabled = False
                        else:
                            if emotion_mode == "hold":
                                update_emotion_enabled = True

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if emotion_mode == "hold":
                            if not emotion_button_rect.collidepoint(event.pos):
                                update_emotion_enabled = False
                elif event.type == pygame.KEYUP:
                    if event.key in KEY_BINDINGS:
                        note = KEY_BINDINGS[event.key]
                        pressed_keys.discard(note)
    finally:
        stop_camera = True
        pygame.quit()
        try:
            cam_thread.join(timeout=1)
        except:
            pass
        cv2.destroyAllWindows()
        raise SystemExit 

if __name__ == "__main__":
    loading_screen("Loading sounds...")
    preload_all_sounds()
    loading_screen("Loading camera...")
    main()
