import pygame # used for the gui
import numpy as np # used for changing audio
import threading # allows us to do emotion detection and piano at same time
import cv2 # computer vision to detect camera input
import wave # reads .wav sound files
import io
from scipy.signal import resample # used for pitch shifting
from deepface import DeepFace # used to get emotion from camera data
import keyboard
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)


current_emotion = "neutral" # the default is neutral so it doesn't start as happy or sad
stop_camera = False # stop camera will be true when program is turned off
current_emotion = "neutral"
update_emotion_enabled = False

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

def emotion_thread():
    global current_emotion, stop_camera, update_emotion_enabled
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    current_emotion = "neutral"

    while not stop_camera:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if update_emotion_enabled:
            for (x, y, w, h) in faces:
                face_roi = rgb[y:y + h, x:x + w]
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotions = result[0]['dominant_emotion']

                current_emotion = emotions
                break

        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera = True
            break

    cap.release()
    cv2.destroyAllWindows()

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
    ratio = 2 ** (semitones / 12.0)
    
    new_length = int(len(base_audio) / ratio)
    resampled = resample(base_audio, new_length)

    fade_len = min(500, len(resampled))
    fade_out = np.linspace(1, 0, fade_len)
    if channels > 1:
        resampled[-fade_len:, :] *= fade_out[:, None]
    else:
        resampled[-fade_len:] *= fade_out

    resampled /= max(1e-9, np.max(np.abs(resampled)))
    resampled = (resampled * 32767).astype(np.int16)

    virtual_wav = io.BytesIO()
    with wave.open(virtual_wav, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(base_rate)
        wf.writeframes(resampled.tobytes())
    virtual_wav.seek(0)
    sound = pygame.mixer.Sound(virtual_wav)

    # # For reference/debug printing
    # freq_hz = base_freq * ratio
    # note_frequencies[note] = freq_hz
    # print(f"Cached {note}: {freq_hz:.2f} Hz (ratio={ratio:.4f})")

    return sound

def load_sample_for_emotion():
    base_file = "happy.wav" if current_emotion.upper() == "HAPPY" else "new.wav"
    base_audio, base_rate, channels = load_wav(base_file)
    base_freq = detect_base_frequency(base_audio, base_rate)
    print(f"[{current_emotion.upper()}] Base Frequency: {base_freq:.2f} Hz")

    sounds = {}
    for n in NOTE_MAP.keys():
        sounds[n] = generate_note(n, base_audio, base_rate, channels, base_freq)
    return sounds


def play_note(note):
    global cached_sounds
    if current_emotion.upper() not in cached_sounds:
        cached_sounds[current_emotion.upper()] = load_sample_for_emotion()

    sounds = cached_sounds[current_emotion.upper()]
    sound = sounds[note]
    channel = pygame.mixer.find_channel(True)
    if channel:
        channel.play(sound)

threading.Thread(target=emotion_thread, daemon=True).start()

WIDTH, HEIGHT = 1920, 1020
screen = pygame.display.set_mode((WIDTH, HEIGHT-300))
pygame.display.set_caption("Empathy Piano Digital")

font = pygame.font.SysFont("Heebo", 32)
big_font = pygame.font.SysFont("Heebo", 100)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GREEN = (144, 238, 144)
ANGRY = (135, 14, 14)
DISGUSTED = (49, 135, 59)
FEARFUL = (147, 104, 166)
HAPPY = (252, 192, 40)
NEUTRAL = (65, 65, 65)
SAD = (0, 0, 139)
SURPRISED = (171, 245, 244)

EMOTION_COLORS = {
    "ANGRY": ANGRY,
    "DISGUSTED": DISGUSTED,
    "FEARFUL": FEARFUL,
    "HAPPY": HAPPY,
    "NEUTRAL": NEUTRAL,
    "SAD": SAD,
    "SURPRISED": SURPRISED
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


def draw_piano():
    color = EMOTION_COLORS.get(current_emotion.upper(), NEUTRAL)
    screen.fill((color))
    title_label = big_font.render("THE " + current_emotion.upper() + " PIANO", True, WHITE)

    screen.blit(title_label, (WIDTH // 2 - title_label.get_width() // 2, 30))

    for i, note in enumerate(white_keys):
        rect = pygame.Rect(piano_x + i * key_width, piano_y, key_width, key_height)
        color = LIGHT_GREEN if note in pressed_keys else WHITE
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        white_rects[note] = rect
        label = font.render(KEY_LABELS[note], True, BLACK)
        screen.blit(label, (rect.centerx - label.get_width() / 2, rect.bottom - 60))

    for i, note in enumerate(black_keys):
        if note != '':
            rect = pygame.Rect(piano_x + i * key_width + key_width * 0.7, piano_y, black_key_width, black_key_height)
            color = LIGHT_GREEN if note in pressed_keys else BLACK
            pygame.draw.rect(screen, color, rect)
            black_rects[note] = rect
            label = font.render(KEY_LABELS[note], True, WHITE)
            screen.blit(label, (rect.centerx - label.get_width() / 2, rect.bottom - 60))

running = True
camera_thread = None
while running:
    draw_piano()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_BINDINGS:
                note = KEY_BINDINGS[event.key]
                if note not in pressed_keys:
                    pressed_keys.add(note)
                    play_note(note)

            elif event.key == pygame.K_5:
                update_emotion_enabled = True 

        elif event.type == pygame.KEYUP:
            if event.key in KEY_BINDINGS:
                note = KEY_BINDINGS[event.key]
                pressed_keys.discard(note)

            elif event.key == pygame.K_5:
                update_emotion_enabled = False

pygame.quit()
stop_camera = True