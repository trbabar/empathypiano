import pygame
import pyaudio
import wave
import numpy as np
from scipy.signal import resample
import threading
import time
import sounddevice as sd
import shutil

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
BASE_NOTE_FILE = "C.wav"
DEFAULT_NOTE_FILE = "dog.wav"

recording = []
is_recording = False
start_time = None
record_thread = None


NOTE_MAP = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4,
    'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

KEY_BINDINGS = {
    pygame.K_a: 'C', pygame.K_w: 'C#', pygame.K_s: 'D',
    pygame.K_e: 'D#', pygame.K_d: 'E', pygame.K_f: 'F',
    pygame.K_t: 'F#', pygame.K_g: 'G', pygame.K_y: 'G#',
    pygame.K_h: 'A', pygame.K_u: 'A#', pygame.K_j: 'B'
}

KEY_LABELS = {
    'C': 'A', 'C#': 'W', 'D': 'S', 'D#': 'E', 'E': 'D',
    'F': 'F', 'F#': 'T', 'G': 'G', 'G#': 'Y',
    'A': 'H', 'A#': 'U', 'B': 'J'
}

def trim_silence(audio, threshold=500):
    abs_audio = np.abs(audio)
    non_silent = np.where(abs_audio > threshold)[0]
    if len(non_silent) == 0:
        return audio
    start = non_silent[0]
    end = non_silent[-1]
    return audio[start:end+1]

def start_recording():
    global is_recording, recording, start_time, record_thread
    is_recording = True
    recording = []
    start_time = time.time()

    def record():
        global recording
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        while is_recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            recording.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()

        if recording:
            wf = wave.open(BASE_NOTE_FILE, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(recording))
            wf.close()
            print("Saved:", BASE_NOTE_FILE)
        else:
            print("No audio captured.")

    record_thread = threading.Thread(target=record)
    record_thread.start()

def stop_recording():
    global is_recording, start_time, record_thread
    is_recording = False
    start_time = None
    if record_thread:
        record_thread.join()  
        record_thread = None


def load_base_note():
    wf = wave.open(BASE_NOTE_FILE, 'rb')
    data = wf.readframes(wf.getnframes())
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    wf.close()
    return trim_silence(audio)

def generate_note(note):
    base_audio = load_base_note()
    semitones = NOTE_MAP[note]
    factor = 2 ** (semitones / 12.0)

    new_length = int(len(base_audio) / factor)
    resampled = resample(base_audio, new_length)

    resampled = resampled / np.max(np.abs(resampled))
    return resampled.astype(np.float32)

active_notes = []
lock = threading.Lock()

def audio_callback(outdata, frames, time_info, status):
    with lock:
        if not active_notes:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)
            return

        mixed = np.zeros(frames, dtype=np.float32)
        new_active = []
        for buffer, idx in active_notes:
            chunk = buffer[idx:idx+frames]
            if len(chunk) < frames:
                continue
            else:
                idx += frames
                new_active.append((buffer, idx))
            mixed[:len(chunk)] += chunk

        active_notes[:] = new_active
        mixed = mixed / max(1, np.max(np.abs(mixed)))
        outdata[:] = mixed.reshape(-1, 1)

stream = sd.OutputStream(channels=1, samplerate=RATE, callback=audio_callback)
stream.start()

def play_note(note):
    with lock:
        buffer = generate_note(note)
        active_notes.append((buffer, 0))    

pygame.init()
WIDTH, HEIGHT = 1920, 1020 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Piano Prototype")

font = pygame.font.SysFont("Heebo", 32)
big_font = pygame.font.SysFont("Heebo", 100)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

piano_width = WIDTH * 0.7
piano_height = HEIGHT//2.05
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

record_center = (WIDTH//2, HEIGHT-120)
record_radius = 100

def draw_piano():
    screen.fill((65, 65, 65))
    
    title_label = big_font.render("THE DOG PIANO", True, WHITE)
    screen.blit(title_label, (WIDTH//2 - title_label.get_width()//2, 30))

    for i, note in enumerate(white_keys):
        rect = pygame.Rect(piano_x + i*key_width, piano_y, key_width, key_height)
        pygame.draw.rect(screen, WHITE, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        white_rects[note] = rect
        label = font.render(KEY_LABELS[note], True, BLACK)
        screen.blit(label, (rect.centerx - label.get_width()/2, rect.bottom - 60))

    for i, note in enumerate(black_keys):
        if note != '':
            rect = pygame.Rect(piano_x + i*key_width + key_width*0.7, piano_y, black_key_width, black_key_height)
            pygame.draw.rect(screen, BLACK, rect)
            black_rects[note] = rect
            label = font.render(KEY_LABELS[note], True, WHITE)
            screen.blit(label, (rect.centerx - label.get_width()/2, rect.bottom - 60))

    if is_recording:
        pygame.draw.circle(screen, BLUE, record_center, record_radius)
    else:
        pygame.draw.circle(screen, RED, record_center, record_radius)
    text = "RECORDING" if is_recording else "RECORD"
    label = font.render(text, True, WHITE)
    screen.blit(label, (record_center[0] - label.get_width()//2, record_center[1] - label.get_height()//2))

    if is_recording and start_time:
        elapsed = int(time.time() - start_time)
        timer_label = font.render(f"{elapsed}s", True, BLACK)
        screen.blit(timer_label, (record_center[0] + 150, record_center[1]-20))

def get_note_from_mouse(pos):
    for note, rect in black_rects.items():
        if rect.collidepoint(pos):
            return note
    for note, rect in white_rects.items():
        if rect.collidepoint(pos):
            return note
    return None


running = True

while running:
    draw_piano()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            if (x - record_center[0])**2 + (y - record_center[1])**2 <= record_radius**2:
                if not is_recording:
                    start_recording()
                else:
                    stop_recording()
            else:
                note = get_note_from_mouse(event.pos)
                if note:
                    play_note(note)

        elif event.type == pygame.KEYDOWN:
            if event.key in KEY_BINDINGS:
                play_note(KEY_BINDINGS[event.key])

pygame.quit()
stream.stop()
stream.close()

try:
    shutil.copy(DEFAULT_NOTE_FILE, BASE_NOTE_FILE)
except Exception as e:
    print("Error resetting sound:", e)
