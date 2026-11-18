import math
import os
import random
import threading
import time

import numpy as np
import pygame
try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from pedalboard import Compressor, Pedalboard, Reverb
except ImportError:
    Compressor = Pedalboard = Reverb = None


WIDTH, HEIGHT = 1280, 720
FPS = 60
SAMPLE_RATE = 44100
IDLE_RPM = 550.0
CAR_RPM_BLEND = 0.40
TRAIN_SPEED_BOOST = 1.02
LOCOMOTIVES = (
    {"name": "VALLEY 4-6-0", "kind": "STEAM PASSENGER", "propulsion": "steam",
     "mass_tonnes": 290, "power_kw": 1900, "tractive_effort_kn": 180,
     "brake_decel": 0.72, "drag_area": 8.0, "adhesion": 0.04,
     "max_speed": 105.0, "cars": 4, "color": (54, 100, 79)},
    {"name": "METRO 240", "kind": "PASSENGER", "propulsion": "diesel",
     "mass_tonnes": 380,
     "power_kw": 2800, "tractive_effort_kn": 220, "brake_decel": 0.82,
     "drag_area": 9.0, "adhesion": 0.045, "max_speed": 125.0, "cars": 4,
     "color": (44, 146, 160)},
    {"name": "FREIGHT 880", "kind": "HEAVY FREIGHT", "propulsion": "diesel",
     "mass_tonnes": 1800,
     "power_kw": 4000, "tractive_effort_kn": 480, "brake_decel": 0.58,
     "drag_area": 19.0, "adhesion": 0.018, "max_speed": 92.0, "cars": 7,
     "color": (176, 112, 47)},
    {"name": "EXPRESS 500", "kind": "HIGH-SPEED", "propulsion": "diesel",
     "mass_tonnes": 520,
     "power_kw": 6000, "tractive_effort_kn": 320, "brake_decel": 1.02,
     "drag_area": 12.0, "adhesion": 0.04, "max_speed": 190.0, "cars": 5,
     "color": (177, 59, 67)},
)
FREIGHT_CARGOES = (
    {"name": "SAND", "tonnes": 450, "loading_units": 15},
    {"name": "TIMBER", "tonnes": 320, "loading_units": 13},
    {"name": "IRON ORE", "tonnes": 700, "loading_units": 18},
    {"name": "GRAIN", "tonnes": 260, "loading_units": 11},
)
LOADING_POINTS = ("QUARRY", "SAWMILL", "IRON MINE", "GRAIN SILO")
FREIGHT_DESTINATIONS = ("HARBOR TERMINAL", "NORTH DEPOT", "EAST YARD", "MILL TERMINAL")
ROUTE_OPTIONS = (("SHORT", 1400.0), ("REGIONAL", 2300.0), ("LONG HAUL", 3300.0))

BG = (11, 19, 28)
PANEL = (17, 29, 40)
WHITE = (235, 242, 247)
MUTED = (137, 158, 171)
CYAN = (69, 220, 229)
ORANGE = (255, 169, 79)
GREEN = (111, 220, 151)
RED = (255, 81, 80)


def clamp(value, low, high):
    return max(low, min(high, value))


class EngineAudio:
    """Procedural locomotive, track, brake, steam, and horn audio."""

    def __init__(self):
        self.parameters = (IDLE_RPM, 0.0, 0.0, 0.72, 0.0, 0.0, 0.0, 0.0)
        self.control_state = np.asarray(self.parameters, dtype=np.float64)
        self.phase = 0.0
        self.horn_phase = 0.0
        self.rail_phase = 0.0
        self.joint_phase = 0.0
        self.chuff_phase = 0.0
        self.horn_envelope = 0.0
        self.noise_tail = 0.0
        self.rail_noise_tail = 0.0
        self.effects = None
        if Pedalboard is not None:
            self.effects = Pedalboard([
                Compressor(threshold_db=-18.0, ratio=2.2),
                Reverb(room_size=0.42, damping=0.58, wet_level=0.07,
                       dry_level=0.93, width=0.85),
            ])
        self.stream = None
        self.audio_channel = None
        self.audio_thread = None
        self.audio_stop = threading.Event()
        self.error = ""
        self.warning = ""
        self.lock = threading.Lock()

    def set_controls(self, rpm, throttle, load, volume, horn=0.0,
                     speed=0.0, steam=0.0, brake=0.0):
        with self.lock:
            self.parameters = (rpm, throttle, load, volume, horn, speed, steam, brake)

    def render(self, frames):
        with self.lock:
            target_controls = np.asarray(self.parameters, dtype=np.float64)
        blend = 1.0 - np.exp(
            -np.arange(1, frames + 1, dtype=np.float64) / (SAMPLE_RATE * 0.04)
        )
        controls = self.control_state[:, None] + (
            target_controls - self.control_state
        )[:, None] * blend
        self.control_state = controls[:, -1].copy()
        rpm, throttle, load, volume, horn, speed, steam, brake = controls

        engine_steps = 2.0 * math.pi * np.maximum(10.0, rpm / 7.5) / SAMPLE_RATE
        engine_offsets = np.concatenate(([0.0], np.cumsum(engine_steps[:-1])))
        phase = self.phase + engine_offsets
        self.phase = float((self.phase + np.sum(engine_steps)) % (2 * math.pi))

        wave = 0.48 * np.sin(phase * 0.5)
        wave += 0.17 * np.sin(phase * 0.25 + 0.3)
        wave += 0.22 * np.sin(phase + 0.035)
        wave += 0.075 * np.sin(phase * 2.0 + 0.08)
        wave += 0.028 * np.sin(phase * 3.0 + 0.13)
        combustion = np.maximum(0.0, np.sin(phase * 0.5 + 0.4)) ** 3
        wave += combustion * (0.018 + load * 0.07)
        wave += np.sin(phase * 1.5 + 0.2) * throttle * 0.018

        noise = self._smooth_noise(frames, "noise_tail")
        wave += noise * (
            0.004 + throttle * 0.012 + load * 0.009 + brake * 0.015
            + steam * 0.022
        )
        speed_ms = np.maximum(0.0, speed)
        motion = np.clip(speed_ms / 9.0, 0.0, 1.0)
        rail_steps = 2.0 * math.pi * (18.0 + speed_ms * 2.7) / SAMPLE_RATE
        rail_offsets = np.concatenate(([0.0], np.cumsum(rail_steps[:-1])))
        rail_phase = self.rail_phase + rail_offsets
        self.rail_phase = float((self.rail_phase + np.sum(rail_steps)) % (2 * math.pi))
        rail_noise = self._smooth_noise(frames, "rail_noise_tail")
        rolling = (
            np.sin(rail_phase) + 0.32 * np.sin(rail_phase * 1.73)
            + rail_noise * 0.8
        )
        wave += rolling * motion * (0.045 + load * 0.035)

        joint_steps = 2.0 * math.pi * np.maximum(0.15, speed_ms / 18.0) / SAMPLE_RATE
        joint_offsets = np.concatenate(([0.0], np.cumsum(joint_steps[:-1])))
        joint_phase = self.joint_phase + joint_offsets
        self.joint_phase = float((self.joint_phase + np.sum(joint_steps)) % (2 * math.pi))
        clack = np.maximum(0.0, np.sin(joint_phase)) ** 20
        wave += clack * motion * 0.07
        brake_squeal = np.sin(rail_phase * 14.0) * (
            0.5 + 0.5 * np.sin(joint_phase)
        )
        wave += brake_squeal * motion * brake * 0.008

        chuff_steps = 2.0 * math.pi * np.maximum(0.7, speed_ms / 2.2) / SAMPLE_RATE
        chuff_offsets = np.concatenate(([0.0], np.cumsum(chuff_steps[:-1])))
        chuff_phase = self.chuff_phase + chuff_offsets
        self.chuff_phase = float((self.chuff_phase + np.sum(chuff_steps)) % (2 * math.pi))
        chuff = np.maximum(0.0, np.sin(chuff_phase)) ** 5
        wave += chuff * steam * 0.13

        horn_step = 2.0 * math.pi * 392.0 / SAMPLE_RATE
        horn_phase = self.horn_phase + horn_step * np.arange(frames)
        self.horn_phase = float((horn_phase[-1] + horn_step) % (2 * math.pi))
        horn_target = horn
        horn_tau = 0.035 if horn_target[-1] > self.horn_envelope else 0.12
        horn_times = np.arange(1, frames + 1) / SAMPLE_RATE
        horn_envelope = horn_target + (self.horn_envelope - horn_target) * np.exp(
            -horn_times / horn_tau
        )
        self.horn_envelope = float(horn_envelope[-1])
        horn_wave = (
            0.72 * np.sin(horn_phase) + 0.82 * np.sin(horn_phase * 1.19)
            + 0.34 * np.sin(horn_phase * 1.5) + 0.22 * np.sin(horn_phase * 2.0)
        ) * horn_envelope * 0.24
        wave = (wave * (0.10 + 0.16 * load) * volume + horn_wave).astype(np.float32)
        stereo_width = motion * volume * 0.032
        stereo = np.vstack((wave + rolling * stereo_width,
                            wave - rolling * stereo_width))
        if self.effects is not None:
            stereo = self.effects(stereo, SAMPLE_RATE, buffer_size=512, reset=False)
        return np.clip(stereo.T, -0.98, 0.98)

    def _smooth_noise(self, frames, state_name):
        point_count = max(2, frames // 96 + 2)
        points = np.random.standard_normal(point_count)
        points[0] = getattr(self, state_name)
        values = np.interp(
            np.linspace(0.0, point_count - 1, frames),
            np.arange(point_count),
            points,
        )
        setattr(self, state_name, float(values[-1]))
        return values

    def _callback(self, outdata, frames, _time_info, status):
        del _time_info
        if status:
            self.warning = str(status)
        outdata[:] = self.render(frames)

    def start(self):
        if sd is None:
            try:
                mixer_settings = pygame.mixer.get_init()
                if mixer_settings != (SAMPLE_RATE, -16, 2):
                    pygame.mixer.quit()
                    pygame.mixer.init(
                        frequency=SAMPLE_RATE, size=-16, channels=2, buffer=1024
                    )
                self.audio_channel = pygame.mixer.Channel(0)
                self.audio_stop.clear()
                self.audio_thread = threading.Thread(
                    target=self._mixer_loop, daemon=True
                )
                self.audio_thread.start()
            except pygame.error as exc:
                self.error = str(exc)
            return
        try:
            self.stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype="float32",
                blocksize=512, callback=self._callback,
            )
            self.stream.start()
        except Exception as exc:
            self.error = str(exc)
            self.stream = None

    def _mixer_loop(self):
        while not self.audio_stop.is_set():
            if self.audio_channel.get_queue() is None:
                samples = np.clip(self.render(4096) * 32767, -32768, 32767)
                sound = pygame.mixer.Sound(
                    buffer=samples.astype(np.int16).tobytes()
                )
                if self.audio_channel.get_busy():
                    self.audio_channel.queue(sound)
                else:
                    self.audio_channel.play(sound)
            else:
                time.sleep(0.01)

    def close(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
        if self.audio_thread is not None:
            self.audio_stop.set()
            self.audio_channel.stop()
            self.audio_thread.join(timeout=1.0)
            self.audio_thread = None


class TrainGame:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("RAILWAY 2D  |  TRAIN OPERATOR")
        if os.environ.get("ANDROID_ARGUMENT"):
            self.screen = pygame.display.set_mode(
                (WIDTH, HEIGHT), pygame.SCALED | pygame.FULLSCREEN
            )
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font_cache = {}
        self.audio = EngineAudio()
        self.audio.start()
        self.audio_volume = 0.72
        self.locomotive_index = 0
        self.train_selection_index = 0
        self.upgrades = {"traction": 0, "brakes": 0, "efficiency": 0}
        self.credits = 900
        self.mission_type = None
        self.mission_stage = "main_menu"
        self.mission_result = ""
        self.paused = False
        self.running = True
        self.pointer_actions = {}
        self.touch_controls = set()
        self.elapsed = 0.0
        self.distance = 0.0
        self.target_distance = 1250.0
        self.destination = "CENTRAL"
        self.cargo_type = "sand"
        self.cargo_tonnes = 0
        self.contract_reward = 550
        self.cargo_index = 0
        self.loading_point_index = 0
        self.route_index = 1
        self.load_progress = 0
        self.unload_progress = 0
        self.load_target_units = 1
        self.loading_elapsed = 0.0
        self.unloading_elapsed = 0.0
        self.loading_bonus = 0
        self.reset_train()

    @property
    def locomotive(self):
        return LOCOMOTIVES[self.locomotive_index]

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.font_cache:
            self.font_cache[key] = pygame.font.SysFont("segoeui", size, bold=bold)
        return self.font_cache[key]

    def text(self, value, x, y, size=18, color=WHITE, bold=False, anchor="topleft"):
        image = self.font(size, bold).render(str(value), True, color)
        rect = image.get_rect()
        setattr(rect, anchor, (int(x), int(y)))
        self.screen.blit(image, rect)
        return rect

    def reset_train(self):
        self.speed = 0.0
        self.rpm = IDLE_RPM
        self.throttle = 0.0
        self.brake = 0.0
        self.horn = 0.0
        self.track_offset = 0.0
        self.station_stopped = False
        self.sand_level = 100.0
        self.water_level = 100.0
        self.engine_temp = 72.0
        self.steam_pressure = 1.0
        self.steam_output = 0.0
        self.air_pressure = 5.0
        self.sanding = False
        self.cooling = False
        self.wheel_slip = False
        self.slip_intensity = 0.0

    def start_mission(self, mission_type):
        self.mission_type = mission_type
        self.mission_result = ""
        self.distance = 0.0
        self.elapsed = 0.0
        self.loading_bonus = 0
        if mission_type == "passenger":
            self.mission_stage = "running"
            self.destination, self.target_distance = random.choice((
                ("CENTRAL", 1100.0), ("NORTH JUNCTION", 1450.0),
                ("RIVER STATION", 1750.0),
            ))
            self.cargo_tonnes = 0
            self.contract_reward = random.randint(500, 750)
        else:
            cargo = FREIGHT_CARGOES[self.cargo_index]
            self.cargo_type = cargo["name"]
            self.cargo_tonnes = cargo["tonnes"]
            self.load_target_units = cargo["loading_units"]
            self.destination = FREIGHT_DESTINATIONS[self.loading_point_index]
            route_name, self.target_distance = ROUTE_OPTIONS[self.route_index]
            self.contract_reward = int(450 + self.cargo_tonnes * 0.35
                                       + self.target_distance * 0.15)
            self.route_name = route_name
            average_speed = max(10.0, self.locomotive["max_speed"] / 3.6 * 0.55)
            self.route_deadline = 60.0 + self.target_distance / average_speed
            self.load_progress = 0
            self.loading_elapsed = 0.0
            self.mission_stage = "loading"
        self.reset_train()
        self.paused = False

    def open_freight_setup(self):
        self.mission_type = "freight"
        self.mission_stage = "freight_setup"
        self.mission_result = ""
        self.load_progress = 0
        cargo = FREIGHT_CARGOES[self.cargo_index]
        self.cargo_type = cargo["name"]
        self.cargo_tonnes = cargo["tonnes"]
        self.load_target_units = cargo["loading_units"]

    def finish_loading(self):
        if self.load_progress < self.load_target_units:
            return
        self.loading_bonus = max(
            0, int((self.load_target_units - self.loading_elapsed) * 35.0)
        )
        self.contract_reward += self.loading_bonus
        self.distance = 0.0
        self.elapsed = 0.0
        self.mission_stage = "running"
        self.reset_train()

    def finish_unloading(self):
        if self.unload_progress < self.load_target_units:
            return
        self.complete_mission(
            self.contract_reward,
            f"{self.cargo_type} unloaded at {self.destination}",
        )

    def load_one_unit(self):
        if self.mission_stage != "loading":
            return
        self.load_progress = min(self.load_target_units, self.load_progress + 1)
        if self.load_progress >= self.load_target_units:
            self.finish_loading()

    def unload_one_unit(self):
        if self.mission_stage != "unloading":
            return
        self.unload_progress = min(self.load_target_units, self.unload_progress + 1)
        if self.unload_progress >= self.load_target_units:
            self.finish_unloading()

    def mission_target(self):
        return self.target_distance

    def track_grade(self):
        return 0.006 * math.sin(self.distance / 850.0)

    def update(self, dt):
        if self.paused or self.mission_result:
            self.audio.set_controls(IDLE_RPM, 0.0, 0.0, self.audio_volume, self.horn)
            return
        if self.mission_stage in ("loading", "unloading"):
            if self.mission_stage == "loading":
                self.loading_elapsed += dt
            else:
                self.unloading_elapsed += dt
            self.audio.set_controls(IDLE_RPM, 0.0, 0.0, self.audio_volume)
            return
        if self.mission_stage != "running":
            self.audio.set_controls(IDLE_RPM, 0.0, 0.0, self.audio_volume)
            return

        keys = pygame.key.get_pressed()
        accelerating = (
            keys[pygame.K_w] or keys[pygame.K_UP] or "power" in self.touch_controls
        )
        braking = (
            keys[pygame.K_s] or keys[pygame.K_DOWN] or "brake" in self.touch_controls
        )
        self.horn = 1.0 if keys[pygame.K_h] or "horn" in self.touch_controls else 0.0
        self.sanding = bool(
            (keys[pygame.K_x] or "sand" in self.touch_controls)
            and accelerating and self.sand_level > 0.0
        )
        self.cooling = bool(
            (keys[pygame.K_c] or "cool" in self.touch_controls)
            and self.water_level > 0.0
        )

        self.throttle = clamp(
            self.throttle + (0.8 if accelerating else -0.34) * dt, 0.0, 1.0
        )
        self.brake = clamp(self.brake + (1.45 if braking else -0.82) * dt, 0.0, 1.0)
        loco = self.locomotive
        upgrade_power = 1.0 + self.upgrades["traction"] * 0.12
        upgrade_brake = 1.0 + self.upgrades["brakes"] * 0.15
        speed_ms = self.speed / 3.6
        mass = (loco["mass_tonnes"] + self.cargo_tonnes) * 1000.0
        requested_traction = min(
            loco["tractive_effort_kn"] * 1000.0,
            loco["power_kw"] * 1000.0 / max(speed_ms, 2.0),
        ) * self.throttle * upgrade_power * TRAIN_SPEED_BOOST
        adhesion = loco["adhesion"] + (0.055 if self.sanding else 0.0)
        adhesion_limit = mass * 9.81 * adhesion
        self.slip_intensity = clamp(
            (requested_traction - adhesion_limit) / max(adhesion_limit, 1.0),
            0.0, 1.0,
        )
        self.wheel_slip = self.slip_intensity > 0.0
        traction_force = min(requested_traction, adhesion_limit)
        brake_force = mass * loco["brake_decel"] * upgrade_brake * self.brake
        rolling_force = mass * 9.81 * 0.0015 * (1.0 - self.upgrades["efficiency"] * 0.04)
        aerodynamic_force = 0.5 * 1.225 * loco["drag_area"] * speed_ms * speed_ms
        grade = self.track_grade()
        grade_force = mass * 9.81 * grade
        acceleration = (
            traction_force - brake_force - rolling_force - aerodynamic_force - grade_force
        ) / mass
        speed_ms = clamp(
            speed_ms + acceleration * dt,
            0.0,
            loco["max_speed"] * TRAIN_SPEED_BOOST / 3.6,
        )
        self.speed = speed_ms * 3.6
        self.distance += speed_ms * dt
        self.track_offset += speed_ms * dt * 3.0
        self.elapsed += dt
        load = clamp(self.throttle * 0.68 + speed_ms / 45.0 * 0.32, 0.0, 1.0)
        self.sand_level = max(0.0, self.sand_level - (5.0 * dt if self.sanding else 0.0))
        heat_gain = self.throttle * load
        heat_rate = 1.7 * heat_gain * (1.0 - self.upgrades["efficiency"] * 0.08) - 0.55
        self.engine_temp += heat_rate * dt
        if self.cooling:
            self.engine_temp -= 11.0 * dt
            self.water_level = max(0.0, self.water_level - 4.0 * dt)
        self.engine_temp = clamp(self.engine_temp, 20.0, 140.0)
        self.steam_pressure = clamp(
            1.0 + max(0.0, self.engine_temp - 60.0) * 0.075 + self.throttle * 2.0,
            0.0, 12.0,
        )
        self.steam_output = clamp(
            (self.steam_pressure / 12.0) * (0.25 + self.throttle * 0.75), 0.0, 1.0
        ) if loco["propulsion"] == "steam" else 0.0
        self.air_pressure = 5.0 + self.brake * 3.0

        rail_rpm = IDLE_RPM + self.throttle * (760.0 + load * 480.0)
        car_rpm = IDLE_RPM + self.throttle * (4800.0 + load * 1000.0)
        target_rpm = rail_rpm * (1.0 - CAR_RPM_BLEND) + car_rpm * CAR_RPM_BLEND
        response_rate = 3.8 if target_rpm > self.rpm else 2.1
        self.rpm += (target_rpm - self.rpm) * (1.0 - math.exp(-response_rate * dt))
        self.audio.set_controls(
            self.rpm, self.throttle, load, self.audio_volume, self.horn,
            speed_ms, self.steam_output, self.brake,
        )

        target = self.mission_target()
        if self.engine_temp >= 125.0:
            self.fail_mission("Engine overheated — use water cooling before the orange zone")
            return
        if self.mission_type == "passenger":
            at_platform = abs(self.distance - target) <= 45.0
            if at_platform and self.speed <= 5.0:
                self.station_stopped = True
                self.complete_mission(self.contract_reward, f"Passengers delivered to {self.destination}")
            elif self.distance > target + 45.0:
                self.fail_mission("You passed the passenger platform")
        else:
            if self.distance >= target:
                if self.speed <= 15.0:
                    self.mission_stage = "unloading"
                    self.unload_progress = 0
                    self.unloading_elapsed = 0.0
                else:
                    self.fail_mission("Slow down before entering the freight yard")
            elif self.elapsed > self.route_deadline:
                self.fail_mission("Freight delivery deadline missed")

    def complete_mission(self, reward, message):
        reward = int(reward * (1.0 + self.upgrades["efficiency"] * 0.04))
        self.credits += reward
        bonus = f" (loading bonus ¢{self.loading_bonus})" if self.mission_type == "freight" else ""
        self.mission_result = f"{message} — +¢{reward}{bonus}"
        self.mission_stage = "result"
        self.audio.set_controls(IDLE_RPM, 0.0, 0.0, self.audio_volume)

    def fail_mission(self, message):
        self.mission_result = message
        self.mission_stage = "result"
        self.audio.set_controls(IDLE_RPM, 0.0, 0.0, self.audio_volume)

    def draw_world(self, width, height):
        horizon = int(height * 0.48)
        rail_y = int(height * 0.65)
        pygame.draw.rect(self.screen, (120, 177, 201), (0, 0, width, horizon))
        sun_x = int(width * 0.82)
        pygame.draw.circle(self.screen, (255, 222, 164), (sun_x, 105), 34)
        pygame.draw.circle(self.screen, (255, 235, 190), (sun_x, 105), 25)
        cloud_phase = (self.track_offset * 0.05) % 223
        for cloud_index in range(-1, int(width / 223) + 2):
            cloud_x = int(cloud_index * 223 - cloud_phase + 70)
            cloud_y = 63 + (cloud_index * 41 % 120)
            pygame.draw.ellipse(self.screen, (214, 231, 231),
                                (cloud_x, cloud_y, 95, 22))
            pygame.draw.circle(self.screen, (222, 236, 235), (cloud_x + 23, cloud_y), 15)
            pygame.draw.circle(self.screen, (222, 236, 235), (cloud_x + 56, cloud_y - 5), 19)

        pygame.draw.rect(self.screen, (102, 144, 105), (0, horizon, width, height - horizon))
        far_hills = [(0, horizon + 12)]
        near_hills = [(0, horizon + 35)]
        for x in range(0, width + 80, 80):
            far_hills.append((x, horizon - 42 - (x * 37 % 53)))
            near_hills.append((x, horizon + 4 - (x * 23 % 45)))
        far_hills.extend([(width, horizon + 25), (width, height)])
        near_hills.extend([(width, horizon + 44), (width, height)])
        pygame.draw.polygon(self.screen, (89, 130, 134), far_hills)
        pygame.draw.polygon(self.screen, (91, 135, 101), near_hills)

        vanishing_x = int(width * 0.56)
        vanishing_y = horizon + 28
        near_y = rail_y - 42
        near_center = int(width * 0.73)
        near_half_width = int(width * 0.17)
        pygame.draw.polygon(self.screen, (87, 90, 82), [
            (vanishing_x - 4, vanishing_y), (vanishing_x + 4, vanishing_y),
            (near_center + near_half_width, near_y),
            (near_center - near_half_width, near_y),
        ])
        for side in (-1, 1):
            pygame.draw.line(
                self.screen, (134, 144, 143),
                (vanishing_x + side * 2, vanishing_y),
                (near_center + side * near_half_width, near_y), 4,
            )
            pygame.draw.line(
                self.screen, (214, 219, 213),
                (vanishing_x + side * 2, vanishing_y),
                (near_center + side * near_half_width, near_y), 1,
            )
        perspective_phase = (self.track_offset * 0.0015) % 1.0
        for sleeper in range(18):
            depth = (sleeper + perspective_phase) / 18.0
            if depth > 1:
                continue
            sleeper_y = int(vanishing_y + (near_y - vanishing_y) * depth)
            center_x = vanishing_x + (near_center - vanishing_x) * depth
            half_width = 3 + near_half_width * depth
            pygame.draw.line(
                self.screen, (93, 66, 47),
                (int(center_x - half_width), sleeper_y),
                (int(center_x + half_width), sleeper_y),
                max(1, int(1 + depth * 7)),
            )

        field_offset = (self.track_offset * 0.45) % 54
        for row in range(5):
            y = horizon + 73 + row * 22
            start = int(-(field_offset * (1 + row * 0.12)) % 54)
            for x in range(start, width, 54):
                pygame.draw.line(self.screen, (119, 150, 94),
                                 (x, y), (x + 29, y + 9), 2)

        for index in range(-1, int(width / 302) + 2):
            for layer, base_x, depth_scale in ((0, 0, 0.16), (1, 151, 0.42)):
                tree_phase = (self.track_offset * depth_scale) % 302
                x = int(index * 302 + base_x - tree_phase + 35)
                tree_index = index * 2 + layer
                base_y = horizon + 76 + (tree_index * 19 % 58)
                tree_height = 30 + tree_index * 11 % 31
                pygame.draw.rect(self.screen, (91, 70, 50),
                                 (x + 15, base_y - tree_height // 2, 8,
                                  tree_height // 2 + 5))
                if tree_index % 3 == 0:
                    crown = [(x + 2, base_y - 20), (x + 7, base_y - 48),
                             (x + 16, base_y - 65), (x + 27, base_y - 45),
                             (x + 34, base_y - 20)]
                    pygame.draw.polygon(self.screen, (48, 96, 65), crown)
                    pygame.draw.polygon(self.screen, (70, 117, 72), crown, 2)
                else:
                    pygame.draw.circle(self.screen, (47, 92, 63),
                                       (x + 18, base_y - tree_height), 20 + tree_index % 9)
                    pygame.draw.circle(self.screen, (65, 111, 70),
                                       (x + 2, base_y - tree_height + 12), 13)
                    pygame.draw.circle(self.screen, (78, 123, 75),
                                       (x + 33, base_y - tree_height + 9), 14)

        pole_phase = (self.track_offset * 0.65) % 290
        for index in range(-1, int(width / 290) + 2):
            pole_x = int(index * 290 - pole_phase + 125)
            pole_top = horizon + 22
            pole_bottom = int(height * 0.70)
            pygame.draw.line(self.screen, (103, 91, 70),
                             (pole_x, pole_top), (pole_x, pole_bottom), 4)
            pygame.draw.line(self.screen, (103, 91, 70),
                             (pole_x - 17, pole_top + 9), (pole_x + 17, pole_top + 9), 3)
            pygame.draw.circle(self.screen, (66, 75, 74), (pole_x - 12, pole_top + 10), 3)
            pygame.draw.circle(self.screen, (66, 75, 74), (pole_x + 12, pole_top + 10), 3)
            next_x = pole_x + 290
            pygame.draw.line(self.screen, (72, 81, 80),
                             (pole_x - 12, pole_top + 10), (next_x - 12, pole_top + 10), 1)
            pygame.draw.line(self.screen, (72, 81, 80),
                             (pole_x + 12, pole_top + 10), (next_x + 12, pole_top + 10), 1)

        pygame.draw.rect(self.screen, (95, 93, 82), (0, rail_y + 1, width, height - rail_y))
        for rock in range(100):
            rock_x = (rock * 79 + rock * rock % 31) % width
            rock_y = rail_y + 7 + (rock * 17 % 42)
            pygame.draw.circle(self.screen, (124, 120, 103),
                               (rock_x, rock_y), 1 + rock % 2)
        sleeper_phase = self.track_offset % 48
        for x in range(-48, width + 96, 48):
            sleeper_x = int(x - sleeper_phase)
            pygame.draw.rect(self.screen, (89, 64, 46),
                             (sleeper_x, rail_y + 9, 34, 36), border_radius=3)
            pygame.draw.rect(self.screen, (123, 91, 65),
                             (sleeper_x, rail_y + 9, 34, 4), border_radius=2)
        for rail_offset in (16, 22):
            pygame.draw.rect(self.screen, (128, 138, 139),
                             (0, rail_y + rail_offset, width, 5))
            pygame.draw.line(self.screen, (220, 226, 224),
                             (0, rail_y + rail_offset), (width, rail_y + rail_offset), 2)
            pygame.draw.line(self.screen, (76, 84, 86),
                             (0, rail_y + rail_offset + 4),
                             (width, rail_y + rail_offset + 4), 1)

        train_x = int(width * 0.22)
        target_x = train_x + 200 + int((self.mission_target() - self.distance) * 0.22)
        if self.mission_stage == "loading":
            self.draw_station(train_x + 200, rail_y, False)
        elif self.mission_stage in ("running", "unloading") and -180 < target_x < width + 180:
            self.draw_station(target_x, rail_y, self.mission_type == "passenger")
        self.draw_train(train_x, rail_y)

    def draw_station(self, x, rail_y, passenger):
        color = CYAN if passenger else ORANGE
        platform = pygame.Rect(x - 86, rail_y - 46, 172, 18)
        pygame.draw.rect(self.screen, (171, 177, 166), platform, border_radius=3)
        pygame.draw.rect(self.screen, color, (platform.x, platform.y, platform.width, 4))
        pygame.draw.rect(self.screen, (65, 71, 72), (x - 4, rail_y - 147, 8, 100))
        sign = pygame.Rect(x - 60, rail_y - 187, 120, 42)
        pygame.draw.rect(self.screen, (18, 35, 44), sign, border_radius=5)
        pygame.draw.rect(self.screen, color, sign, width=2, border_radius=5)
        is_loading = self.mission_stage == "loading"
        name = LOADING_POINTS[self.loading_point_index] if is_loading else self.destination
        self.text(name, sign.centerx, sign.centery - 5, 10, WHITE, True, "center")
        self.text("LOAD" if is_loading else ("PLATFORM" if passenger else "UNLOAD"),
                  sign.centerx, sign.centery + 10, 7, color, True, "center")
        if passenger:
            shelter = pygame.Rect(x - 48, rail_y - 127, 88, 41)
            pygame.draw.rect(self.screen, (74, 91, 91), shelter, border_radius=3)
            pygame.draw.rect(self.screen, (43, 62, 67),
                             (shelter.x + 7, shelter.y + 9, 74, 29))
            pygame.draw.polygon(self.screen, (46, 57, 57), [
                (shelter.x - 8, shelter.y + 5), (shelter.centerx, shelter.y - 12),
                (shelter.right + 8, shelter.y + 5),
            ])
            pygame.draw.line(self.screen, (196, 190, 166),
                             (shelter.x - 8, shelter.y + 5),
                             (shelter.right + 8, shelter.y + 5), 3)
            for offset in (-48, 0, 48):
                pygame.draw.circle(self.screen, (49, 76, 77), (x + offset, rail_y - 62), 8)
                pygame.draw.line(self.screen, (46, 54, 53),
                                 (x + offset, rail_y - 54),
                                 (x + offset - 2, rail_y - 39), 3)
            pygame.draw.rect(self.screen, (95, 73, 52), (x - 95, rail_y - 53, 39, 5))
            pygame.draw.rect(self.screen, (61, 69, 68), (x - 92, rail_y - 62, 4, 10))
            pygame.draw.rect(self.screen, (61, 69, 68), (x - 60, rail_y - 62, 4, 10))
        elif is_loading:
            self.draw_loading_site(x, rail_y)
        else:
            self.draw_unloading_site(x, rail_y)

    def draw_loading_site(self, x, rail_y):
        point = LOADING_POINTS[self.loading_point_index]
        if point == "QUARRY":
            pygame.draw.polygon(self.screen, (104, 100, 86), [
                (x - 165, rail_y - 38), (x - 139, rail_y - 83),
                (x - 108, rail_y - 52), (x - 86, rail_y - 38),
            ])
            pygame.draw.line(self.screen, (83, 75, 62),
                             (x - 184, rail_y - 125), (x - 83, rail_y - 69), 7)
            pygame.draw.line(self.screen, (167, 146, 104),
                             (x - 184, rail_y - 125), (x - 83, rail_y - 69), 2)
            pygame.draw.line(self.screen, (78, 71, 62),
                             (x - 174, rail_y - 116), (x - 174, rail_y - 44), 5)
            pygame.draw.line(self.screen, (78, 71, 62),
                             (x - 119, rail_y - 85), (x - 119, rail_y - 42), 5)
        elif point == "SAWMILL":
            shed = pygame.Rect(x - 175, rail_y - 115, 93, 69)
            pygame.draw.rect(self.screen, (106, 83, 59), shed)
            pygame.draw.polygon(self.screen, (67, 65, 57), [
                (shed.x - 10, shed.y + 5), (shed.centerx, shed.y - 22),
                (shed.right + 9, shed.y + 5),
            ])
            pygame.draw.rect(self.screen, (44, 49, 48),
                             (shed.x + 14, shed.y + 24, 58, 44))
            for pile in range(3):
                log_y = rail_y - 49 - pile * 11
                pygame.draw.line(self.screen, (142, 93, 52),
                                 (x - 79, log_y), (x - 24, log_y), 7)
                pygame.draw.circle(self.screen, (200, 150, 99), (x - 79, log_y), 4)
        elif point == "IRON MINE":
            pygame.draw.line(self.screen, (50, 62, 62),
                             (x - 149, rail_y - 42), (x - 149, rail_y - 156), 7)
            pygame.draw.line(self.screen, (50, 62, 62),
                             (x - 96, rail_y - 42), (x - 96, rail_y - 156), 7)
            pygame.draw.line(self.screen, (74, 85, 82),
                             (x - 161, rail_y - 152), (x - 84, rail_y - 152), 8)
            pygame.draw.line(self.screen, (64, 75, 73),
                             (x - 150, rail_y - 113), (x - 96, rail_y - 154), 5)
            pygame.draw.line(self.screen, (64, 75, 73),
                             (x - 96, rail_y - 113), (x - 150, rail_y - 154), 5)
            pygame.draw.rect(self.screen, (96, 91, 76),
                             (x - 172, rail_y - 40, 100, 12))
        else:
            for silo_x in (x - 167, x - 119):
                pygame.draw.rect(self.screen, (181, 178, 153),
                                 (silo_x, rail_y - 133, 37, 89), border_radius=12)
                pygame.draw.ellipse(self.screen, (210, 204, 172),
                                    (silo_x, rail_y - 143, 37, 20))
                pygame.draw.rect(self.screen, (113, 120, 108),
                                 (silo_x + 6, rail_y - 42, 4, 9))
            pygame.draw.line(self.screen, (112, 111, 95),
                             (x - 151, rail_y - 122), (x - 101, rail_y - 122), 4)
            pygame.draw.line(self.screen, (112, 111, 95),
                             (x - 101, rail_y - 122), (x - 101, rail_y - 52), 4)

    def draw_unloading_site(self, x, rail_y):
        warehouse = pygame.Rect(x + 76, rail_y - 116, 144, 70)
        pygame.draw.rect(self.screen, (94, 103, 99), warehouse)
        pygame.draw.polygon(self.screen, (62, 75, 74), [
            (warehouse.x - 8, warehouse.y + 8), (warehouse.centerx, warehouse.y - 25),
            (warehouse.right + 8, warehouse.y + 8),
        ])
        for bay in range(3):
            pygame.draw.rect(self.screen, (43, 54, 55),
                             (warehouse.x + 12 + bay * 42, warehouse.y + 30, 30, 40))
        pygame.draw.rect(self.screen, (42, 51, 53), (x + 42, rail_y - 179, 9, 138))
        pygame.draw.rect(self.screen, (42, 51, 53), (x + 246, rail_y - 179, 9, 138))
        pygame.draw.rect(self.screen, (54, 64, 63), (x + 42, rail_y - 181, 214, 9))
        pygame.draw.line(self.screen, (101, 111, 107),
                         (x + 162, rail_y - 172), (x + 162, rail_y - 119), 4)
        pygame.draw.rect(self.screen, (171, 152, 110),
                         (x + 151, rail_y - 125, 23, 14))
        if "HARBOR" in self.destination:
            pygame.draw.rect(self.screen, (55, 96, 111),
                             (x + 86, rail_y - 40, 155, 9))
            for stack in range(4):
                pygame.draw.rect(self.screen, (158, 112 + stack * 9, 68),
                                 (x + 91 + stack * 34, rail_y - 70, 28, 29))

    def draw_train(self, x, rail_y):
        loco = self.locomotive
        color = loco["color"]
        car_count = loco["cars"]
        car_width = 116
        gap = 8
        body_y = rail_y - 100
        shadow = pygame.Rect(x - car_count * (car_width + gap) - 14,
                             rail_y + 17, car_count * (car_width + gap) + 225, 22)
        pygame.draw.ellipse(self.screen, (61, 64, 56), shadow)
        for index in range(car_count, 0, -1):
            car_x = x - index * (car_width + gap)
            if loco["propulsion"] == "steam" and index == 1:
                body = pygame.Rect(car_x, body_y + 27, car_width, 47)
                pygame.draw.rect(self.screen, (44, 49, 47), body, border_radius=4)
                pygame.draw.rect(self.screen, (137, 133, 111), body, width=2, border_radius=4)
                pygame.draw.polygon(self.screen, (40, 43, 40), [
                    (car_x + 9, body_y + 28), (car_x + 25, body_y + 13),
                    (car_x + 86, body_y + 13), (car_x + 105, body_y + 28),
                ])
                for coal in range(5):
                    pygame.draw.circle(self.screen, (35, 38, 36),
                                       (car_x + 25 + coal * 16, body_y + 24), 8)
            elif loco["kind"] == "HEAVY FREIGHT":
                body = [
                    (car_x + 7, body_y + 20), (car_x + car_width - 7, body_y + 20),
                    (car_x + car_width - 20, body_y + 76), (car_x + 20, body_y + 76),
                ]
                pygame.draw.polygon(self.screen, (74, 83, 82), body)
                pygame.draw.lines(self.screen, (167, 130, 82), True, body, 3)
                pygame.draw.line(self.screen, (44, 54, 57),
                                 (car_x + 15, body_y + 35),
                                 (car_x + car_width - 15, body_y + 35), 2)
                for rib in range(4):
                    rib_x = car_x + 19 + rib * 25
                    pygame.draw.line(self.screen, (124, 112, 91),
                                     (rib_x, body_y + 23), (rib_x, body_y + 69), 2)
                loading_scene = self.mission_stage in ("freight_setup", "loading")
                unloading_scene = self.mission_stage == "unloading"
                if self.cargo_type == "TIMBER":
                    loaded_logs = 3
                    if loading_scene:
                        loaded_logs = int(3 * self.load_progress / self.load_target_units)
                    elif unloading_scene:
                        loaded_logs = int(3 * (1 - self.unload_progress / self.load_target_units))
                    for log in range(loaded_logs):
                        log_y = body_y + 31 + log * 10
                        pygame.draw.line(self.screen, (142, 93, 52),
                                         (car_x + 18, log_y), (car_x + 98, log_y), 6)
                        pygame.draw.circle(self.screen, (195, 146, 96),
                                           (car_x + 18, log_y), 3)
                else:
                    cargo_fill = 1.0
                    if loading_scene:
                        cargo_fill = self.load_progress / self.load_target_units
                    elif unloading_scene:
                        cargo_fill = 1.0 - self.unload_progress / self.load_target_units
                    cargo_color = (194, 162, 108) if self.cargo_type == "SAND" else (
                        (122, 91, 67) if self.cargo_type == "IRON ORE" else (189, 168, 96)
                    )
                    if cargo_fill > 0:
                        peak_y = body_y + int(49 - 22 * cargo_fill)
                        pygame.draw.polygon(self.screen, cargo_color, [
                            (car_x + 20, body_y + 48), (car_x + 39, peak_y + 4),
                            (car_x + 57, body_y + 43), (car_x + 76, peak_y),
                            (car_x + 97, body_y + 48),
                        ])
            else:
                body = pygame.Rect(car_x, body_y + 13, car_width, 63)
                pygame.draw.rect(self.screen, (48, 65, 73), body, border_radius=3)
                pygame.draw.rect(self.screen, (151, 164, 166), body,
                                 width=2, border_radius=3)
                pygame.draw.polygon(self.screen, (101, 116, 117), [
                    (car_x, body_y + 15), (car_x + 12, body_y + 5),
                    (car_x + car_width - 4, body_y + 5),
                    (car_x + car_width, body_y + 15),
                ])
                pygame.draw.line(self.screen, color,
                                 (car_x + 3, body_y + 66),
                                 (car_x + car_width - 3, body_y + 66), 4)
                for window in range(4):
                    window_x = car_x + 9 + window * 25
                    pygame.draw.rect(self.screen, (89, 143, 157),
                                     (window_x, body_y + 23, 19, 21), border_radius=2)
                    pygame.draw.line(self.screen, (177, 208, 208),
                                     (window_x + 3, body_y + 25),
                                     (window_x + 14, body_y + 25), 1)
                pygame.draw.rect(self.screen, (191, 199, 193),
                                 (car_x + 96, body_y + 18, 12, 53), 2)
            pygame.draw.rect(self.screen, (39, 47, 50),
                             (car_x + 10, body_y + 76, car_width - 20, 9))
            self.draw_bogie(car_x + 18, rail_y + 6)
            self.draw_bogie(car_x + car_width - 53, rail_y + 6)
            pygame.draw.rect(self.screen, (31, 37, 39),
                             (car_x - gap, rail_y - 26, gap + 3, 5))

        if loco["propulsion"] == "steam":
            self.draw_steam_locomotive(x, body_y, rail_y, color)
            return

        # A long-hood diesel with a separate cab, radiator panels and running gear.
        engine = [
            (x + 3, body_y + 38), (x + 18, body_y + 24), (x + 108, body_y + 24),
            (x + 108, body_y + 5), (x + 174, body_y + 5), (x + 185, body_y + 24),
            (x + 192, body_y + 51), (x + 184, body_y + 83), (x + 4, body_y + 83),
        ]
        pygame.draw.polygon(self.screen, color, engine)
        pygame.draw.lines(self.screen, (222, 226, 218), True, engine, 2)
        pygame.draw.rect(self.screen, (31, 42, 48), (x + 106, body_y - 5, 76, 15),
                         border_radius=4)
        pygame.draw.polygon(self.screen, (56, 71, 78), [
            (x + 119, body_y + 4), (x + 171, body_y + 4),
            (x + 177, body_y + 21), (x + 113, body_y + 21),
        ])
        pygame.draw.polygon(self.screen, (98, 164, 177), [
            (x + 124, body_y + 7), (x + 166, body_y + 7),
            (x + 171, body_y + 18), (x + 121, body_y + 18),
        ])
        pygame.draw.line(self.screen, (37, 49, 53),
                         (x + 108, body_y + 5), (x + 108, body_y + 76), 3)
        pygame.draw.rect(self.screen, (47, 59, 62), (x + 26, body_y + 37, 68, 32))
        for vent in range(6):
            vent_x = x + 32 + vent * 9
            pygame.draw.line(self.screen, (181, 190, 181),
                             (vent_x, body_y + 40), (vent_x, body_y + 64), 2)
        pygame.draw.line(self.screen, (231, 183, 87),
                         (x + 12, body_y + 70), (x + 183, body_y + 70), 4)
        pygame.draw.line(self.screen, (211, 218, 210),
                         (x + 14, body_y + 75), (x + 179, body_y + 75), 1)
        for panel_x in (x + 20, x + 101):
            pygame.draw.rect(self.screen, (39, 51, 55),
                             (panel_x, body_y + 39, 4, 25), border_radius=1)
        pygame.draw.line(self.screen, (220, 196, 128),
                         (x + 20, body_y + 29), (x + 99, body_y + 29), 2)
        for rail_x in (x + 23, x + 96):
            pygame.draw.line(self.screen, (203, 194, 158),
                             (rail_x, body_y + 29), (rail_x, body_y + 34), 1)
        pygame.draw.rect(self.screen, (31, 43, 48), (x + 76, body_y + 34, 20, 36))
        pygame.draw.rect(self.screen, color, (x + 78, body_y + 36, 16, 32))
        pygame.draw.rect(self.screen, (31, 37, 39), (x + 7, body_y + 76, 176, 10))
        for left, right in ((x + 25, x + 98), (x + 126, x + 178)):
            pygame.draw.polygon(self.screen, (40, 47, 49), [
                (left, rail_y - 17), (right, rail_y - 17),
                (right - 6, rail_y + 3), (left + 6, rail_y + 3),
            ])
            pygame.draw.line(self.screen, (116, 124, 120),
                             (left + 5, rail_y - 14), (right - 5, rail_y - 14), 2)
        pygame.draw.rect(self.screen, (43, 50, 52), (x + 16, rail_y - 21, 182, 7))
        pygame.draw.polygon(self.screen, (46, 53, 53), [
            (x + 181, rail_y - 19), (x + 203, rail_y - 17),
            (x + 198, rail_y - 7), (x + 185, rail_y - 9),
        ])
        pygame.draw.rect(self.screen, (32, 38, 40), (x + 190, rail_y - 16, 14, 8))
        pygame.draw.circle(self.screen, (255, 238, 188), (x + 187, body_y + 57), 5)
        pygame.draw.circle(self.screen, (255, 247, 215), (x + 187, body_y + 57), 2)
        pygame.draw.circle(self.screen, (240, 224, 172), (x + 179, body_y + 70), 3)
        self.text(loco["name"], x + 64, body_y + 70, 9, WHITE, True, "center")
        for wheel_x in (x + 38, x + 83, x + 139, x + 170):
            self.draw_wheel(wheel_x, rail_y + 8, powered=True)
        pygame.draw.line(self.screen, (192, 151, 91),
                         (x + 38, rail_y + 8), (x + 83, rail_y + 8), 3)
        pygame.draw.line(self.screen, (192, 151, 91),
                         (x + 139, rail_y + 8), (x + 170, rail_y + 8), 3)
        for puff in range(3):
            phase = (self.elapsed * 42 + puff * 22) % 70
            smoke_x = int(x + 74 - phase * 0.65)
            smoke_y = int(body_y - 5 - phase * 0.8)
            smoke_radius = max(2, int(4 + phase * 0.09))
            pygame.draw.circle(self.screen, (187, 198, 194), (smoke_x, smoke_y),
                               smoke_radius)

    def draw_steam_locomotive(self, x, body_y, rail_y, color):
        pygame.draw.polygon(self.screen, color, [
            (x + 4, body_y + 38), (x + 23, body_y + 22), (x + 132, body_y + 22),
            (x + 158, body_y + 35), (x + 180, body_y + 51), (x + 174, body_y + 78),
            (x + 5, body_y + 78),
        ])
        pygame.draw.ellipse(self.screen, (80, 127, 99),
                            (x + 32, body_y + 27, 105, 43))
        pygame.draw.ellipse(self.screen, (193, 205, 190),
                            (x + 31, body_y + 27, 105, 43), 2)
        pygame.draw.ellipse(self.screen, (47, 69, 58),
                            (x + 132, body_y + 30, 34, 39))
        pygame.draw.ellipse(self.screen, (185, 196, 180),
                            (x + 132, body_y + 30, 34, 39), 2)
        for band_x in (x + 59, x + 91, x + 123):
            pygame.draw.line(self.screen, (176, 146, 94),
                             (band_x, body_y + 29), (band_x, body_y + 66), 2)
        pygame.draw.line(self.screen, (171, 183, 164),
                         (x + 39, body_y + 37), (x + 125, body_y + 37), 2)
        pygame.draw.polygon(self.screen, (44, 72, 59), [
            (x + 8, body_y + 25), (x + 17, body_y + 5), (x + 58, body_y + 5),
            (x + 67, body_y + 23), (x + 63, body_y + 59), (x + 9, body_y + 59),
        ])
        pygame.draw.polygon(self.screen, (104, 166, 177), [
            (x + 18, body_y + 11), (x + 49, body_y + 11),
            (x + 55, body_y + 25), (x + 14, body_y + 25),
        ])
        pygame.draw.rect(self.screen, (37, 46, 41), (x + 141, body_y + 3, 18, 30))
        pygame.draw.ellipse(self.screen, (43, 53, 47), (x + 137, body_y - 1, 26, 10))
        pygame.draw.rect(self.screen, (37, 45, 40), (x + 5, body_y + 72, 176, 10))
        pygame.draw.polygon(self.screen, (46, 51, 46), [
            (x + 178, rail_y - 15), (x + 202, rail_y - 12),
            (x + 191, rail_y - 5), (x + 176, rail_y - 7),
        ])
        pygame.draw.circle(self.screen, (255, 235, 171), (x + 175, body_y + 53), 6)
        pygame.draw.circle(self.screen, (183, 197, 182), (x + 71, body_y + 7), 5)
        self.text("VALLEY 4-6-0", x + 104, body_y + 55, 9, WHITE, True, "center")
        wheel_positions = (x + 49, x + 86, x + 123)
        wheel_phase = self.distance * 2.0 * math.pi / 2.0
        crank_points = []
        for index, wheel_x in enumerate(wheel_positions):
            phase = wheel_phase + index * 2.0 * math.pi / 3.0
            self.draw_wheel(wheel_x, rail_y + 2, 17, phase, powered=True)
            crank_points.append((wheel_x, int(rail_y + 2 + 11 * math.sin(phase))))
        pygame.draw.lines(self.screen, (204, 172, 115), False, crank_points, 4)
        pygame.draw.line(self.screen, (197, 193, 175),
                         crank_points[0], crank_points[1], 2)
        pygame.draw.rect(self.screen, (34, 41, 38), (x + 178, rail_y - 16, 20, 7))
        for puff in range(4):
            phase = (self.elapsed * 34 + puff * 18) % 76
            smoke_x = int(x + 150 - phase * 0.6)
            smoke_y = int(body_y - 4 - phase * 0.72)
            puff_size = max(2, int((3 + phase * 0.07) * (0.4 + self.steam_output)))
            shade = int(169 + self.steam_output * 65)
            pygame.draw.circle(self.screen, (shade, min(250, shade + 5), shade),
                               (smoke_x, smoke_y), puff_size)

    def draw_bogie(self, x, y):
        pygame.draw.rect(self.screen, (40, 47, 48), (x + 2, y - 10, 32, 7))
        for support_x in (x + 4, x + 30):
            pygame.draw.rect(self.screen, (73, 82, 81),
                             (support_x, y - 23, 3, 15))
        for spring_x in (x + 10, x + 25):
            pygame.draw.line(self.screen, (155, 151, 129),
                             (spring_x, y - 21), (spring_x, y - 11), 2)
            pygame.draw.line(self.screen, (155, 151, 129),
                             (spring_x - 2, y - 18), (spring_x + 2, y - 16), 1)
            pygame.draw.line(self.screen, (155, 151, 129),
                             (spring_x - 2, y - 14), (spring_x + 2, y - 12), 1)
        self.draw_wheel(x + 9, y)
        self.draw_wheel(x + 27, y)

    def draw_wheel(self, x, y, radius=11, phase=None, powered=False):
        if phase is None:
            phase = self.distance * 2.0 * math.pi / 1.8
        pygame.draw.circle(self.screen, (24, 29, 31), (x, y), radius)
        pygame.draw.circle(self.screen, (145, 153, 153), (x, y), max(3, radius - 4), 2)
        for spoke in range(4):
            angle = phase + spoke * math.pi / 2.0
            pygame.draw.line(
                self.screen, (151, 158, 155), (x, y),
                (int(x + math.cos(angle) * (radius - 4)),
                 int(y + math.sin(angle) * (radius - 4))), 1,
            )
        pygame.draw.circle(self.screen, (195, 199, 191), (x, y), 2)
        if powered and self.sanding and self.sand_level > 0.0 and self.speed > 0.5:
            nozzle = (x - radius // 2, y + radius - 5)
            pygame.draw.line(self.screen, (118, 86, 51), nozzle,
                             (nozzle[0] - 3, y + radius + 1), 2)
            for grain in range(5):
                phase = (self.elapsed * 8.0 + x * 0.019 + grain * 0.23) % 1.0
                grain_x = int(nozzle[0] - phase * 14)
                grain_y = int(nozzle[1] + phase * 9 + math.sin(phase * math.pi) * 2)
                grain_color = (190, 145, 83) if grain % 2 else (219, 180, 112)
                pygame.draw.circle(self.screen, grain_color, (grain_x, grain_y),
                                   2 if grain % 3 == 0 else 1)
        if powered and self.wheel_slip and self.speed > 2.0:
            contact = (x, y + radius - 1)
            pulse = 0.5 + 0.5 * math.sin(self.elapsed * 38.0 + x * 0.17)
            glow_radius = 2 + int(2 * self.slip_intensity * pulse)
            pygame.draw.circle(self.screen, (255, 125, 45), contact, glow_radius)
            for spark in range(3):
                phase = (self.elapsed * (7.0 + spark) + x * 0.013 + spark * 0.37) % 1.0
                angle = math.pi + (phase - 0.5) * 0.9
                length = 3 + int(7 * self.slip_intensity * (1.0 - phase))
                end = (
                    int(contact[0] + math.cos(angle) * length),
                    int(contact[1] + math.sin(angle) * length),
                )
                color = (255, 222, 130) if spark == 0 else (255, 151, 66)
                pygame.draw.line(self.screen, color, contact, end, 2 if spark == 0 else 1)

    def draw_bar(self, x, y, width, label, value, color):
        self.text(label, x, y, 11, MUTED, True)
        bar = pygame.Rect(x, y + 18, width, 8)
        pygame.draw.rect(self.screen, (42, 56, 64), bar, border_radius=4)
        pygame.draw.rect(self.screen, color,
                         (bar.x, bar.y, int(bar.width * clamp(value, 0.0, 1.0)), bar.height),
                         border_radius=4)

    def draw_speedometer(self, cx, cy, radius):
        pygame.draw.circle(self.screen, (6, 12, 17), (cx, cy), radius)
        pygame.draw.circle(self.screen, (64, 82, 91), (cx, cy), radius, 2)
        speed_limit = self.locomotive["max_speed"] * TRAIN_SPEED_BOOST
        for tick in range(21):
            fraction = tick / 20.0
            angle = math.radians(135.0 + 270.0 * fraction)
            outer = radius - 4
            inner = radius - (12 if tick % 2 == 0 else 7)
            color = RED if fraction * 200.0 > speed_limit else (161, 179, 183)
            pygame.draw.line(
                self.screen, color,
                (int(cx + math.cos(angle) * inner), int(cy + math.sin(angle) * inner)),
                (int(cx + math.cos(angle) * outer), int(cy + math.sin(angle) * outer)),
                2 if tick % 2 == 0 else 1,
            )
        fraction = clamp(self.speed / 200.0, 0.0, 1.0)
        angle = math.radians(135.0 + 270.0 * fraction)
        endpoint = (int(cx + math.cos(angle) * (radius - 15)),
                    int(cy + math.sin(angle) * (radius - 15)))
        pygame.draw.line(self.screen, ORANGE, (cx, cy), endpoint, 3)
        pygame.draw.circle(self.screen, WHITE, (cx, cy), 3)
        for value in (0, 50, 100, 150, 200):
            fraction = value / 200.0
            label_angle = math.radians(135.0 + 270.0 * fraction)
            label_x = int(cx + math.cos(label_angle) * (radius - 22))
            label_y = int(cy + math.sin(label_angle) * (radius - 22))
            self.text(str(value), label_x, label_y, 7, MUTED, True, "center")
        limit_angle = math.radians(135.0 + 270.0 * speed_limit / 200.0)
        pygame.draw.line(
            self.screen, RED,
            (int(cx + math.cos(limit_angle) * (radius - 17)),
             int(cy + math.sin(limit_angle) * (radius - 17))),
            (int(cx + math.cos(limit_angle) * (radius - 3)),
             int(cy + math.sin(limit_angle) * (radius - 3))),
            3,
        )
        self.text("km/h", cx, cy + 10, 8, WHITE, True, "center")
        self.text(f"MAX {speed_limit:.0f}", cx, cy + radius - 2, 7, RED, True, "midtop")

    def draw_resource_gauge(self, cx, cy, radius, value, maximum, label, color,
                            warning_start=None):
        pygame.draw.circle(self.screen, (6, 12, 17), (cx, cy), radius)
        pygame.draw.circle(self.screen, (64, 82, 91), (cx, cy), radius, 2)
        fraction = clamp(value / maximum, 0.0, 1.0)
        for tick in range(11):
            tick_fraction = tick / 10.0
            angle = math.radians(135.0 + 270.0 * tick_fraction)
            tick_color = color
            if warning_start is not None and tick_fraction >= warning_start:
                tick_color = RED if tick_fraction >= warning_start + 0.18 else ORANGE
            inner = radius - (11 if tick % 2 == 0 else 7)
            outer = radius - 3
            pygame.draw.line(
                self.screen, tick_color,
                (int(cx + math.cos(angle) * inner), int(cy + math.sin(angle) * inner)),
                (int(cx + math.cos(angle) * outer), int(cy + math.sin(angle) * outer)),
                2 if tick % 2 == 0 else 1,
            )
        angle = math.radians(135.0 + 270.0 * fraction)
        pygame.draw.line(
            self.screen, WHITE, (cx, cy),
            (int(cx + math.cos(angle) * (radius - 14)),
             int(cy + math.sin(angle) * (radius - 14))), 2,
        )
        pygame.draw.circle(self.screen, ORANGE, (cx, cy), 3)
        if warning_start is not None:
            warning_angle = math.radians(135.0 + 270.0 * warning_start)
            pygame.draw.line(
                self.screen, ORANGE,
                (int(cx + math.cos(warning_angle) * (radius - 15)),
                 int(cy + math.sin(warning_angle) * (radius - 15))),
                (int(cx + math.cos(warning_angle) * (radius - 2)),
                 int(cy + math.sin(warning_angle) * (radius - 2))), 3,
            )
        self.text(label, cx, cy + radius - 1, 8, color, True, "midtop")

    def draw_rpm_gauge(self, cx, cy, radius):
        pygame.draw.circle(self.screen, (6, 12, 17), (cx, cy), radius)
        pygame.draw.circle(self.screen, (64, 82, 91), (cx, cy), radius, 2)
        for tick in range(13):
            fraction = tick / 12.0
            angle = math.radians(135.0 + 270.0 * fraction)
            color = RED if fraction >= 0.84 else (164, 177, 180)
            inner = radius - (11 if tick % 2 == 0 else 7)
            pygame.draw.line(
                self.screen, color,
                (int(cx + math.cos(angle) * inner),
                 int(cy + math.sin(angle) * inner)),
                (int(cx + math.cos(angle) * (radius - 3)),
                 int(cy + math.sin(angle) * (radius - 3))),
                2 if tick % 2 == 0 else 1,
            )
        for value in (0, 2, 4, 6):
            angle = math.radians(135.0 + 270.0 * value / 6.0)
            self.text(
                str(value),
                int(cx + math.cos(angle) * (radius - 18)),
                int(cy + math.sin(angle) * (radius - 18)),
                8, MUTED, True, "center",
            )
        fraction = clamp(self.rpm / 6000.0, 0.0, 1.0)
        angle = math.radians(135.0 + 270.0 * fraction)
        pygame.draw.line(
            self.screen, ORANGE, (cx, cy),
            (int(cx + math.cos(angle) * (radius - 14)),
             int(cy + math.sin(angle) * (radius - 14))), 3,
        )
        pygame.draw.circle(self.screen, WHITE, (cx, cy), 3)
        self.text(f"{self.rpm:.0f}", cx, cy + 10, 9, WHITE, True, "center")
        self.text("RPM x1000", cx, cy + radius + 1, 8, ORANGE, True, "midtop")

    def mobile_button_layout(self, width, height):
        panel_top = height - 170
        x_start = int(width * 0.57)
        gap = 8
        button_width = max(52, (width - x_start - 22 - gap * 2) // 3)
        button_height = min(48, max(36, int(height * 0.075)))
        first_y = panel_top + 20
        if self.mission_stage == "running":
            controls = (
                ("POWER", "power", ORANGE),
                ("BRAKE", "brake", RED),
                ("SAND", "sand", (189, 139, 77)),
                ("COOL", "cool", CYAN),
                ("HORN", "horn", GREEN),
                ("PAUSE", "pause", (151, 169, 180)),
            )
            result = []
            for index, (label, action, color) in enumerate(controls):
                row, column = divmod(index, 3)
                rect = pygame.Rect(
                    x_start + column * (button_width + gap),
                    first_y + row * (button_height + 8),
                    button_width, button_height,
                )
                result.append((rect, label, action, color))
            return result
        if self.mission_stage in ("loading", "unloading"):
            rect = pygame.Rect(x_start, first_y, width - x_start - 20, button_height * 2 + 8)
            label = "LOAD CARGO" if self.mission_stage == "loading" else "UNLOAD CARGO"
            color = CYAN if self.mission_stage == "loading" else ORANGE
            return [(rect, label, "cargo", color)]
        return []

    def draw_mobile_controls(self, width, height):
        for rect, label, action, color in self.mobile_button_layout(width, height):
            pressed = action in self.touch_controls
            fill = tuple(min(255, channel + 20) for channel in color) if pressed else (22, 39, 49)
            pygame.draw.rect(self.screen, fill, rect, border_radius=10)
            pygame.draw.rect(self.screen, color, rect, width=2, border_radius=10)
            self.text(label, rect.centerx, rect.centery, 11, WHITE, True, "center")

    def draw_hud(self, width, height):
        top = pygame.Rect(14, 12, width - 28, 62)
        pygame.draw.rect(self.screen, (12, 22, 31), top, border_radius=12)
        pygame.draw.rect(self.screen, (50, 74, 84), top, width=1, border_radius=12)
        self.text("RAILWAY 2D", 30, 20, 17, CYAN, True)
        self.text(self.locomotive["name"], 32, 44, 11, WHITE, True)
        self.text(self.locomotive["kind"], 190, 27, 10, MUTED, True)
        self.text(f"¢ {self.credits:,}", width - 28, 24, 14, GREEN, True, "topright")
        self.text(
            f"T{self.upgrades['traction']}  B{self.upgrades['brakes']}  E{self.upgrades['efficiency']}",
            width - 28, 47, 9, MUTED, True, "topright",
        )

        if self.mission_stage == "main_menu":
            message = "MAIN MENU   |   1 PASSENGERS   2 FREIGHT   3 CHOOSE TRAIN"
        elif self.mission_stage == "train_select":
            message = "LOCOMOTIVE SELECT   |   choose any train, no purchase required"
        elif self.mission_stage == "freight_setup":
            message = "FREIGHT BOARD  |  choose cargo, loading point and route; press ENTER"
        elif self.mission_stage == "loading":
            estimate = max(0, int((self.load_target_units - self.loading_elapsed) * 35.0))
            message = (
                f"LOADING {self.cargo_type} at {LOADING_POINTS[self.loading_point_index]}  |  "
                f"{self.load_progress}/{self.load_target_units} loads  |  "
                f"{self.loading_elapsed:.1f}s  |  bonus ¢{estimate}"
            )
        elif self.mission_stage == "unloading":
            message = (
                f"UNLOADING {self.cargo_type} at {self.destination}  |  "
                f"{self.unload_progress}/{self.load_target_units} loads"
            )
        elif self.mission_type == "passenger":
            left = max(0.0, self.mission_target() - self.distance)
            message = f"PASSENGER RUN   |   stop at {self.destination}   |   {left:.0f} m remaining"
        else:
            left = max(0.0, self.mission_target() - self.distance)
            message = (
                f"FREIGHT   |   {self.cargo_type}  {self.cargo_tonnes} t   |   "
                f"{self.destination} below 15 km/h   |   {left:.0f} m"
            )
        message += (
            f"   |   {self.locomotive['mass_tonnes']:,} t   |   "
            f"GRADE {self.track_grade() * 100:+.1f}%"
        )
        mission = pygame.Rect(14, 82, width - 28, 42)
        pygame.draw.rect(self.screen, (13, 24, 33), mission, border_radius=8)
        self.text(message, mission.x + 14, mission.centery, 13, WHITE, True, "midleft")
        if self.mission_stage == "loading":
            progress = self.load_progress / max(1, self.load_target_units)
            self.draw_bar(22, 132, 300, "LOADING CONSIGNMENT",
                          progress, CYAN)
            self.text(f"{self.loading_elapsed:.1f}s  |  bonus ¢{max(0, int((self.load_target_units - self.loading_elapsed) * 35))}",
                      330, 150, 10, GREEN, True, "midleft")
        elif self.mission_stage == "unloading":
            progress = self.unload_progress / max(1, self.load_target_units)
            self.draw_bar(22, 132, 300, "UNLOADING CONSIGNMENT",
                          progress, ORANGE)
            self.text(f"LOAD BONUS ¢{self.loading_bonus}  |  {self.unloading_elapsed:.1f}s",
                      330, 150, 10, GREEN, True, "midleft")

        panel = pygame.Rect(10, height - 170, width - 20, 158)
        pygame.draw.rect(self.screen, (12, 22, 31), panel, border_radius=14)
        pygame.draw.rect(self.screen, (50, 74, 84), panel, width=1, border_radius=14)
        scale = clamp(width / 1280.0, 0.55, 1.0)
        gauge_y = height - 91
        self.draw_speedometer(int(width * 0.065), gauge_y, int(48 * scale))
        self.draw_rpm_gauge(int(width * 0.16), gauge_y, int(40 * scale))
        pressure = (self.steam_pressure if self.locomotive["propulsion"] == "steam"
                    else self.air_pressure)
        self.draw_resource_gauge(
            int(width * 0.255), gauge_y, int(35 * scale), pressure, 12,
            "BOILER" if self.locomotive["propulsion"] == "steam" else "AIR", CYAN,
        )
        self.draw_resource_gauge(
            int(width * 0.35), gauge_y, int(32 * scale), self.sand_level, 100,
            "SAND", ORANGE,
        )
        self.draw_resource_gauge(
            int(width * 0.44), gauge_y, int(32 * scale), self.water_level, 100,
            "WATER", CYAN,
        )
        temp_label = "STEAM C" if self.locomotive["propulsion"] == "steam" else "TEMP C"
        temp_color = RED if self.engine_temp >= 110 else (
            ORANGE if self.engine_temp >= 90 else GREEN
        )
        self.draw_resource_gauge(
            int(width * 0.53), gauge_y, int(32 * scale), self.engine_temp, 130,
            temp_label, temp_color, 90.0 / 130.0,
        )
        self.draw_mobile_controls(width, height)
        if self.engine_temp >= 90.0:
            self.text("ENGINE HOT",
                      width // 2, height - 174, 10,
                      RED if self.engine_temp >= 110 else ORANGE, True, "midtop")
        elif self.wheel_slip:
            self.text("WHEEL SLIP",
                      width // 2, height - 174, 10, ORANGE, True, "midtop")
        if self.audio.error:
            self.text("AUDIO UNAVAILABLE: " + self.audio.error[:70], 30, height - 45, 11, RED)
        elif self.audio.warning:
            self.text("AUDIO: " + self.audio.warning[:70], 30, height - 45, 11, ORANGE)

    def draw_overlay(self, width, height):
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        overlay.fill((5, 11, 18, 185))
        self.screen.blit(overlay, (0, 0))
        card = pygame.Rect(width // 2 - 300, height // 2 - 220, 600, 440)
        pygame.draw.rect(self.screen, (17, 30, 41), card, border_radius=18)
        pygame.draw.rect(self.screen, (66, 100, 112), card, width=2, border_radius=18)
        self.text("RAILWAY 2D", card.centerx, card.y + 32, 30, CYAN, True, "midtop")
        if self.mission_stage == "freight_setup":
            self.text("FREIGHT CONTRACT SETUP", card.centerx, card.y + 76,
                      13, MUTED, True, "midtop")
            for index, cargo in enumerate(FREIGHT_CARGOES):
                column = index % 2
                row_index = index // 2
                box = pygame.Rect(card.x + 32 + column * 270,
                                  card.y + 108 + row_index * 59, 250, 49)
                selected = index == self.cargo_index
                pygame.draw.rect(self.screen, (25, 45, 55), box, border_radius=8)
                pygame.draw.rect(self.screen, CYAN if selected else (65, 83, 91),
                                 box, width=2, border_radius=8)
                self.text(f"{index + 1}  {cargo['name']}", box.x + 12, box.y + 7,
                          12, WHITE, True)
                self.text(f"{cargo['tonnes']} tonnes  |  {cargo['loading_units']} loads",
                          box.x + 12, box.y + 28, 10, MUTED)
            point = LOADING_POINTS[self.loading_point_index]
            destination = FREIGHT_DESTINATIONS[self.loading_point_index]
            route, route_distance = ROUTE_OPTIONS[self.route_index]
            self.text(f"LOAD AT   {point}   ← / → to change", card.centerx,
                      card.y + 238, 12, ORANGE, True, "midtop")
            self.text(f"ROUTE   {route}  {route_distance:.0f} m  →  {destination}   ↑ / ↓ to change",
                      card.centerx, card.y + 264, 11, WHITE, True, "midtop")
            self.text(f"TRAIN  {self.locomotive['name']}  [C: cycle free trains]",
                      card.centerx, card.y + 293, 11, GREEN, True, "midtop")
            self.text("ENTER: start loading   |   press SPACE quickly to load each cargo lot",
                      card.centerx, card.y + 325, 11, WHITE, True, "midtop")
            self.text("Faster loading = larger delivery bonus   |   N: main menu",
                      card.centerx, card.y + 354, 10, MUTED, False, "midtop")
            if self.mission_result:
                self.text(self.mission_result, card.centerx, card.y + 389, 12,
                          RED, True, "midtop")
            return

        if self.mission_stage == "train_select":
            self.text("LOCOMOTIVE SELECT  |  ALL TRAINS AVAILABLE",
                      card.centerx, card.y + 76, 13, MUTED, True, "midtop")
            for index, locomotive in enumerate(LOCOMOTIVES):
                column = index % 2
                row_index = index // 2
                box = pygame.Rect(card.x + 32 + column * 270,
                                  card.y + 111 + row_index * 86, 250, 73)
                selected = index == self.train_selection_index
                pygame.draw.rect(self.screen, (25, 45, 55), box, border_radius=8)
                pygame.draw.rect(
                    self.screen, CYAN if selected else (65, 83, 91),
                    box, width=2, border_radius=8,
                )
                self.text(f"{index + 1}  {locomotive['name']}",
                          box.x + 12, box.y + 10, 12, WHITE, True)
                self.text(locomotive["kind"], box.x + 12, box.y + 31,
                          9, locomotive["color"], True)
                self.text(
                    f"{locomotive['max_speed']} km/h  |  "
                    f"{locomotive['mass_tonnes']:,} t  |  "
                    f"{locomotive['power_kw']:,} kW",
                    box.x + 12, box.y + 51, 9, MUTED,
                )
            preview = LOCOMOTIVES[self.train_selection_index]
            self.text(f"SELECTED: {preview['name']}  —  NO COST",
                      card.centerx, card.y + 301, 12, GREEN, True, "midtop")
            self.text("← / → preview     ENTER confirm     ESC back",
                      card.centerx, card.y + 337, 11, WHITE, True, "midtop")
            self.text("1–4 selects a locomotive directly",
                      card.centerx, card.y + 366, 10, MUTED, False, "midtop")
            return

        self.text("MAIN MENU  |  CHOOSE YOUR SHIFT",
                  card.centerx, card.y + 76, 13, MUTED, True, "midtop")

        options = (
            ("1", "PASSENGER SERVICE", "Stop at the assigned station below 5 km/h", CYAN),
            ("2", "FREIGHT DELIVERY", "Choose cargo, loading point and route; earn a fast-loading bonus", ORANGE),
            ("3", "CHOOSE LOCOMOTIVE", "Select any of the four trains for free", GREEN),
        )
        for index, (key, title, detail, color) in enumerate(options):
            row = pygame.Rect(card.x + 32, card.y + 108 + index * 64,
                              card.width - 64, 54)
            pygame.draw.rect(self.screen, (23, 41, 52), row, border_radius=9)
            pygame.draw.rect(self.screen, color, row, width=2, border_radius=9)
            self.text(key, row.x + 21, row.centery, 22, color, True, "center")
            self.text(title, row.x + 50, row.y + 10, 14, WHITE, True)
            self.text(detail, row.x + 50, row.y + 31, 10, MUTED)

        self.text(f"TRAIN  {self.locomotive['name']}",
                  card.centerx, card.y + 307, 12, WHITE, True, "midtop")
        levels = (f"T{self.upgrades['traction']}  B{self.upgrades['brakes']}  "
                  f"E{self.upgrades['efficiency']}")
        self.text(f"UPGRADES {levels}   |   Credits ¢{self.credits:,}",
                  card.centerx, card.y + 329, 11, GREEN, False, "midtop")
        self.text("6 traction   7 brakes   8 efficiency  (buy upgrades)",
                  card.centerx, card.y + 354, 10, WHITE, True, "midtop")
        self.text("W/S power/brake   •   X sand   •   C cooling   •   H horn",
                  card.centerx, card.y + 376, 9, MUTED, False, "midtop")
        if self.mission_result:
            self.text(self.mission_result, card.centerx, card.y + 386, 14,
                      GREEN if "delivered" in self.mission_result.lower()
                      or "safely" in self.mission_result.lower() else RED,
                      True, "midtop")

    def draw(self):
        width, height = self.screen.get_size()
        self.draw_world(width, height)
        self.draw_hud(width, height)
        if (self.mission_stage in ("main_menu", "train_select", "freight_setup")
                or self.mission_result or self.paused):
            self.draw_overlay(width, height)
            if self.paused:
                self.text("PAUSED  |  Press P to resume", width // 2, height // 2 + 198,
                          13, WHITE, True, "midtop")
        pygame.display.flip()

    def cycle_locomotive(self):
        next_index = (self.locomotive_index + 1) % len(LOCOMOTIVES)
        self.locomotive_index = next_index
        self.train_selection_index = next_index
        self.mission_result = ""
        self.reset_train()

    def confirm_train_selection(self):
        self.locomotive_index = self.train_selection_index
        self.reset_train()
        self.mission_result = ""
        self.mission_stage = "main_menu"

    def buy_upgrade(self, category):
        if category not in self.upgrades:
            return
        if self.upgrades[category] >= 3:
            self.mission_result = f"{category.title()} system is fully upgraded"
            return
        cost = 350 * (self.upgrades[category] + 1)
        if self.credits < cost:
            self.mission_result = f"Need ¢{cost:,} for the next upgrade"
            return
        self.credits -= cost
        self.upgrades[category] += 1
        self.mission_result = f"{category.title()} upgrade installed: level {self.upgrades[category]}"

    def _refresh_touch_controls(self):
        held = {"power", "brake", "sand", "cool", "horn"}
        self.touch_controls = {
            action for action in self.pointer_actions.values() if action in held
        }

    def _menu_tap(self, position):
        x, y = position
        width, height = self.screen.get_size()
        card = pygame.Rect(width // 2 - 300, height // 2 - 220, 600, 440)
        if self.mission_stage in ("main_menu", "result"):
            row = (y - (card.y + 108)) // 64
            if card.x + 24 <= x <= card.right - 24 and 0 <= row <= 2:
                key = (pygame.K_1, pygame.K_2, pygame.K_3)[row]
                self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            return
        if self.mission_stage == "train_select":
            column = 0 if x < card.centerx else 1
            row = 0 if y < card.y + 197 else 1
            index = row * 2 + column
            if 0 <= index < len(LOCOMOTIVES):
                key = pygame.K_1 + index
                self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            return
        if self.mission_stage == "freight_setup":
            cargo_top = card.y + 108
            if cargo_top <= y < cargo_top + 118 and card.x + 32 <= x < card.right - 32:
                column = 0 if x < card.centerx else 1
                row = int((y - cargo_top) // 59)
                index = row * 2 + column
                if 0 <= index < len(FREIGHT_CARGOES):
                    key = pygame.K_1 + index
                    self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
                return
            if card.y + 222 <= y < card.y + 251:
                key = pygame.K_LEFT if x < card.centerx else pygame.K_RIGHT
                self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            elif card.y + 251 <= y < card.y + 282:
                key = pygame.K_DOWN if x < card.centerx else pygame.K_UP
                self.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key))
            elif y >= card.y + 300:
                self.handle_event(
                    pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
                )

    def _pointer_down(self, pointer, position):
        if self.mission_stage == "running":
            for rect, _label, action, _color in self.mobile_button_layout(
                *self.screen.get_size()
            ):
                if rect.collidepoint(position):
                    if action == "pause":
                        self.paused = not self.paused
                    else:
                        self.pointer_actions[pointer] = action
                        self._refresh_touch_controls()
                    return
        elif self.mission_stage in ("loading", "unloading"):
            for rect, _label, action, _color in self.mobile_button_layout(
                *self.screen.get_size()
            ):
                if action == "cargo" and rect.collidepoint(position):
                    if self.mission_stage == "loading":
                        self.load_one_unit()
                    else:
                        self.unload_one_unit()
                    return
        self._menu_tap(position)

    def _pointer_move(self, pointer, position):
        if pointer not in self.pointer_actions:
            return
        action = None
        for rect, _label, candidate, _color in self.mobile_button_layout(
            *self.screen.get_size()
        ):
            if candidate != "pause" and rect.collidepoint(position):
                action = candidate
                break
        if action is None:
            self.pointer_actions.pop(pointer, None)
        else:
            self.pointer_actions[pointer] = action
        self._refresh_touch_controls()

    def _pointer_up(self, pointer):
        self.pointer_actions.pop(pointer, None)
        self._refresh_touch_controls()

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not getattr(event, "touch", False):
                self._pointer_down("mouse", event.pos)
            return
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if not getattr(event, "touch", False):
                self._pointer_up("mouse")
            return
        if event.type == pygame.MOUSEMOTION and "mouse" in self.pointer_actions:
            self._pointer_move("mouse", event.pos)
            return
        if event.type == pygame.FINGERDOWN:
            width, height = self.screen.get_size()
            pointer = ("finger", event.finger_id)
            self._pointer_down(pointer, (int(event.x * width), int(event.y * height)))
            return
        if event.type == pygame.FINGERMOTION:
            width, height = self.screen.get_size()
            pointer = ("finger", event.finger_id)
            self._pointer_move(pointer, (int(event.x * width), int(event.y * height)))
            return
        if event.type == pygame.FINGERUP:
            self._pointer_up(("finger", event.finger_id))
            return
        if event.type != pygame.KEYDOWN:
            return

        key = event.key
        if key == pygame.K_ESCAPE:
            if self.mission_stage == "train_select":
                self.mission_stage = "main_menu"
                return
            self.running = False
        elif key == pygame.K_n:
            self.mission_type = None
            self.mission_stage = "main_menu"
            self.mission_result = ""
            self.reset_train()
        elif self.mission_stage == "train_select":
            if key in (pygame.K_LEFT, pygame.K_UP):
                self.train_selection_index = (
                    self.train_selection_index - 1
                ) % len(LOCOMOTIVES)
            elif key in (pygame.K_RIGHT, pygame.K_DOWN):
                self.train_selection_index = (
                    self.train_selection_index + 1
                ) % len(LOCOMOTIVES)
            elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                self.train_selection_index = key - pygame.K_1
                self.confirm_train_selection()
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.confirm_train_selection()
        elif self.mission_stage == "freight_setup":
            if key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                self.cargo_index = key - pygame.K_1
                cargo = FREIGHT_CARGOES[self.cargo_index]
                self.cargo_type = cargo["name"]
                self.cargo_tonnes = cargo["tonnes"]
                self.load_target_units = cargo["loading_units"]
            elif key == pygame.K_LEFT:
                self.loading_point_index = (self.loading_point_index - 1) % len(LOADING_POINTS)
            elif key == pygame.K_RIGHT:
                self.loading_point_index = (self.loading_point_index + 1) % len(LOADING_POINTS)
            elif key == pygame.K_UP:
                self.route_index = (self.route_index + 1) % len(ROUTE_OPTIONS)
            elif key == pygame.K_DOWN:
                self.route_index = (self.route_index - 1) % len(ROUTE_OPTIONS)
            elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.start_mission("freight")
            elif key == pygame.K_c:
                self.cycle_locomotive()
        elif self.mission_stage == "loading":
            if key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self.load_one_unit()
        elif self.mission_stage == "unloading":
            if key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self.unload_one_unit()
        elif key == pygame.K_p and self.mission_stage == "running":
            self.paused = not self.paused
        elif self.mission_stage in ("main_menu", "result"):
            if key == pygame.K_1:
                self.start_mission("passenger")
            elif key == pygame.K_2:
                self.open_freight_setup()
            elif key == pygame.K_3:
                self.train_selection_index = self.locomotive_index
                self.mission_stage = "train_select"
            elif key == pygame.K_6:
                self.buy_upgrade("traction")
            elif key == pygame.K_7:
                self.buy_upgrade("brakes")
            elif key == pygame.K_8:
                self.buy_upgrade("efficiency")

    def run(self):
        try:
            while self.running:
                dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
                for event in pygame.event.get():
                    self.handle_event(event)
                self.update(dt)
                self.draw()
        finally:
            self.audio.close()
            pygame.quit()


if __name__ == "__main__":
    TrainGame().run()
