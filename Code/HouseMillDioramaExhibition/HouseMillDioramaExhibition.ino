#include <Arduino.h>
#include <ArduinoJson.h>
#include <Preferences.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <time.h>

#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif

// ----------------------- User settings -----------------------

const char *WIFI_SSID = "UCL_IoT";
const char *WIFI_PASSWORD = "casace2026";

// const char *WIFI_SSID = "CE-Hub-Student";
// const char *WIFI_PASSWORD = "casa-ce-gagarin-public-service";

const char *MQTT_HOST = "mqtt.cetools.org";
const uint16_t MQTT_PORT = 1884;
const char *MQTT_USER = "student";
const char *MQTT_PASSWORD = "ce2021-mqtt-forget-whale";

const char *MQTT_TOPIC_STATUS = "student/housemill/flood/status";
const char *MQTT_TOPIC_COMMAND = "student/housemill/flood/command";
const char *MQTT_TOPIC_CONTROL_STATUS = "student/housemill/control/diorama/status";
const char *MQTT_CONTROL_OFFLINE_PAYLOAD = "{\"device\":\"diorama\",\"online\":false}";
const bool MQTT_RETAIN_STATUS = false;

// Pump schedule uses Europe/London local time, including BST daylight saving.
// If NTP has not provided a valid time, the schedule fails open and the
// exhibition continues with its existing behavior.
const char *TIME_ZONE = "GMT0BST,M3.5.0/1,M10.5.0/2";
const char *NTP_SERVER_PRIMARY = "pool.ntp.org";
const char *NTP_SERVER_SECONDARY = "time.google.com";
const char *NTP_SERVER_TERTIARY = "time.cloudflare.com";
const uint8_t PUMP_WINDOW_START_HOUR = 10;
const uint8_t PUMP_WINDOW_END_HOUR = 18;
const time_t MIN_VALID_UNIX_TIME = 1704067200; // 2024-01-01 UTC.

// DRV8833 wiring.
const int PUMP_PWM_PIN = 4;        // DRV8833 AIN1
const int PUMP_LOW_PIN = -1;       // Set to a GPIO if AIN2 is controlled by ESP32. Use -1 if AIN2 is wired to GND.
const int DRV_SLEEP_PIN = -1;      // Set to a GPIO if SLP is controlled by ESP32. Use -1 if SLP is wired to 3V3.
const uint8_t PUMP_PWM_CHANNEL = 0;
const uint32_t PUMP_PWM_FREQ_HZ = 20000;
const uint8_t PUMP_PWM_BITS = 8;

// This pump starts reliably only at full power, so water volume is tuned by time.
const uint8_t PUMP_FULL_DUTY = 255;
const uint8_t DEFAULT_PUMP_DUTY = PUMP_FULL_DUTY;
const uint16_t START_KICK_MS = 1200;

const uint8_t PUMP_MODE_PWM_AFTER_KICK = 0;
const uint8_t PUMP_MODE_BURST_FULL_POWER = 1;
const uint8_t DEFAULT_PUMP_MODE = PUMP_MODE_PWM_AFTER_KICK;
const uint8_t DEFAULT_BURST_PERCENT = 100;
const uint16_t BURST_PERIOD_MS = 2500;

// Exhibition timing. Pumping now happens during countdown and stops when ETA is 0h.
// The shorter topoff stage is only a visual transition kept for compatibility.
const uint32_t DEFAULT_COUNTDOWN_SECONDS = 110; // Meter goes 24h -> 0h; pump stops at 0h.
const uint32_t DEFAULT_TOPOFF_SECONDS = 1;      // Minimal 0h transition; no extra pumping.
const uint32_t DEFAULT_HOLD_ZERO_SECONDS = 8;   // Short peak-water hold at 0h.
const uint32_t DEFAULT_DRAIN_SECONDS = 25;      // Faster reset while the water drains.
const uint32_t DEFAULT_GAP_SECONDS = 15UL * 60UL; // 15-minute rest between cycles.
const uint32_t MIN_GAP_SECONDS = 8;             // Never allow less than the previous 8-second gap.
const bool DEFAULT_HOLD_PUMP_DURING_ZERO = false;
const uint16_t DEFAULT_PUMP_VOLUME_PERCENT = 100;
const uint16_t MIN_PUMP_VOLUME_PERCENT = 50;
const uint16_t MAX_PUMP_VOLUME_PERCENT = 150;
const uint32_t MAX_ACTIVE_CYCLE_MS = 180000UL;
const uint32_t MIN_COMPRESSED_HOLD_MS = 3000UL;
const uint32_t MIN_COMPRESSED_DRAIN_MS = 5000UL;
const uint32_t MIN_GAP_MS = MIN_GAP_SECONDS * 1000UL;
const uint8_t LOW_RISK_BASE_PUMP_PERCENT = 45;
const uint8_t MEDIUM_RISK_BASE_PUMP_PERCENT = 72;
const uint8_t HIGH_RISK_BASE_PUMP_PERCENT = 120;

const uint32_t CONFIG_VERSION = 4;
const uint32_t PREVIOUS_CONFIG_VERSION = 3;
const uint32_t PUBLISH_INTERVAL_MS = 1000;
const uint32_t SERIAL_STATUS_INTERVAL_MS = 5000;
const float FLOOD_DISPLAY_MAX_HOURS = 24.0f;
const bool DEFAULT_RANDOMISE_INTENSITY = true;
const uint32_t PUMP_START_DELAY_SECONDS = 5;  // Meter starts moving before every pump stage.

// One risk scenario is chosen randomly at the start of each exhibition cycle.
// Low risk starts pumping later. Medium risk pumps for the new longer countdown.
// High risk starts immediately and can show virtual excess water.
const uint8_t LOW_RISK_WEIGHT_PERCENT = 35;
const uint8_t MEDIUM_RISK_WEIGHT_PERCENT = 40;
const float HIGH_RISK_EXTRA_DRAIN_START_PERCENT = 20.0f;
const uint32_t MODEL_RAISE_COMPENSATION_SECONDS = 0;  // Peak pump compensation is folded into countdown.
const uint8_t DRAIN_TIME_PERCENT = 100;
const uint8_t LOW_RISK_DRAIN_TIME_PERCENT = 60;

// ----------------------- Runtime state -----------------------

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
Preferences prefs;

enum class Phase : uint8_t {
  Gap,
  CountdownToFlood,
  FinalRise,
  HoldAtZero,
  Draining,
  Paused
};

enum class RiskLevel : uint8_t {
  Low,
  Medium,
  High
};

enum class RunMode : uint8_t {
  Exhibition,
  Simulation,
  Paused
};

struct RuntimeConfig {
  uint8_t pumpDuty = DEFAULT_PUMP_DUTY;
  uint8_t pumpMode = DEFAULT_PUMP_MODE;
  uint8_t burstPercent = DEFAULT_BURST_PERCENT;
  uint32_t countdownMs = DEFAULT_COUNTDOWN_SECONDS * 1000UL;
  uint32_t topoffMs = DEFAULT_TOPOFF_SECONDS * 1000UL;
  uint32_t holdZeroMs = DEFAULT_HOLD_ZERO_SECONDS * 1000UL;
  uint32_t drainMs = DEFAULT_DRAIN_SECONDS * 1000UL;
  uint32_t gapMs = DEFAULT_GAP_SECONDS * 1000UL;
  uint16_t pumpVolumePercent = DEFAULT_PUMP_VOLUME_PERCENT;
  bool holdPumpDuringZero = DEFAULT_HOLD_PUMP_DURING_ZERO;
  bool randomiseIntensity = DEFAULT_RANDOMISE_INTENSITY;
};

struct CycleTiming {
  uint32_t countdownMs = DEFAULT_COUNTDOWN_SECONDS * 1000UL;
  uint32_t topoffMs = DEFAULT_TOPOFF_SECONDS * 1000UL;
  uint32_t holdMs = DEFAULT_HOLD_ZERO_SECONDS * 1000UL;
  uint32_t drainMs = DEFAULT_DRAIN_SECONDS * 1000UL;
  uint32_t gapMs = DEFAULT_GAP_SECONDS * 1000UL;
  uint32_t pumpStartMs = PUMP_START_DELAY_SECONDS * 1000UL;
  uint32_t pumpDurationMs = 0;
  bool volumeLimited = false;
};

struct PumpScheduleState {
  bool timeKnown = false;
  bool windowOpen = true;
  uint8_t hour = 0;
  uint8_t minute = 0;
};

RuntimeConfig cfg;
Phase phase = Phase::Gap;
uint32_t phaseStartedMs = 0;
uint32_t lastPublishMs = 0;
uint32_t lastSerialStatusMs = 0;
uint32_t lastWifiAttemptMs = 0;
uint32_t lastMqttAttemptMs = 0;
uint32_t sequenceNumber = 0;
uint32_t cycleNumber = 0;
RiskLevel currentRisk = RiskLevel::Medium;
int cycleIntensity = 60;
RunMode runMode = RunMode::Exhibition;
bool pumpEnabled = true;
uint16_t activePumpVolumePercent = DEFAULT_PUMP_VOLUME_PERCENT;
CycleTiming cycleTiming;

bool pumpRequested = false;
bool pumpKickActive = false;
uint32_t pumpKickStartedMs = 0;
uint32_t burstCycleStartedMs = 0;
String serialLine;

template <typename T>
T clampValue(T value, T low, T high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
}

template <typename T>
T maxValue(T left, T right) {
  return left > right ? left : right;
}

float clamp01(float value) {
  return clampValue(value, 0.0f, 1.0f);
}

float round1(float value) {
  return roundf(value * 10.0f) / 10.0f;
}

void attachPwm(uint8_t pin, uint8_t channel, uint32_t freqHz, uint8_t resolutionBits) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(pin, freqHz, resolutionBits);
#else
  ledcSetup(channel, freqHz, resolutionBits);
  ledcAttachPin(pin, channel);
#endif
}

void writePwm(uint8_t pin, uint8_t channel, uint32_t duty) {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(pin, duty);
#else
  ledcWrite(channel, duty);
#endif
}

const char *phaseName(Phase value) {
  switch (value) {
    case Phase::Gap:
      return "gap";
    case Phase::CountdownToFlood:
      return "countdown";
    case Phase::FinalRise:
      return "topoff";
    case Phase::HoldAtZero:
      return "hold";
    case Phase::Draining:
      return "drain";
    case Phase::Paused:
      return "paused";
  }
  return "unknown";
}

const char *riskName(RiskLevel value) {
  switch (value) {
    case RiskLevel::Low:
      return "low";
    case RiskLevel::Medium:
      return "medium";
    case RiskLevel::High:
      return "high";
  }
  return "unknown";
}

const char *runModeName(RunMode value) {
  switch (value) {
    case RunMode::Exhibition:
      return "exhibition";
    case RunMode::Simulation:
      return "simulation";
    case RunMode::Paused:
      return "pause";
  }
  return "pause";
}

uint32_t activeCycleMs() {
  return cycleTiming.countdownMs + cycleTiming.topoffMs + cycleTiming.holdMs +
         cycleTiming.drainMs;
}

uint32_t completeCycleMs() {
  return activeCycleMs() + cycleTiming.gapMs;
}

void reducePhaseToFit(uint32_t &durationMs, uint32_t minimumMs, uint32_t &excessMs) {
  if (excessMs == 0 || durationMs <= minimumMs) return;

  const uint32_t reducibleMs = durationMs - minimumMs;
  const uint32_t reductionMs = reducibleMs < excessMs ? reducibleMs : excessMs;
  durationMs -= reductionMs;
  excessMs -= reductionMs;
}

uint32_t baselinePumpLeadMs() {
  return min(cfg.countdownMs, PUMP_START_DELAY_SECONDS * 1000UL);
}

uint8_t riskBasePumpPercent() {
  if (currentRisk == RiskLevel::Low) {
    return LOW_RISK_BASE_PUMP_PERCENT;
  }
  if (currentRisk == RiskLevel::Medium) {
    return MEDIUM_RISK_BASE_PUMP_PERCENT;
  }
  return HIGH_RISK_BASE_PUMP_PERCENT;
}

void recalculateCycleTiming() {
  const uint32_t leadMs = baselinePumpLeadMs();
  const uint32_t baselinePumpMs = static_cast<uint32_t>(
      (static_cast<uint64_t>(cfg.countdownMs) * riskBasePumpPercent()) / 100ULL);
  const uint32_t requestedPumpMs = static_cast<uint32_t>(
      (static_cast<uint64_t>(baselinePumpMs) * activePumpVolumePercent) / 100ULL);

  cycleTiming.pumpStartMs = leadMs;
  cycleTiming.pumpDurationMs = requestedPumpMs;
  cycleTiming.countdownMs = leadMs + requestedPumpMs;
  cycleTiming.topoffMs = maxValue<uint32_t>(cfg.topoffMs, 1000UL);
  cycleTiming.holdMs = cfg.holdZeroMs;
  cycleTiming.drainMs = currentRisk == RiskLevel::Low
                            ? maxValue<uint32_t>((cfg.drainMs * LOW_RISK_DRAIN_TIME_PERCENT) / 100UL,
                                                 MIN_COMPRESSED_DRAIN_MS)
                            : maxValue<uint32_t>((cfg.drainMs * DRAIN_TIME_PERCENT) / 100UL,
                                                 MIN_COMPRESSED_DRAIN_MS);
  cycleTiming.gapMs = cfg.gapMs;
  cycleTiming.volumeLimited = false;

  const uint32_t totalMs = activeCycleMs();
  uint32_t excessMs = totalMs > MAX_ACTIVE_CYCLE_MS
                          ? totalMs - MAX_ACTIVE_CYCLE_MS
                          : 0;

  // Keep the active flood demonstration within three minutes. The separate
  // inter-cycle gap is never compressed.
  reducePhaseToFit(cycleTiming.holdMs,
                   min(cycleTiming.holdMs, MIN_COMPRESSED_HOLD_MS), excessMs);
  reducePhaseToFit(cycleTiming.drainMs,
                   min(cycleTiming.drainMs, MIN_COMPRESSED_DRAIN_MS), excessMs);
  reducePhaseToFit(cycleTiming.topoffMs, 1000UL, excessMs);
  reducePhaseToFit(cycleTiming.countdownMs, 5000UL, excessMs);

  if (cycleTiming.pumpStartMs + cycleTiming.pumpDurationMs > cycleTiming.countdownMs) {
    cycleTiming.volumeLimited = true;
    if (cycleTiming.pumpDurationMs >= cycleTiming.countdownMs) {
      cycleTiming.pumpStartMs = 0;
      cycleTiming.pumpDurationMs = cycleTiming.countdownMs;
    } else {
      cycleTiming.pumpStartMs = cycleTiming.countdownMs - cycleTiming.pumpDurationMs;
    }
  }
}

RiskLevel randomRiskLevel() {
  const uint8_t roll = random(0, 100);
  if (roll < LOW_RISK_WEIGHT_PERCENT) {
    return RiskLevel::Low;
  }
  if (roll < LOW_RISK_WEIGHT_PERCENT + MEDIUM_RISK_WEIGHT_PERCENT) {
    return RiskLevel::Medium;
  }
  return RiskLevel::High;
}

void chooseRiskScenario(RiskLevel level) {
  currentRisk = level;
  activePumpVolumePercent = cfg.pumpVolumePercent;
  recalculateCycleTiming();
  cycleNumber++;

  if (currentRisk == RiskLevel::Low) {
    cycleIntensity = random(10, 36);
  } else if (currentRisk == RiskLevel::Medium) {
    cycleIntensity = random(45, 71);
  } else {
    cycleIntensity = random(80, 101);
  }

  Serial.print("New risk scenario: ");
  Serial.print(riskName(currentRisk));
  Serial.print(" intensity=");
  Serial.print(cycleIntensity);
  Serial.print(" volume=");
  Serial.print(activePumpVolumePercent);
  Serial.print("% cycle=");
  Serial.print(completeCycleMs() / 1000UL);
  Serial.println("s");
}

void chooseRandomRiskScenario() {
  chooseRiskScenario(randomRiskLevel());
}

uint8_t riskBurstPercent() {
  if (currentRisk == RiskLevel::Low) {
    return 55;
  }
  return 100;
}

float countdownWaterTargetPercent() {
  const uint32_t highRiskReferencePumpMs = static_cast<uint32_t>(
      (static_cast<uint64_t>(cfg.countdownMs) * HIGH_RISK_BASE_PUMP_PERCENT) / 100ULL);
  if (highRiskReferencePumpMs == 0) {
    return 0.0f;
  }
  return 100.0f * static_cast<float>(cycleTiming.pumpDurationMs) /
         static_cast<float>(highRiskReferencePumpMs);
}

float topoffWaterTargetPercent() {
  return countdownWaterTargetPercent();
}

float holdWaterTargetPercent() {
  return countdownWaterTargetPercent();
}

float drainEndWaterTargetPercent() {
  if (currentRisk == RiskLevel::High) {
    return HIGH_RISK_EXTRA_DRAIN_START_PERCENT;
  }
  return 0.0f;
}

float countdownPumpStopProgress() {
  return 1.0f;
}

float countdownPumpStartProgress() {
  if (cycleTiming.countdownMs == 0) {
    return 0.0f;
  }
  return clamp01(static_cast<float>(cycleTiming.pumpStartMs) /
                 static_cast<float>(cycleTiming.countdownMs));
}

float activeCountdownPumpProgress() {
  const float p = phaseProgress();
  const float start = countdownPumpStartProgress();
  const float stop = maxValue(countdownPumpStopProgress(), start + 0.05f);

  if (p <= start) {
    return 0.0f;
  }
  if (p >= stop) {
    return 1.0f;
  }
  return clamp01((p - start) / (stop - start));
}

float holdPumpStopProgress() {
  const float raisedModelCompensation =
      clampValue(static_cast<float>(MODEL_RAISE_COMPENSATION_SECONDS * 1000UL) /
                     static_cast<float>(maxValue<uint32_t>(cfg.holdZeroMs, 1000UL)),
                 0.0f, 1.0f);

  if (currentRisk == RiskLevel::High) {
    return maxValue(0.60f, raisedModelCompensation);
  }
  if (currentRisk == RiskLevel::Medium) {
    return cfg.holdPumpDuringZero ? 1.0f : raisedModelCompensation;
  }
  return 0.0f;
}

uint32_t effectiveTopoffMs() {
  return cycleTiming.topoffMs;
}

uint32_t effectiveDrainMs() {
  return cycleTiming.drainMs;
}

uint32_t phaseDurationMs() {
  switch (phase) {
    case Phase::Gap:
      return cycleTiming.gapMs;
    case Phase::CountdownToFlood:
      return cycleTiming.countdownMs;
    case Phase::FinalRise:
      return effectiveTopoffMs();
    case Phase::HoldAtZero:
      return cycleTiming.holdMs;
    case Phase::Draining:
      return effectiveDrainMs();
    case Phase::Paused:
      return 0;
  }
  return 0;
}

float phaseProgress() {
  const uint32_t duration = phaseDurationMs();
  if (duration == 0) return 0.0f;
  return clamp01(static_cast<float>(millis() - phaseStartedMs) / static_cast<float>(duration));
}

void saveRuntimeConfig() {
  prefs.begin("diorama", false);
  prefs.putUInt("version", CONFIG_VERSION);
  prefs.putUChar("duty", cfg.pumpDuty);
  prefs.putUChar("mode", cfg.pumpMode);
  prefs.putUChar("burst", cfg.burstPercent);
  prefs.putUInt("count", cfg.countdownMs);
  prefs.putUInt("topoff", cfg.topoffMs);
  prefs.putUInt("hold", cfg.holdZeroMs);
  prefs.putUInt("drain", cfg.drainMs);
  prefs.putUInt("gap", cfg.gapMs);
  prefs.putUShort("volume", cfg.pumpVolumePercent);
  prefs.putBool("holdpump", cfg.holdPumpDuringZero);
  prefs.putBool("rand", cfg.randomiseIntensity);
  prefs.end();
}

void loadRuntimeConfig() {
  prefs.begin("diorama", true);
  const uint32_t savedVersion = prefs.getUInt("version", 0);
  const bool useSavedConfig =
      savedVersion == CONFIG_VERSION || savedVersion == PREVIOUS_CONFIG_VERSION;
  const bool migratePreviousConfig = savedVersion == PREVIOUS_CONFIG_VERSION;

  if (useSavedConfig) {
    cfg.pumpDuty = prefs.getUChar("duty", cfg.pumpDuty);
    cfg.pumpMode = prefs.getUChar("mode", cfg.pumpMode);
    cfg.burstPercent = prefs.getUChar("burst", cfg.burstPercent);
    cfg.countdownMs = prefs.getUInt("count", cfg.countdownMs);
    cfg.topoffMs = prefs.getUInt("topoff", cfg.topoffMs);
    cfg.holdZeroMs = prefs.getUInt("hold", cfg.holdZeroMs);
    cfg.drainMs = prefs.getUInt("drain", cfg.drainMs);
    cfg.gapMs = prefs.getUInt("gap", cfg.gapMs);
    cfg.pumpVolumePercent = prefs.getUShort("volume", cfg.pumpVolumePercent);
    cfg.holdPumpDuringZero = prefs.getBool("holdpump", cfg.holdPumpDuringZero);
    cfg.randomiseIntensity = prefs.getBool("rand", cfg.randomiseIntensity);
  }
  prefs.end();

  if (migratePreviousConfig && cfg.gapMs <= MIN_GAP_MS) {
    cfg.gapMs = DEFAULT_GAP_SECONDS * 1000UL;
    Serial.println("Migrating the previous 8-second gap to the new 15-minute default.");
  }

  cfg.pumpDuty = clampValue<uint8_t>(cfg.pumpDuty, 0, 255);
  cfg.pumpMode = cfg.pumpMode == PUMP_MODE_BURST_FULL_POWER ? PUMP_MODE_BURST_FULL_POWER : PUMP_MODE_PWM_AFTER_KICK;
  cfg.burstPercent = clampValue<uint8_t>(cfg.burstPercent, 0, 100);
  cfg.pumpDuty = PUMP_FULL_DUTY;
  cfg.pumpMode = PUMP_MODE_PWM_AFTER_KICK;
  cfg.burstPercent = DEFAULT_BURST_PERCENT;
  cfg.countdownMs = maxValue<uint32_t>(cfg.countdownMs, 5000UL);
  cfg.topoffMs = maxValue<uint32_t>(cfg.topoffMs, 1000UL);
  cfg.holdZeroMs = maxValue<uint32_t>(cfg.holdZeroMs, 1000UL);
  cfg.drainMs = maxValue<uint32_t>(cfg.drainMs, 5000UL);
  cfg.gapMs = maxValue<uint32_t>(cfg.gapMs, MIN_GAP_MS);
  cfg.pumpVolumePercent = clampValue<uint16_t>(cfg.pumpVolumePercent,
                                               MIN_PUMP_VOLUME_PERCENT,
                                               MAX_PUMP_VOLUME_PERCENT);

  activePumpVolumePercent = cfg.pumpVolumePercent;
  recalculateCycleTiming();

  if (!useSavedConfig) {
    Serial.println("Applying new default exhibition timing.");
  }
  if (savedVersion != CONFIG_VERSION) {
    saveRuntimeConfig();
  }
}

void writePumpDuty(uint8_t duty) {
  writePwm(PUMP_PWM_PIN, PUMP_PWM_CHANNEL, duty);
}

void setPumpLowPin() {
  if (PUMP_LOW_PIN >= 0) {
    digitalWrite(PUMP_LOW_PIN, LOW);
  }
}

void setDriverAwake(bool awake) {
  if (DRV_SLEEP_PIN >= 0) {
    digitalWrite(DRV_SLEEP_PIN, awake ? HIGH : LOW);
  }
}

void pumpOff() {
  pumpRequested = false;
  pumpKickActive = false;
  writePumpDuty(0);
  setPumpLowPin();
}

void pumpOnWithKick() {
  pumpRequested = true;
  pumpKickActive = false;
  pumpKickStartedMs = millis();
  burstCycleStartedMs = pumpKickStartedMs;
  setDriverAwake(true);
  setPumpLowPin();
  writePumpDuty(PUMP_FULL_DUTY);
}

void updatePump() {
  if (!pumpRequested) {
    writePumpDuty(0);
    return;
  }

  writePumpDuty(PUMP_FULL_DUTY);
}

PumpScheduleState pumpScheduleState() {
  PumpScheduleState state;
  const time_t now = time(nullptr);
  if (now < MIN_VALID_UNIX_TIME) {
    return state;
  }

  struct tm localTime;
  if (localtime_r(&now, &localTime) == nullptr) {
    return state;
  }

  state.timeKnown = true;
  state.hour = static_cast<uint8_t>(localTime.tm_hour);
  state.minute = static_cast<uint8_t>(localTime.tm_min);
  state.windowOpen =
      state.hour >= PUMP_WINDOW_START_HOUR && state.hour < PUMP_WINDOW_END_HOUR;
  return state;
}

const char *scheduleStatusText(const PumpScheduleState &state) {
  if (!state.timeKnown) {
    return "time_unknown_allow";
  }
  return state.windowOpen ? "open" : "closed";
}

void formatLocalTime(const PumpScheduleState &state, char *buffer, size_t size) {
  if (!state.timeKnown) {
    snprintf(buffer, size, "unknown");
    return;
  }
  snprintf(buffer, size, "%02u:%02u", state.hour, state.minute);
}

bool pumpShouldRunNow() {
  if (!pumpEnabled || runMode != RunMode::Exhibition) {
    return false;
  }
  if (!pumpScheduleState().windowOpen) {
    return false;
  }

  if (phase == Phase::CountdownToFlood) {
    return phaseProgress() >= countdownPumpStartProgress() &&
           phaseProgress() < countdownPumpStopProgress();
  }
  if (phase == Phase::FinalRise) {
    return false;
  }
  if (phase == Phase::HoldAtZero) {
    return false;
  }
  return false;
}

void syncPumpForCurrentPhase() {
  const bool shouldRun = pumpShouldRunNow();
  if (shouldRun && !pumpRequested) {
    pumpOnWithKick();
  } else if (!shouldRun && pumpRequested) {
    pumpOff();
  }

  updatePump();
}

void enterPhase(Phase nextPhase) {
  phase = nextPhase;
  phaseStartedMs = millis();
  syncPumpForCurrentPhase();
}

float simulatedTimeToFloodHours() {
  if (phase == Phase::CountdownToFlood) {
    return FLOOD_DISPLAY_MAX_HOURS * (1.0f - phaseProgress());
  }
  if (phase == Phase::FinalRise || phase == Phase::HoldAtZero) {
    return 0.0f;
  }
  if (phase == Phase::Draining) {
    return FLOOD_DISPLAY_MAX_HOURS * phaseProgress();
  }
  return FLOOD_DISPLAY_MAX_HOURS;
}

float simulatedWaterModelPercent() {
  const float p = phaseProgress();
  if (phase == Phase::CountdownToFlood) {
    return countdownWaterTargetPercent() * activeCountdownPumpProgress();
  }
  if (phase == Phase::FinalRise) {
    const float start = countdownWaterTargetPercent();
    return start + (topoffWaterTargetPercent() - start) * p;
  }
  if (phase == Phase::HoldAtZero) {
    const float start = topoffWaterTargetPercent();
    return start + (holdWaterTargetPercent() - start) * p;
  }
  if (phase == Phase::Draining) {
    const float start = holdWaterTargetPercent();
    return start + (drainEndWaterTargetPercent() - start) * p;
  }
  if (phase == Phase::Gap && currentRisk == RiskLevel::High) {
    return HIGH_RISK_EXTRA_DRAIN_START_PERCENT * (1.0f - p);
  }
  return 0.0f;
}

float simulatedWaterPercent() {
  return clampValue(simulatedWaterModelPercent(), 0.0f, 100.0f);
}

float simulatedExcessWaterPercent() {
  const float water = simulatedWaterModelPercent();
  if (water <= 100.0f) {
    return 0.0f;
  }
  return water - 100.0f;
}

int simulatedIntensity() {
  if (phase == Phase::Paused || phase == Phase::Gap) {
    return 0;
  }

  float base = static_cast<float>(cycleIntensity);

  if (phase == Phase::Draining) {
    base *= 1.0f - 0.35f * phaseProgress();
  }

  if (cfg.randomiseIntensity) {
    base += static_cast<float>(random(-2, 3));
  }

  return clampValue<int>(static_cast<int>(roundf(base)), 0, 100);
}

const char *wifiStatusText() {
  switch (WiFi.status()) {
    case WL_CONNECTED:
      return "ok";
    case WL_NO_SSID_AVAIL:
      return "no_ssid";
    case WL_CONNECT_FAILED:
      return "failed";
    case WL_CONNECTION_LOST:
      return "lost";
    case WL_IDLE_STATUS:
      return "idle";
    case WL_DISCONNECTED:
      return "down";
    default:
      return "down";
  }
}

uint32_t mainCycleSeconds() {
  return (completeCycleMs() + 999UL) / 1000UL;
}

uint32_t currentPhaseRemainingMs() {
  const uint32_t durationMs = phaseDurationMs();
  if (durationMs == 0) return 0;

  const uint32_t elapsedMs = millis() - phaseStartedMs;
  return elapsedMs >= durationMs ? 0 : durationMs - elapsedMs;
}

uint32_t remainingCycleMs() {
  const uint32_t currentRemainingMs = currentPhaseRemainingMs();
  switch (phase) {
    case Phase::Gap:
      return currentRemainingMs;
    case Phase::CountdownToFlood:
      return currentRemainingMs + cycleTiming.topoffMs + cycleTiming.holdMs +
             cycleTiming.drainMs + cycleTiming.gapMs;
    case Phase::FinalRise:
      return currentRemainingMs + cycleTiming.holdMs + cycleTiming.drainMs +
             cycleTiming.gapMs;
    case Phase::HoldAtZero:
      return currentRemainingMs + cycleTiming.drainMs + cycleTiming.gapMs;
    case Phase::Draining:
      return currentRemainingMs + cycleTiming.gapMs;
    case Phase::Paused:
      return 0;
  }
  return 0;
}

uint32_t remainingCycleSeconds() {
  const uint32_t remainingMs = remainingCycleMs();
  return remainingMs == 0 ? 0 : (remainingMs + 999UL) / 1000UL;
}

void connectWiFiIfNeeded() {
  if (WiFi.status() == WL_CONNECTED) return;

  const uint32_t now = millis();
  if (now - lastWifiAttemptMs < 5000UL) return;

  lastWifiAttemptMs = now;
  Serial.print("WiFi connecting to ");
  Serial.println(WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

String mqttClientId() {
  String id = "housemill-diorama-";
  id += String(static_cast<uint32_t>(ESP.getEfuseMac()), HEX);
  return id;
}

bool commandTargetsDiorama(JsonDocument &doc) {
  if (!doc["target"].is<const char *>()) {
    return true;
  }

  const String target = doc["target"].as<String>();
  return target == "all" || target == "diorama";
}

void publishControlState() {
  if (!mqtt.connected()) return;

  const PumpScheduleState schedule = pumpScheduleState();
  char localTimeText[8];
  formatLocalTime(schedule, localTimeText, sizeof(localTimeText));

  StaticJsonDocument<512> doc;
  doc["device"] = "diorama";
  doc["online"] = true;
  doc["mode"] = runModeName(runMode);
  doc["pump_enabled"] = pumpEnabled;
  doc["pump_running"] = pumpRequested;
  doc["pump_duty"] = pumpRequested ? PUMP_FULL_DUTY : 0;
  doc["pump_volume_percent"] = cfg.pumpVolumePercent;
  doc["active_pump_volume_percent"] = activePumpVolumePercent;
  doc["complete_cycle_seconds"] = mainCycleSeconds();
  doc["remaining_cycle_seconds"] = remainingCycleSeconds();
  doc["interval_seconds"] = cycleTiming.gapMs / 1000UL;
  doc["minimum_interval_seconds"] = MIN_GAP_SECONDS;
  doc["schedule"] = scheduleStatusText(schedule);
  doc["schedule_time_known"] = schedule.timeKnown;
  doc["pump_window_open"] = schedule.windowOpen;
  doc["local_time"] = localTimeText;
  doc["volume_limited"] = cycleTiming.volumeLimited;
  doc["phase"] = phaseName(phase);
  doc["wifi"] = wifiStatusText();

  char buffer[512];
  const size_t len = serializeJson(doc, buffer);
  mqtt.publish(MQTT_TOPIC_CONTROL_STATUS,
               reinterpret_cast<const uint8_t *>(buffer), len, true);
}

void applyCommand(JsonDocument &doc) {
  if (!commandTargetsDiorama(doc)) {
    return;
  }

  bool changed = false;

  if (doc["pump_enabled"].is<bool>()) {
    pumpEnabled = doc["pump_enabled"].as<bool>();
    syncPumpForCurrentPhase();
    Serial.print("Remote pump master switch: ");
    Serial.println(pumpEnabled ? "enabled" : "disabled");
  }

  if (doc["pump_volume_percent"].is<int>()) {
    cfg.pumpVolumePercent = clampValue<int>(doc["pump_volume_percent"].as<int>(),
                                            MIN_PUMP_VOLUME_PERCENT,
                                            MAX_PUMP_VOLUME_PERCENT);
    changed = true;
    Serial.print("Next-cycle pump volume: ");
    Serial.print(cfg.pumpVolumePercent);
    Serial.println("%");
  }

  if (doc["mode"].is<const char *>()) {
    const String requestedMode = doc["mode"].as<String>();
    if (requestedMode == "exhibition") {
      runMode = RunMode::Exhibition;
      enterPhase(Phase::Gap);
    } else if (requestedMode == "simulation") {
      runMode = RunMode::Simulation;
      enterPhase(Phase::Paused);
    } else if (requestedMode == "pause" || requestedMode == "paused") {
      runMode = RunMode::Paused;
      enterPhase(Phase::Paused);
    }
    Serial.print("Run mode: ");
    Serial.println(runModeName(runMode));
  }

  if (doc["pump_duty"].is<int>()) {
    cfg.pumpDuty = clampValue<int>(doc["pump_duty"].as<int>(), 0, 255);
    changed = true;
  }
  if (doc["pump_mode"].is<const char *>()) {
    const String mode = doc["pump_mode"].as<String>();
    cfg.pumpMode = mode == "burst" ? PUMP_MODE_BURST_FULL_POWER : PUMP_MODE_PWM_AFTER_KICK;
    changed = true;
  }
  if (doc["burst_percent"].is<int>()) {
    cfg.burstPercent = clampValue<int>(doc["burst_percent"].as<int>(), 0, 100);
    changed = true;
  }
  if (doc["countdown_seconds"].is<int>()) {
    cfg.countdownMs = maxValue<int>(doc["countdown_seconds"].as<int>(), 5) * 1000UL;
    changed = true;
  }
  if (doc["topoff_seconds"].is<int>()) {
    cfg.topoffMs = maxValue<int>(doc["topoff_seconds"].as<int>(), 1) * 1000UL;
    changed = true;
  }
  if (doc["hold_seconds"].is<int>()) {
    cfg.holdZeroMs = maxValue<int>(doc["hold_seconds"].as<int>(), 1) * 1000UL;
    changed = true;
  }
  if (doc["drain_seconds"].is<int>()) {
    cfg.drainMs = maxValue<int>(doc["drain_seconds"].as<int>(), 5) * 1000UL;
    changed = true;
  }
  if (doc["interval_seconds"].is<int>() || doc["gap_seconds"].is<int>()) {
    const int requestedSeconds = doc["interval_seconds"].is<int>()
                                     ? doc["interval_seconds"].as<int>()
                                     : doc["gap_seconds"].as<int>();
    const uint32_t intervalSeconds =
        static_cast<uint32_t>(maxValue<int>(requestedSeconds, MIN_GAP_SECONDS));
    cfg.gapMs = intervalSeconds * 1000UL;
    cycleTiming.gapMs = cfg.gapMs;
    if (phase == Phase::Gap) {
      phaseStartedMs = millis();
    }
    changed = true;
    Serial.print("Inter-cycle interval: ");
    Serial.print(intervalSeconds);
    Serial.println("s");
  }
  if (doc["hold_pump"].is<bool>()) {
    cfg.holdPumpDuringZero = doc["hold_pump"].as<bool>();
    changed = true;
  }
  if (doc["random_intensity"].is<bool>()) {
    cfg.randomiseIntensity = doc["random_intensity"].as<bool>();
    changed = true;
  }

  cfg.pumpDuty = PUMP_FULL_DUTY;
  cfg.pumpMode = PUMP_MODE_PWM_AFTER_KICK;
  cfg.burstPercent = DEFAULT_BURST_PERCENT;

  if (doc["risk"].is<const char *>()) {
    const String risk = doc["risk"].as<String>();
    if (risk == "low") {
      chooseRiskScenario(RiskLevel::Low);
    } else if (risk == "medium") {
      chooseRiskScenario(RiskLevel::Medium);
    } else if (risk == "high") {
      chooseRiskScenario(RiskLevel::High);
    } else if (risk == "random") {
      chooseRandomRiskScenario();
    }
  }

  if (doc["paused"].is<bool>()) {
    if (doc["paused"].as<bool>()) {
      runMode = RunMode::Paused;
      enterPhase(Phase::Paused);
    } else {
      runMode = RunMode::Exhibition;
      enterPhase(Phase::Gap);
    }
  }
  if (doc["reset_cycle"].is<bool>() && doc["reset_cycle"].as<bool>()) {
    activePumpVolumePercent = cfg.pumpVolumePercent;
    recalculateCycleTiming();
    enterPhase(Phase::Gap);
  }

  if (changed) {
    saveRuntimeConfig();
    Serial.println("Runtime config saved.");
  }
}

void mqttCallback(char *topic, uint8_t *payload, unsigned int length) {
  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print("Command JSON error: ");
    Serial.println(err.c_str());
    return;
  }

  Serial.print("Command on ");
  Serial.println(topic);
  applyCommand(doc);
  publishControlState();
}

void connectMqttIfNeeded() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;

  const uint32_t now = millis();
  if (now - lastMqttAttemptMs < 5000UL) return;
  lastMqttAttemptMs = now;

  Serial.print("MQTT connecting to ");
  Serial.println(MQTT_HOST);

  const String clientId = mqttClientId();
  bool ok = strlen(MQTT_USER) > 0
                ? mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD,
                               MQTT_TOPIC_CONTROL_STATUS, 0, true,
                               MQTT_CONTROL_OFFLINE_PAYLOAD)
                : mqtt.connect(clientId.c_str(), MQTT_TOPIC_CONTROL_STATUS, 0,
                               true, MQTT_CONTROL_OFFLINE_PAYLOAD);

  if (ok) {
    Serial.println("MQTT connected.");
    mqtt.subscribe(MQTT_TOPIC_COMMAND);
    publishControlState();
  } else {
    Serial.print("MQTT failed, rc=");
    Serial.println(mqtt.state());
  }
}

void publishStatus(bool force = false) {
  const uint32_t now = millis();
  if (!force && now - lastPublishMs < PUBLISH_INTERVAL_MS) return;
  lastPublishMs = now;

  const float etaHours = round1(simulatedTimeToFloodHours());
  const int intensity = simulatedIntensity();
  const float waterPercent = round1(simulatedWaterPercent());
  const float virtualWaterPercent = round1(simulatedWaterModelPercent());
  const char *phaseText = phaseName(phase);
  const char *riskText = riskName(currentRisk);
  const char *wifiText = wifiStatusText();
  const PumpScheduleState schedule = pumpScheduleState();
  char localTimeText[8];
  formatLocalTime(schedule, localTimeText, sizeof(localTimeText));

  StaticJsonDocument<512> doc;
  doc["mode"] = runModeName(runMode);
  doc["phase"] = phaseText;
  doc["risk_level"] = riskText;
  doc["time_to_flood"] = etaHours;
  doc["intensity"] = intensity;
  doc["water_level_percent"] = waterPercent;
  doc["virtual_water_percent"] = virtualWaterPercent;
  doc["pump_running"] = pumpRequested;
  doc["pump_volume_percent"] = activePumpVolumePercent;
  if (cfg.pumpVolumePercent != activePumpVolumePercent) {
    doc["next_pump_volume_percent"] = cfg.pumpVolumePercent;
  }
  doc["interval_seconds"] = cycleTiming.gapMs / 1000UL;
  doc["complete_cycle_seconds"] = mainCycleSeconds();
  doc["remaining_cycle_seconds"] = remainingCycleSeconds();
  doc["wifi"] = wifiText;
  doc["schedule"] = scheduleStatusText(schedule);
  doc["schedule_time_known"] = schedule.timeKnown;
  doc["pump_window_open"] = schedule.windowOpen;
  doc["local_time"] = localTimeText;
  doc["cycle"] = cycleNumber;
  doc["seq"] = sequenceNumber++;

  char buffer[512];
  const size_t len = serializeJson(doc, buffer);

  if (mqtt.connected() && runMode == RunMode::Exhibition) {
    mqtt.publish(MQTT_TOPIC_STATUS, reinterpret_cast<const uint8_t *>(buffer), len, MQTT_RETAIN_STATUS);
  }

  if (force || now - lastSerialStatusMs >= SERIAL_STATUS_INTERVAL_MS) {
    lastSerialStatusMs = now;
    Serial.print("Status phase=");
    Serial.print(phaseText);
    Serial.print(" risk=");
    Serial.print(riskText);
    Serial.print(" eta=");
    Serial.print(etaHours, 1);
    Serial.print("h intensity=");
    Serial.print(intensity);
    Serial.print(" virtual_water=");
    Serial.print(virtualWaterPercent, 1);
    Serial.print("% pump=");
    Serial.print(pumpRequested ? "on" : "off");
    Serial.print(" duty=");
    Serial.print(pumpRequested ? PUMP_FULL_DUTY : 0);
    Serial.print(" wifi=");
    Serial.print(wifiText);
    Serial.print(" schedule=");
    Serial.print(scheduleStatusText(schedule));
    Serial.print(" local_time=");
    Serial.print(localTimeText);
    Serial.print(" mode=");
    Serial.print(runModeName(runMode));
    Serial.print(" pump_enabled=");
    Serial.print(pumpEnabled ? "yes" : "no");
    Serial.print(" volume=");
    Serial.print(activePumpVolumePercent);
    Serial.print("% total=");
    Serial.print(mainCycleSeconds());
    Serial.print("s remaining=");
    Serial.print(remainingCycleSeconds());
    Serial.print("s");
    Serial.print(" cycle=");
    Serial.println(cycleNumber);
  }
}

void printDebugJson() {
  StaticJsonDocument<768> doc;
  doc["mode"] = runModeName(runMode);
  doc["phase"] = phaseName(phase);
  doc["risk_level"] = riskName(currentRisk);
  doc["time_to_flood"] = round1(simulatedTimeToFloodHours());
  doc["intensity"] = simulatedIntensity();
  doc["water_level_percent"] = round1(simulatedWaterPercent());
  doc["water_model_percent"] = round1(simulatedWaterModelPercent());
  doc["excess_water_percent"] = round1(simulatedExcessWaterPercent());
  doc["pump_running"] = pumpRequested;
  doc["pump_enabled"] = pumpEnabled;
  doc["pump_duty"] = pumpRequested ? PUMP_FULL_DUTY : 0;
  doc["pump_mode"] = "full";
  doc["pump_volume_percent"] = activePumpVolumePercent;
  doc["next_pump_volume_percent"] = cfg.pumpVolumePercent;
  doc["pump_duration_seconds"] = cycleTiming.pumpDurationMs / 1000UL;
  doc["countdown_seconds"] = cycleTiming.countdownMs / 1000UL;
  doc["hold_seconds"] = cycleTiming.holdMs / 1000UL;
  doc["gap_seconds"] = cycleTiming.gapMs / 1000UL;
  doc["minimum_gap_seconds"] = MIN_GAP_SECONDS;
  doc["complete_cycle_seconds"] = mainCycleSeconds();
  doc["remaining_cycle_seconds"] = remainingCycleSeconds();
  doc["volume_limited"] = cycleTiming.volumeLimited;
  doc["wifi"] = wifiStatusText();
  if (WiFi.status() == WL_CONNECTED) {
    doc["wifi_rssi"] = WiFi.RSSI();
  }
  doc["risk_burst_percent"] = riskBurstPercent();
  doc["topoff_seconds"] = effectiveTopoffMs() / 1000UL;
  doc["drain_seconds"] = effectiveDrainMs() / 1000UL;
  doc["phase_progress"] = round1(phaseProgress() * 100.0f);
  doc["cycle"] = cycleNumber;
  doc["uptime_ms"] = millis();

  Serial.print("Debug ");
  serializeJson(doc, Serial);
  Serial.println();
}

void updateCycle() {
  if (runMode != RunMode::Exhibition || phase == Phase::Paused) {
    pumpOff();
    return;
  }

  const PumpScheduleState schedule = pumpScheduleState();
  if (schedule.timeKnown && !schedule.windowOpen) {
    if (phase != Phase::Gap) {
      enterPhase(Phase::Gap);
    }
    pumpOff();
    return;
  }

  const uint32_t duration = phaseDurationMs();
  if (duration == 0 || millis() - phaseStartedMs < duration) {
    syncPumpForCurrentPhase();
    return;
  }

  if (phase == Phase::Gap) {
    chooseRandomRiskScenario();
    enterPhase(Phase::CountdownToFlood);
  } else if (phase == Phase::CountdownToFlood) {
    enterPhase(Phase::FinalRise);
  } else if (phase == Phase::FinalRise) {
    enterPhase(Phase::HoldAtZero);
  } else if (phase == Phase::HoldAtZero) {
    enterPhase(Phase::Draining);
  } else if (phase == Phase::Draining) {
    enterPhase(Phase::Gap);
  }

  syncPumpForCurrentPhase();
}

void printHelp() {
  Serial.println();
  Serial.println("House Mill diorama exhibition commands:");
  Serial.println("  Pump output is locked at PWM 255 when running; tune water with timing.");
  Serial.println("  volume 50..150     Pump-water percentage; applies from next cycle");
  Serial.println("  countdown 110      Pumping countdown: meter 24h -> 0h, pump stops at 0h");
  Serial.println("  topoff 1           Short 0h visual transition, pump stays off");
  Serial.println("  hold 8             Seconds to keep meter at 0h");
  Serial.println("  drain 25           Base seconds for water to drain while meter returns to 24h");
  Serial.println("  gap 900            Delay between cycles; minimum 8 seconds");
  Serial.println("  random on/off      Toggle random intensity jitter");
  Serial.println("  risk random        Choose a new random risk scenario now");
  Serial.println("  risk low/medium/high  Force one risk scenario for testing");
  Serial.println("  pause / resume     Stop or restart");
  Serial.println("  status             Print one short status line");
  Serial.println("  json               Print full debug JSON once");
  Serial.println();
}

void handleSerialLine(String line) {
  line.trim();
  line.toLowerCase();
  if (line.length() == 0) return;

  if (line == "help") {
    printHelp();
  } else if (line == "status") {
    publishStatus(true);
  } else if (line == "json") {
    printDebugJson();
  } else if (line == "pause") {
    runMode = RunMode::Paused;
    enterPhase(Phase::Paused);
    publishStatus(true);
  } else if (line == "resume") {
    runMode = RunMode::Exhibition;
    enterPhase(Phase::Gap);
    publishStatus(true);
  } else if (line.startsWith("speed ")) {
    cfg.pumpDuty = PUMP_FULL_DUTY;
    saveRuntimeConfig();
    Serial.println("Pump output is locked at PWM 255 while running.");
  } else if (line.startsWith("flow ")) {
    cfg.burstPercent = DEFAULT_BURST_PERCENT;
    saveRuntimeConfig();
    Serial.println("Pump flow command ignored; output is locked at PWM 255 while running.");
  } else if (line.startsWith("pumpmode ")) {
    cfg.pumpMode = PUMP_MODE_PWM_AFTER_KICK;
    saveRuntimeConfig();
    Serial.println("Pump mode is locked to full-power while running.");
  } else if (line.startsWith("volume ")) {
    cfg.pumpVolumePercent = clampValue<int>(line.substring(7).toInt(),
                                            MIN_PUMP_VOLUME_PERCENT,
                                            MAX_PUMP_VOLUME_PERCENT);
    saveRuntimeConfig();
    Serial.print("Next-cycle pump volume set to ");
    Serial.print(cfg.pumpVolumePercent);
    Serial.println("%. Use a risk command or reset the cycle to apply now.");
  } else if (line.startsWith("countdown ")) {
    cfg.countdownMs = maxValue<int>(line.substring(10).toInt(), 5) * 1000UL;
    saveRuntimeConfig();
  } else if (line.startsWith("topoff ")) {
    cfg.topoffMs = maxValue<int>(line.substring(7).toInt(), 1) * 1000UL;
    saveRuntimeConfig();
  } else if (line.startsWith("hold ")) {
    cfg.holdZeroMs = maxValue<int>(line.substring(5).toInt(), 1) * 1000UL;
    saveRuntimeConfig();
  } else if (line.startsWith("drain ")) {
    cfg.drainMs = maxValue<int>(line.substring(6).toInt(), 5) * 1000UL;
    saveRuntimeConfig();
  } else if (line.startsWith("gap ")) {
    const uint32_t intervalSeconds = static_cast<uint32_t>(
        maxValue<int>(line.substring(4).toInt(), MIN_GAP_SECONDS));
    cfg.gapMs = intervalSeconds * 1000UL;
    cycleTiming.gapMs = cfg.gapMs;
    if (phase == Phase::Gap) {
      phaseStartedMs = millis();
    }
    saveRuntimeConfig();
  } else if (line.startsWith("holdpump ")) {
    cfg.holdPumpDuringZero = line.endsWith("on");
    saveRuntimeConfig();
  } else if (line.startsWith("random ")) {
    cfg.randomiseIntensity = line.endsWith("on");
    saveRuntimeConfig();
  } else if (line.startsWith("risk ")) {
    const String risk = line.substring(5);
    if (risk == "low") {
      chooseRiskScenario(RiskLevel::Low);
    } else if (risk == "medium") {
      chooseRiskScenario(RiskLevel::Medium);
    } else if (risk == "high") {
      chooseRiskScenario(RiskLevel::High);
    } else {
      chooseRandomRiskScenario();
    }
  } else {
    Serial.println("Unknown command. Type: help");
    return;
  }

  publishStatus(true);
}

void handleSerial() {
  while (Serial.available() > 0) {
    const char c = static_cast<char>(Serial.read());
    if (c == '\n' || c == '\r') {
      handleSerialLine(serialLine);
      serialLine = "";
    } else {
      serialLine += c;
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  randomSeed(esp_random());
  loadRuntimeConfig();

  if (PUMP_LOW_PIN >= 0) {
    pinMode(PUMP_LOW_PIN, OUTPUT);
    digitalWrite(PUMP_LOW_PIN, LOW);
  }
  if (DRV_SLEEP_PIN >= 0) {
    pinMode(DRV_SLEEP_PIN, OUTPUT);
    digitalWrite(DRV_SLEEP_PIN, HIGH);
  }

  attachPwm(PUMP_PWM_PIN, PUMP_PWM_CHANNEL, PUMP_PWM_FREQ_HZ, PUMP_PWM_BITS);
  pumpOff();

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(512);

  connectWiFiIfNeeded();
  configTzTime(TIME_ZONE, NTP_SERVER_PRIMARY, NTP_SERVER_SECONDARY,
               NTP_SERVER_TERTIARY);
  enterPhase(Phase::Gap);
  printHelp();
}

void loop() {
  handleSerial();
  connectWiFiIfNeeded();
  connectMqttIfNeeded();

  if (mqtt.connected()) {
    mqtt.loop();
  }

  updateCycle();
  publishStatus();
}
