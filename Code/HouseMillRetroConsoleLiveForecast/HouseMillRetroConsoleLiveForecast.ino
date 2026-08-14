#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <WiFi.h>
#include <time.h>

#if __has_include(<esp_arduino_version.h>)
#include <esp_arduino_version.h>
#endif

// ----------------------- User settings -----------------------

const char *WIFI_SSID = "UCL_IoT";
const char *WIFI_PASSWORD = "casace2026";

const char *MQTT_HOST = "mqtt.cetools.org";
const uint16_t MQTT_PORT = 1884;
const char *MQTT_USER = "student";
const char *MQTT_PASSWORD = "ce2021-mqtt-forget-whale";

const char *MQTT_TOPIC_FORECAST = "student/housemill/flood/forecast";

const float FLOOD_DISPLAY_MAX_HOURS = 24.0f;

// TN-73 VU meter and WS2812B wiring.
const int VU_PWM_PIN = 5;
const uint8_t VU_PWM_CHANNEL = 1;
const uint32_t VU_PWM_FREQ_HZ = 20000;
const uint8_t VU_PWM_BITS = 8;
const uint8_t VU_DUTY_AT_0_HOURS = 0;    // Left side of the meter.
const uint8_t VU_DUTY_AT_24_HOURS = 255; // Right side of the meter, 8 equal 3h divisions.
const float VU_SMOOTHING = 0.08f;

const int LED_DATA_PIN = 6;
const uint16_t LED_COUNT = 16;
const uint8_t LED_BRIGHTNESS = 90;
const uint32_t STALE_TIMEOUT_MS = 35UL * 60UL * 1000UL;
const float ZERO_HOURS_THRESHOLD = 0.05f;
const uint32_t ZERO_FLASH_DURATION_MS = 3000UL;
const uint32_t ZERO_FLASH_TOGGLE_MS = 250UL;
const bool LED_PROGRESS_REVERSED = false;
const time_t MIN_VALID_UNIX_TIME = 1704067200; // 2024-01-01 UTC.

// ----------------------- Runtime state -----------------------

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
Adafruit_NeoPixel pixels(LED_COUNT, LED_DATA_PIN, NEO_GRB + NEO_KHZ800);

uint32_t lastWifiAttemptMs = 0;
uint32_t lastMqttAttemptMs = 0;
uint32_t lastMessageMs = 0;
uint32_t etaReceivedMs = 0;
float currentVuDuty = VU_DUTY_AT_24_HOURS;
float targetVuDuty = VU_DUTY_AT_24_HOURS;
float lastTimeToFloodHours = FLOOD_DISPLAY_MAX_HOURS;
String lastRiskLevel = "no risk";
String lastValidUntilUtc = "";
bool floodEtaAvailable = false;
bool zeroHoursActive = false;
uint32_t zeroFlashStartedMs = 0;

template <typename T>
T clampValue(T value, T low, T high) {
  if (value < low) return low;
  if (value > high) return high;
  return value;
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

String mqttClientId() {
  String id = "housemill-live-console-";
  id += String(static_cast<uint32_t>(ESP.getEfuseMac()), HEX);
  return id;
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

float vuDutyForHours(float hours) {
  hours = clampValue(hours, 0.0f, FLOOD_DISPLAY_MAX_HOURS);
  const float scale = hours / FLOOD_DISPLAY_MAX_HOURS;
  return static_cast<float>(VU_DUTY_AT_0_HOURS) +
         scale * static_cast<float>(static_cast<int16_t>(VU_DUTY_AT_24_HOURS) -
                                    static_cast<int16_t>(VU_DUTY_AT_0_HOURS));
}

void setVuTargetFromHours(float hours) {
  targetVuDuty = vuDutyForHours(hours);
}

uint32_t rgb(uint8_t red, uint8_t green, uint8_t blue) {
  return pixels.Color(red, green, blue);
}

void fillPixels(uint32_t color) {
  for (uint16_t i = 0; i < LED_COUNT; i++) {
    pixels.setPixelColor(i, color);
  }
}

const char *riskName(int riskLevel) {
  if (riskLevel == 1) return "caution";
  if (riskLevel == 2) return "warning";
  if (riskLevel == 3) return "severe";
  return "no risk";
}

uint32_t colorForRisk() {
  if (lastRiskLevel == "caution") return rgb(0, 76, 255);
  if (lastRiskLevel == "warning") return rgb(255, 180, 0);
  if (lastRiskLevel == "severe") return rgb(255, 0, 0);
  return 0;
}

bool clockReady() {
  return time(nullptr) >= MIN_VALID_UNIX_TIME;
}

bool validUntilPassed() {
  if (!clockReady() || lastValidUntilUtc.length() < 19) return false;

  const time_t now = time(nullptr);
  struct tm utcNow;
  gmtime_r(&now, &utcNow);

  char utcText[20];
  strftime(utcText, sizeof(utcText), "%Y-%m-%dT%H:%M:%S", &utcNow);
  return strncmp(utcText, lastValidUntilUtc.c_str(), 19) > 0;
}

bool forecastIsStale() {
  if (lastMessageMs == 0) return true;
  if (millis() - lastMessageMs > STALE_TIMEOUT_MS) return true;
  return validUntilPassed();
}

float currentTimeToFloodHours() {
  if (!floodEtaAvailable) return FLOOD_DISPLAY_MAX_HOURS;
  const uint32_t elapsedMinutes = (millis() - etaReceivedMs) / 60000UL;
  const float elapsedHours = elapsedMinutes / 60.0f;
  return clampValue(lastTimeToFloodHours - elapsedHours,
                    0.0f, FLOOD_DISPLAY_MAX_HOURS);
}

void showTimeProgress(uint32_t color) {
  const float progress = clampValue(currentTimeToFloodHours() / FLOOD_DISPLAY_MAX_HOURS,
                                    0.0f, 1.0f);
  const uint16_t litCount = clampValue<uint16_t>(
      static_cast<uint16_t>(ceilf(progress * LED_COUNT)), 0, LED_COUNT);

  pixels.clear();
  for (uint16_t i = 0; i < litCount; i++) {
    const uint16_t pixelIndex = LED_PROGRESS_REVERSED ? LED_COUNT - 1 - i : i;
    pixels.setPixelColor(pixelIndex, color);
  }
}

void updateLeds() {
  if (forecastIsStale()) {
    pixels.clear();
    pixels.setPixelColor((millis() / 180UL) % LED_COUNT, rgb(90, 55, 0));
    pixels.setBrightness(35);
    pixels.show();
    return;
  }

  if (lastRiskLevel == "no risk") {
    pixels.clear();
    pixels.show();
    return;
  }

  const bool reachedZero = floodEtaAvailable &&
                           currentTimeToFloodHours() <= ZERO_HOURS_THRESHOLD;
  if (reachedZero && !zeroHoursActive) {
    zeroFlashStartedMs = millis();
  }
  zeroHoursActive = reachedZero;

  const uint32_t riskColor = colorForRisk();
  if (zeroHoursActive) {
    const uint32_t elapsedMs = millis() - zeroFlashStartedMs;
    const bool flashOn = elapsedMs < ZERO_FLASH_DURATION_MS &&
                         ((elapsedMs / ZERO_FLASH_TOGGLE_MS) % 2UL == 1UL);
    fillPixels(flashOn ? riskColor : 0);
    pixels.setBrightness(LED_BRIGHTNESS);
    pixels.show();
    return;
  }

  showTimeProgress(riskColor);
  pixels.setBrightness(LED_BRIGHTNESS);
  pixels.show();
}

void updateVu() {
  targetVuDuty = vuDutyForHours(currentTimeToFloodHours());
  const float desired = forecastIsStale()
                            ? vuDutyForHours(FLOOD_DISPLAY_MAX_HOURS)
                            : targetVuDuty;
  currentVuDuty += (desired - currentVuDuty) * VU_SMOOTHING;
  writePwm(VU_PWM_PIN, VU_PWM_CHANNEL,
           clampValue<int>(static_cast<int>(roundf(currentVuDuty)), 0, 255));
}

void handleForecastPayload(uint8_t *payload, unsigned int length) {
  StaticJsonDocument<256> filter;
  filter["status"] = true;
  filter["valid_until_utc"] = true;
  filter["risk_level"] = true;
  filter["eta_minutes"] = true;
  filter["max_predicted_m"] = true;

  StaticJsonDocument<512> doc;
  DeserializationError err = deserializeJson(
      doc, payload, length, DeserializationOption::Filter(filter));
  if (err) {
    Serial.print("Forecast JSON error: ");
    Serial.println(err.c_str());
    return;
  }

  const char *status = doc["status"] | "";
  if (strcmp(status, "ok") != 0) {
    Serial.println("Forecast status is not ok.");
    return;
  }

  lastValidUntilUtc = doc["valid_until_utc"] | "";
  if (validUntilPassed()) {
    lastMessageMs = 0;
    Serial.println("Forecast has expired.");
    return;
  }

  const int riskLevel = clampValue<int>(doc["risk_level"] | 0, 0, 3);
  const int etaMinutes = doc["eta_minutes"] | -1;
  const float nextTimeToFloodHours = etaMinutes >= 0
                                         ? etaMinutes / 60.0f
                                         : FLOOD_DISPLAY_MAX_HOURS;

  lastRiskLevel = riskName(riskLevel);
  floodEtaAvailable = etaMinutes >= 0;
  const bool reachedZero = etaMinutes >= 0 &&
                           nextTimeToFloodHours <= ZERO_HOURS_THRESHOLD;
  if (reachedZero && !zeroHoursActive) {
    zeroFlashStartedMs = millis();
  }
  zeroHoursActive = reachedZero;
  lastTimeToFloodHours = clampValue(nextTimeToFloodHours,
                                    0.0f, FLOOD_DISPLAY_MAX_HOURS);
  etaReceivedMs = millis();
  lastMessageMs = millis();
  setVuTargetFromHours(lastTimeToFloodHours);

  Serial.print("risk=");
  Serial.print(lastRiskLevel);
  Serial.print(" eta=");
  if (etaMinutes >= 0) {
    Serial.print(etaMinutes);
    Serial.print("min");
  } else {
    Serial.print("none");
  }
  Serial.print(" max_water=");
  Serial.print(doc["max_predicted_m"] | 0.0f, 3);
  Serial.println("m");
}

void mqttCallback(char *topic, uint8_t *payload, unsigned int length) {
  if (strcmp(topic, MQTT_TOPIC_FORECAST) == 0) {
    handleForecastPayload(payload, length);
  }
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
                ? mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD)
                : mqtt.connect(clientId.c_str());

  if (ok) {
    Serial.println("MQTT connected.");
    mqtt.subscribe(MQTT_TOPIC_FORECAST);
  } else {
    Serial.print("MQTT failed, rc=");
    Serial.println(mqtt.state());
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  attachPwm(VU_PWM_PIN, VU_PWM_CHANNEL, VU_PWM_FREQ_HZ, VU_PWM_BITS);
  writePwm(VU_PWM_PIN, VU_PWM_CHANNEL, static_cast<uint8_t>(currentVuDuty));

  pixels.begin();
  pixels.clear();
  pixels.setBrightness(LED_BRIGHTNESS);
  pixels.show();

  configTime(0, 0, "pool.ntp.org", "time.google.com");

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(16384);

  Serial.println();
  Serial.println("House Mill live forecast console");
  Serial.print("Subscribing to ");
  Serial.println(MQTT_TOPIC_FORECAST);

  connectWiFiIfNeeded();
}

void loop() {
  connectWiFiIfNeeded();
  connectMqttIfNeeded();

  if (mqtt.connected()) {
    mqtt.loop();
  }

  updateVu();
  updateLeds();
  delay(20);
}
