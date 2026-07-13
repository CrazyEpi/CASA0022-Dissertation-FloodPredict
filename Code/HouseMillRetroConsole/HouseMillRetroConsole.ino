#include <Arduino.h>
#include <Adafruit_NeoPixel.h>
#include <ArduinoJson.h>
#include <PubSubClient.h>
#include <WiFi.h>

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

const char *MQTT_TOPIC_STATUS = "student/housemill/flood/status";
const char *MQTT_TOPIC_COMMAND = "student/housemill/flood/command";
const char *MQTT_TOPIC_CONTROL_STATUS = "student/housemill/control/console/status";
const char *MQTT_CONTROL_OFFLINE_PAYLOAD = "{\"device\":\"console\",\"online\":false}";
const bool MQTT_RETAIN_STATUS = false;

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
const uint32_t STALE_TIMEOUT_MS = 10000;

// GPIO 7 must drive a logic-level MOSFET or relay module. Never connect the
// 12V TN-73 lamp directly to an ESP32 GPIO. Set to -1 to control NeoPixels only.
const int BACKLIGHT_CONTROL_PIN = 7;
const bool BACKLIGHT_ACTIVE_HIGH = true;

// ----------------------- Runtime state -----------------------

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
Adafruit_NeoPixel pixels(LED_COUNT, LED_DATA_PIN, NEO_GRB + NEO_KHZ800);

enum class ConsoleMode : uint8_t {
  Exhibition,
  Simulation,
  Paused
};

uint32_t lastWifiAttemptMs = 0;
uint32_t lastMqttAttemptMs = 0;
uint32_t lastMessageMs = 0;
float currentVuDuty = VU_DUTY_AT_24_HOURS;
float targetVuDuty = VU_DUTY_AT_24_HOURS;
float lastTimeToFloodHours = FLOOD_DISPLAY_MAX_HOURS;
int lastIntensity = 0;
String lastMode = "idle";
ConsoleMode consoleMode = ConsoleMode::Exhibition;
bool backlightEnabled = true;

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
  String id = "housemill-console-";
  id += String(static_cast<uint32_t>(ESP.getEfuseMac()), HEX);
  return id;
}

const char *consoleModeName(ConsoleMode value) {
  switch (value) {
    case ConsoleMode::Exhibition:
      return "exhibition";
    case ConsoleMode::Simulation:
      return "simulation";
    case ConsoleMode::Paused:
      return "pause";
  }
  return "pause";
}

void applyBacklightOutput() {
  if (BACKLIGHT_CONTROL_PIN >= 0) {
    const bool outputHigh = BACKLIGHT_ACTIVE_HIGH ? backlightEnabled : !backlightEnabled;
    digitalWrite(BACKLIGHT_CONTROL_PIN, outputHigh ? HIGH : LOW);
  }
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

uint8_t lerp8(uint8_t from, uint8_t to, uint8_t amount) {
  return from + ((static_cast<int16_t>(to) - static_cast<int16_t>(from)) * amount) / 255;
}

uint32_t rgb(uint8_t red, uint8_t green, uint8_t blue) {
  return pixels.Color(red, green, blue);
}

uint32_t blendRgb(uint8_t r1, uint8_t g1, uint8_t b1,
                  uint8_t r2, uint8_t g2, uint8_t b2,
                  uint8_t amount) {
  return rgb(lerp8(r1, r2, amount), lerp8(g1, g2, amount), lerp8(b1, b2, amount));
}

uint8_t pulseBrightness(uint8_t low, uint8_t high, uint16_t periodMs) {
  const float phase = (millis() % periodMs) / static_cast<float>(periodMs);
  const float wave = 0.5f + 0.5f * sinf(phase * TWO_PI);
  return low + static_cast<uint8_t>((high - low) * wave);
}

void fillPixels(uint32_t color) {
  for (uint16_t i = 0; i < LED_COUNT; i++) {
    pixels.setPixelColor(i, color);
  }
}

uint32_t colorForIntensity(int intensity) {
  intensity = clampValue(intensity, 0, 100);

  if (intensity <= 35) {
    return blendRgb(0, 76, 255, 0, 210, 120, map(intensity, 0, 35, 0, 255));
  }
  if (intensity <= 70) {
    return blendRgb(0, 210, 120, 255, 160, 0, map(intensity, 35, 70, 0, 255));
  }
  return blendRgb(255, 160, 0, 255, 0, 0, map(intensity, 70, 100, 0, 255));
}

void updateLeds() {
  if (!backlightEnabled || consoleMode == ConsoleMode::Paused) {
    pixels.clear();
    pixels.show();
    return;
  }

  const bool stale = lastMessageMs == 0 || millis() - lastMessageMs > STALE_TIMEOUT_MS;

  if (stale) {
    fillPixels(rgb(18, 12, 0));
    pixels.setPixelColor((millis() / 180UL) % LED_COUNT, rgb(90, 55, 0));
    pixels.setBrightness(35);
    pixels.show();
    return;
  }

  fillPixels(colorForIntensity(lastIntensity));

  if (lastIntensity >= 80) {
    pixels.setBrightness(pulseBrightness(45, LED_BRIGHTNESS, 850));
  } else {
    pixels.setBrightness(LED_BRIGHTNESS);
  }

  pixels.show();
}

void updateVu() {
  const bool stale = lastMessageMs == 0 || millis() - lastMessageMs > STALE_TIMEOUT_MS;
  const float desired = (consoleMode == ConsoleMode::Paused || stale)
                            ? vuDutyForHours(FLOOD_DISPLAY_MAX_HOURS)
                            : targetVuDuty;
  currentVuDuty += (desired - currentVuDuty) * VU_SMOOTHING;
  writePwm(VU_PWM_PIN, VU_PWM_CHANNEL, clampValue<int>(static_cast<int>(roundf(currentVuDuty)), 0, 255));
}

void handleStatusPayload(uint8_t *payload, unsigned int length) {
  StaticJsonDocument<640> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print("Status JSON error: ");
    Serial.println(err.c_str());
    return;
  }

  lastMode = doc["mode"] | "unknown";
  if ((consoleMode == ConsoleMode::Exhibition && lastMode != "exhibition") ||
      (consoleMode == ConsoleMode::Simulation && lastMode != "simulation") ||
      consoleMode == ConsoleMode::Paused) {
    return;
  }

  lastTimeToFloodHours = doc["time_to_flood"] | FLOOD_DISPLAY_MAX_HOURS;
  lastIntensity = clampValue<int>(doc["intensity"] | 0, 0, 100);
  lastMessageMs = millis();
  setVuTargetFromHours(lastTimeToFloodHours);

  Serial.print("mode=");
  Serial.print(lastMode);
  Serial.print(" phase=");
  Serial.print(doc["phase"] | "n/a");
  Serial.print(" time_to_flood=");
  Serial.print(lastTimeToFloodHours, 1);
  Serial.print("h intensity=");
  Serial.print(lastIntensity);
  Serial.print(" virtual_water=");
  Serial.print(doc["water_level_percent"] | 0.0);
  Serial.println("%");
}

bool commandTargetsConsole(JsonDocument &doc) {
  if (!doc["target"].is<const char *>()) {
    return true;
  }

  const String target = doc["target"].as<String>();
  return target == "all" || target == "console";
}

void publishControlState() {
  if (!mqtt.connected()) return;

  StaticJsonDocument<256> doc;
  doc["device"] = "console";
  doc["online"] = true;
  doc["mode"] = consoleModeName(consoleMode);
  doc["backlight_enabled"] = backlightEnabled;
  doc["backlight_control_pin"] = BACKLIGHT_CONTROL_PIN;
  doc["wifi"] = WiFi.status() == WL_CONNECTED ? "ok" : "down";

  char buffer[256];
  const size_t len = serializeJson(doc, buffer);
  mqtt.publish(MQTT_TOPIC_CONTROL_STATUS,
               reinterpret_cast<const uint8_t *>(buffer), len, true);
}

void handleControlPayload(uint8_t *payload, unsigned int length) {
  StaticJsonDocument<384> doc;
  DeserializationError err = deserializeJson(doc, payload, length);
  if (err) {
    Serial.print("Command JSON error: ");
    Serial.println(err.c_str());
    return;
  }
  if (!commandTargetsConsole(doc)) {
    return;
  }

  if (doc["backlight_enabled"].is<bool>()) {
    backlightEnabled = doc["backlight_enabled"].as<bool>();
    applyBacklightOutput();
  }

  if (doc["mode"].is<const char *>()) {
    const String requestedMode = doc["mode"].as<String>();
    if (requestedMode == "exhibition") {
      consoleMode = ConsoleMode::Exhibition;
    } else if (requestedMode == "simulation") {
      consoleMode = ConsoleMode::Simulation;
    } else if (requestedMode == "pause" || requestedMode == "paused") {
      consoleMode = ConsoleMode::Paused;
    }
    lastMessageMs = 0;
  }

  Serial.print("Control mode=");
  Serial.print(consoleModeName(consoleMode));
  Serial.print(" backlight=");
  Serial.println(backlightEnabled ? "on" : "off");
  publishControlState();
}

void mqttCallback(char *topic, uint8_t *payload, unsigned int length) {
  if (strcmp(topic, MQTT_TOPIC_COMMAND) == 0) {
    handleControlPayload(payload, length);
  } else if (strcmp(topic, MQTT_TOPIC_STATUS) == 0) {
    handleStatusPayload(payload, length);
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
                ? mqtt.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD,
                               MQTT_TOPIC_CONTROL_STATUS, 0, true,
                               MQTT_CONTROL_OFFLINE_PAYLOAD)
                : mqtt.connect(clientId.c_str(), MQTT_TOPIC_CONTROL_STATUS, 0,
                               true, MQTT_CONTROL_OFFLINE_PAYLOAD);

  if (ok) {
    Serial.println("MQTT connected.");
    mqtt.subscribe(MQTT_TOPIC_STATUS);
    mqtt.subscribe(MQTT_TOPIC_COMMAND);
    publishControlState();
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

  if (BACKLIGHT_CONTROL_PIN >= 0) {
    pinMode(BACKLIGHT_CONTROL_PIN, OUTPUT);
    applyBacklightOutput();
  }

  mqtt.setServer(MQTT_HOST, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  mqtt.setBufferSize(768);

  Serial.println();
  Serial.println("House Mill retro console");
  Serial.print("Subscribing to ");
  Serial.println(MQTT_TOPIC_STATUS);
  Serial.print("Control topic: ");
  Serial.println(MQTT_TOPIC_COMMAND);

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
