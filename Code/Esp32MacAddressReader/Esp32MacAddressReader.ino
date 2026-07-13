#include <Arduino.h>
#include <WiFi.h>

String efuseMacToString(uint64_t mac) {
  char buffer[18];
  const uint8_t bytes[6] = {
      static_cast<uint8_t>((mac >> 40) & 0xFF),
      static_cast<uint8_t>((mac >> 32) & 0xFF),
      static_cast<uint8_t>((mac >> 24) & 0xFF),
      static_cast<uint8_t>((mac >> 16) & 0xFF),
      static_cast<uint8_t>((mac >> 8) & 0xFF),
      static_cast<uint8_t>(mac & 0xFF),
  };
  snprintf(buffer, sizeof(buffer), "%02X:%02X:%02X:%02X:%02X:%02X",
           bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5]);
  return String(buffer);
}

String efuseMacRawHex(uint64_t mac) {
  char buffer[19];
  snprintf(buffer, sizeof(buffer), "0x%04X%08X",
           static_cast<uint16_t>((mac >> 32) & 0xFFFF),
           static_cast<uint32_t>(mac & 0xFFFFFFFF));
  return String(buffer);
}

void printMacAddresses() {
  WiFi.mode(WIFI_STA);
  delay(200);

  const uint64_t efuseMac = ESP.getEfuseMac();

  Serial.println();
  Serial.println("ESP32-S3 MAC address report");
  Serial.println("---------------------------");
  Serial.print("Chip model: ");
  Serial.println(ESP.getChipModel());
  Serial.print("Chip revision: ");
  Serial.println(ESP.getChipRevision());
  Serial.print("CPU frequency MHz: ");
  Serial.println(ESP.getCpuFreqMHz());
  Serial.print("Flash size bytes: ");
  Serial.println(ESP.getFlashChipSize());
  Serial.println();
  Serial.print("WiFi STA MAC: ");
  Serial.println(WiFi.macAddress());
  Serial.print("WiFi AP MAC:  ");
  Serial.println(WiFi.softAPmacAddress());
  Serial.print("eFuse MAC:    ");
  Serial.println(efuseMacToString(efuseMac));
  Serial.print("eFuse raw hex: ");
  Serial.println(efuseMacRawHex(efuseMac));
  Serial.println();
  Serial.println("Use the WiFi STA MAC for router reservations, allowlists, and device labels.");
  Serial.println("Press EN/RESET to print again.");
}

void setup() {
  Serial.begin(115200);
  delay(1200);
  printMacAddresses();
}

void loop() {
  delay(1000);
}
