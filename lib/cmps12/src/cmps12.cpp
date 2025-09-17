// lib/cmps12/src/CMPS12.cpp
#include "CMPS12.h"

CMPS12::CMPS12(Stream &serial)
  : _serial(serial)
{}

bool CMPS12::begin() {
  // assume the user already did serial.begin(baud) in their sketch
  return true;
}

bool CMPS12::sendCommand(uint8_t cmd, size_t respLen, uint8_t *buf, uint16_t timeout) {
    // 1) send the single‐byte command
    _serial.write(cmd);

    // 2) wait for exactly respLen bytes (or timeout)
    uint32_t start = millis();
    size_t   idx   = 0;
    while (idx < respLen && (millis() - start) < timeout) {
        if (_serial.available()) {
            buf[idx++] = _serial.read();
        }
    }
    return (idx == respLen);
}

uint8_t CMPS12::getVersion() {
  uint8_t b = 0xFF;
  sendCommand(0x11, 1, &b);
  return b;
}

uint8_t CMPS12::getBearing8() {
  uint8_t b = 0xFF;
  sendCommand(0x12, 1, &b);
  return b;
}

uint16_t CMPS12::getBearing16() {
  uint8_t b[2] = {0,0};
  sendCommand(0x13, 2, b);
  return (uint16_t(b[0]) << 8) | b[1];
}

int8_t CMPS12::getPitch() {
  uint8_t b = 0;
  sendCommand(0x14, 1, &b);
  return int8_t(b);
}

int8_t CMPS12::getRoll() {
  uint8_t b = 0;
  sendCommand(0x15, 1, &b);
  return int8_t(b);
}

void CMPS12::getMagRaw(int16_t &x, int16_t &y, int16_t &z) {
  uint8_t b[6] = {0};
  sendCommand(0x19, 6, b);
  x = (int16_t(b[0]) << 8) | b[1];
  y = (int16_t(b[2]) << 8) | b[3];
  z = (int16_t(b[4]) << 8) | b[5];
}

void CMPS12::getAccelRaw(int16_t &x, int16_t &y, int16_t &z) {
  uint8_t b[6] = {0};
  sendCommand(0x20, 6, b);
  x = (int16_t(b[0]) << 8) | b[1];
  y = (int16_t(b[2]) << 8) | b[3];
  z = (int16_t(b[4]) << 8) | b[5];
}

void CMPS12::getGyroRaw(int16_t &x, int16_t &y, int16_t &z) {
  uint8_t b[6] = {0};
  sendCommand(0x21, 6, b);
  x = (int16_t(b[0]) << 8) | b[1];
  y = (int16_t(b[2]) << 8) | b[3];
  z = (int16_t(b[4]) << 8) | b[5];
}

int16_t CMPS12::getTemp() {
  uint8_t b[2] = {0};
  sendCommand(0x22, 2, b);
  return (int16_t(b[0]) << 8) | b[1];
}

void CMPS12::getAll(uint16_t &bearing, int8_t &pitch, int8_t &roll) {
  uint8_t b[4] = {0};
  sendCommand(0x23, 4, b);
  bearing = (uint16_t(b[0]) << 8) | b[1];
  pitch   = int8_t(b[2]);
  roll    = int8_t(b[3]);
}

uint8_t CMPS12::getCalibrationState() {
  uint8_t b = 0;
  sendCommand(0x24, 1, &b);
  return b;
}

uint16_t CMPS12::getBoschBearing16() {
  uint8_t b[2] = {0};
  sendCommand(0x25, 2, b);
  return (uint16_t(b[0]) << 8) | b[1];
}

int16_t CMPS12::getPitch180() {
  uint8_t b[2] = {0};
  sendCommand(0x26, 2, b);
  return (int16_t(b[0]) << 8) | b[1];
}

bool CMPS12::storeCalibrationProfile() {
  uint8_t ok = 0;
  if (!sendCommand(0xF0, 1, &ok) || ok != 0x55) return false;
  if (!sendCommand(0xF5, 1, &ok) || ok != 0x55) return false;
  if (!sendCommand(0xF6, 1, &ok) || ok != 0x55) return false;
  return true;
}

bool CMPS12::deleteCalibrationProfile() {
  uint8_t ok = 0;
  if (!sendCommand(0xE0, 1, &ok) || ok != 0x55) return false;
  if (!sendCommand(0xE5, 1, &ok) || ok != 0x55) return false;
  if (!sendCommand(0xE2, 1, &ok) || ok != 0x55) return false;
  return true;
}

bool CMPS12::setBaud(uint32_t baud) {
  uint8_t cmd;
  if      (baud == 19200) cmd = 0xA0;
  else if (baud == 38400) cmd = 0xA1;
  else return false;

  uint8_t ok = 0;
  if (!sendCommand(cmd, 1, &ok) || ok != 0x55) return false;

  // NOTE: you must now re‑open your serial in your sketch:
  // e.g. Serial1.begin(baud);
  return true;
}
