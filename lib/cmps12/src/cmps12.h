// lib/cmps12/src/CMPS12.h
#pragma once

#include <Arduino.h>

class CMPS12 {
public:
  /**
   * @param serial  any Stream you’ve already opened (e.g. Serial1)
   */
  explicit CMPS12(Stream &serial);

  /**
   * Call once after you’ve done serial.begin(baud).
   * Returns true always (you can extend it to do a self‑test if you like).
   */
  bool begin();

  uint8_t  getVersion();                // cmd 0x11
  uint8_t  getBearing8();               // cmd 0x12
  uint16_t getBearing16();              // cmd 0x13
  int8_t   getPitch();                  // cmd 0x14
  int8_t   getRoll();                   // cmd 0x15
  void     getMagRaw(int16_t &x, int16_t &y, int16_t &z);     // cmd 0x19
  void     getAccelRaw(int16_t &x, int16_t &y, int16_t &z);   // cmd 0x20
  void     getGyroRaw(int16_t &x, int16_t &y, int16_t &z);    // cmd 0x21
  int16_t  getTemp();                   // cmd 0x22
  void     getAll(uint16_t &bearing, int8_t &pitch, int8_t &roll); // cmd 0x23
  uint8_t  getCalibrationState();       // cmd 0x24
  uint16_t getBoschBearing16();         // cmd 0x25
  int16_t  getPitch180();               // cmd 0x26

  bool     storeCalibrationProfile();   // seq 0xF0,0xF5,0xF6
  bool     deleteCalibrationProfile();  // seq 0xE0,0xE5,0xE2
  bool     setBaud(uint32_t baud);     // cmd 0xA0/A1 (19200/38400)

private:
  Stream & _serial;
  bool      sendCommand(uint8_t cmd, size_t respLen, uint8_t *buf, uint16_t timeout = 100);
};
