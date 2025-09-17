#ifndef SERIALBUSREADER_H
#define SERIALBUSREADER_H

#include <Arduino.h>

class SerialBusReader {
public:
  static constexpr size_t BUFFER_SIZE = 50;

  // user provides the Stream (e.g. Serial1)
  SerialBusReader(Stream &serial);

  // call after your Serial.begin(...)
  bool begin();

  // set callbacks (can be nullptr if you like)
  void setHalfBufferCallback(void (*cb)());
  void setFullBufferCallback(void (*cb)());

  // call this as often as you like (e.g. in loop() or from a serial ISR)
  void readIntoBuffer();

  // inspect the buffer if you want
  const uint8_t* buffer()    const { return _buf; }
  size_t           index()   const { return _idx; }

private:
  Stream&           _serial;
  uint8_t           _buf[BUFFER_SIZE];
  size_t            _idx;
  void            (*_halfCb)();
  void            (*_fullCb)();
};

#endif // SERIALBUSREADER_H
