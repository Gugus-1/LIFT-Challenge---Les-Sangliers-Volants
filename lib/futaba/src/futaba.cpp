#include "futaba.h"

SerialBusReader::SerialBusReader(Stream &serial)
  : _serial(serial), _idx(0), _halfCb(nullptr), _fullCb(nullptr)
{}

bool SerialBusReader::begin() {
  // assume the user already did serial.begin(baud, config) in their sketch
  return true;
}

void SerialBusReader::setHalfBufferCallback(void (*cb)()) {
  _halfCb = cb;
}

void SerialBusReader::setFullBufferCallback(void (*cb)()) {
  _fullCb = cb;
}

void SerialBusReader::readIntoBuffer() {
  while (_serial.available()) {
    _buf[_idx++] = _serial.read();

    if (_idx == BUFFER_SIZE/2) {
      if (_halfCb) _halfCb();
    }
    if (_idx >= BUFFER_SIZE) {
      if (_fullCb) _fullCb();
      _idx = 0;
    }
  }
}
