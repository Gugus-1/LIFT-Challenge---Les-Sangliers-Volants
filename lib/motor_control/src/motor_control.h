#pragma once

#include <Arduino.h>

namespace MotorControl {

  /**
   * @brief  Initialize the motor‑control library.
   *         Must be called once from your sketch’s setup().
   */
  void begin();

  /**
   * @brief  Send 8 raw motor commands (0–255) to your pins
   */
  void applyMotorsCommand(const uint8_t cmd[8]);

  /**
   * @brief  Map a 3‑element RF command [fwd, lat, elev] into 8 motor outputs,
   *         using the 3×8 AT matrix you pass in.
   *
   * @param RFcmd   input array of size 3
   * @param out     output array of size 8 (0–255)
   * @param AT      a 3×8 matrix of int8_t coefficients
   */
  void mapRFtoMotors(const uint8_t RFcmd[3],
                     uint8_t       cmd[8],
                     int8_t  AT[3][8]);

}
