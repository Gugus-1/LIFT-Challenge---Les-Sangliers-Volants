#include <Arduino.h>
#include <motor_control.h>

// 1) Define your pins in a constexpr array
static constexpr uint8_t MOT_PIN[8] = {
  0,  // MOT_PIN_FTL
  1,  // MOT_PIN_FL
  2,  // MOT_PIN_FTR
  3,  // MOT_PIN_FR
  4,  // MOT_PIN_BTR
  5,  // MOT_PIN_BR
  6,  // MOT_PIN_BTL
  7   // MOT_PIN_BL
};

// 2) Default control law of the blimp
constexpr int8_t AT[3][8] = {
      { 0,  1, 0,  1,  0, -1,  0, -1},
      { 0,  1, 0, -1,  0, -1,  0,  1},
      { 1,  0, 1,  0,  1,  0,  1,  0}
    };

void MotorControl::begin() {
  // define the resolution of analogWrite number of bits
  analogWriteResolution(8);
  for (uint8_t i = 0; i < 8; ++i) {
    pinMode(MOT_PIN[i], OUTPUT);
  }
}


void MotorControl::applyMotorsCommand(const uint8_t cmd[8]) {
  // Loop over all motors to apply the individual command
  for (uint8_t i = 0; i < 8; ++i) {
    analogWrite(MOT_PIN[i], cmd[i]);
  }
}

void MotorControl::mapRFtoMotors(const uint8_t RFcmd[3], uint8_t cmd[8], int8_t AT[3][8])
{
  // Convert the RF telecom signal into a command for individual motors
  /* Works under the following asumptions:
    1) RFcmd is 3 uint8_t such that:
      RFcmd[0] - forward component of wanted deplacement
      RFcmd[1] - lateral component of wanted deplacement
      RFcmd[2] - elevation componennt of wanted deplacement
    2) The modelization chosen is:
    A.T @ RFcmd[3] = cmd[8] such that A.T = 
      0  1  0  1  0 -1  0 -1 
      0  1  0 -1  0 -1  0  1
      1  0  1  0  1  0  1  0 
    3) There will be a closed loop control using KALMAN to update A's coefficient given a mesured deplacement
  */

  for (uint8_t motor = 0; motor < 8; ++motor) {
    // dot‑product of rf[0..2] with AT[0..2][motor]
    int16_t sum = 0;
    for (uint8_t i = 0; i < 3; ++i) {
      sum += int16_t(RFcmd[i]) * AT[i][motor];
    }
    // clamp into [0, 255] to have an uint8
    if (sum >  127) sum =  127;
    if (sum < -128) sum = -128;
    cmd[motor] = uint8_t(sum+128);
  }
}

