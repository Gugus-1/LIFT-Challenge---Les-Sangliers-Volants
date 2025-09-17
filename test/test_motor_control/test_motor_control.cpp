#include <unity.h>
#include <Arduino.h>
#include "motor_control.h"

// Copy of your AT matrix for testing:
static const int8_t AT_test[3][8] = {
  { 0,  1,  0,  1,  0, -1,  0, -1},
  { 0,  1,  0, -1,  0, -1,  0,  1},
  { 1,  0,  1,  0,  1,  0,  1,  0}
};

void setUp(void) {
  // nothing to do before each test
}

void tearDown(void) {
  // nothing to do after each test
}

// Helper to compute expected PWM
static uint8_t golden(int16_t raw) {
  if      (raw >  127) raw =  127;
  else if (raw < -128) raw = -128;
  return uint8_t(raw + 128);
}

void test_zero_RFcmd_neutral_PWM_on_all_motors(void) {
  uint8_t rf[3] = {0,0,0};
  uint8_t  cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    TEST_ASSERT_EQUAL_UINT8(128, cmd[i]);
  }
}

void test_pure_forward_rf0_AT0_neutral(void) {
  uint8_t rf[3] = {10,0,0};
  uint8_t  cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum = int16_t(rf[0]) * AT_test[0][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void test_pure_lateral_rf1_AT1_neutral(void) {
  uint8_t rf[3] = {0,20,0};
  uint8_t  cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum = int16_t(rf[1]) * AT_test[1][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void test_pure_elevation_rf2_AT2_neutral(void) {
  uint8_t rf[3] = {0,0,30};
  uint8_t  cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum = int16_t(rf[2]) * AT_test[2][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void test_combined_RFcmd_sum_all_three_neutral(void) {
  uint8_t rf[3] = {5,7,9};
  uint8_t  cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum =
      int16_t(rf[0]) * AT_test[0][i] +
      int16_t(rf[1]) * AT_test[1][i] +
      int16_t(rf[2]) * AT_test[2][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void runUnityTests(void) {
  UNITY_BEGIN();
  RUN_TEST(test_zero_RFcmd_neutral_PWM_on_all_motors);
  RUN_TEST(test_pure_forward_rf0_AT0_neutral);
  RUN_TEST(test_pure_lateral_rf1_AT1_neutral);
  RUN_TEST(test_pure_elevation_rf2_AT2_neutral);
  RUN_TEST(test_combined_RFcmd_sum_all_three_neutral);
  UNITY_END();
}

extern "C" void setup(void) {
  // give Serial/USB time to enumerate
  delay(2000);
  UNITY_BEGIN();
  RUN_TEST(test_zero_RFcmd_neutral_PWM_on_all_motors);
  RUN_TEST(test_pure_forward_rf0_AT0_neutral);
  RUN_TEST(test_pure_lateral_rf1_AT1_neutral);
  RUN_TEST(test_pure_elevation_rf2_AT2_neutral);
  RUN_TEST(test_combined_RFcmd_sum_all_three_neutral);
  UNITY_END();
}

extern "C" void loop(void) {
  // nothing—tests run in setup()
}
