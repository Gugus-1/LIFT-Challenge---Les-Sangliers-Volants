// test/test_combined.cpp

#include <Arduino.h>
#include <unity.h>
#include "futaba.h"
#include "motor_control.h"

// -------- FakeSerial for SerialBusReader tests --------
class FakeSerial : public Stream {
public:
    static const int MAX_BUF = 256;
    uint8_t rx_buf[MAX_BUF];
    int rx_len;
    int rx_pos;

    FakeSerial() : rx_len(0), rx_pos(0) {}

    void addResponse(const uint8_t* data, int len) {
        rx_len = min(len, MAX_BUF);
        memcpy(rx_buf, data, rx_len);
        rx_pos = 0;
    }

    int available() override { return rx_pos < rx_len; }
    int read() override     { return available() ? rx_buf[rx_pos++] : -1; }
    int peek() override     { return available() ? rx_buf[rx_pos] : -1; }
    void flush() override   {}
    size_t write(uint8_t) override { return 1; }
};

// Callback counters for SerialBusReader
static int half_count;
static int full_count;

void onHalf() { half_count++; }
void onFull() { full_count++; }

// Reset before each SerialBusReader test
void setUp(void) {
    half_count = 0;
    full_count = 0;
}

// SerialBusReader Tests
void test_half_callback_triggered(void) {
    FakeSerial fake;
    SerialBusReader reader(fake);
    reader.begin();
    reader.setHalfBufferCallback(onHalf);
    reader.setFullBufferCallback(onFull);

    const int N = SerialBusReader::BUFFER_SIZE / 2;
    uint8_t data[N];
    for (int i = 0; i < N; i++) data[i] = i;
    fake.addResponse(data, N);

    reader.readIntoBuffer();

    TEST_ASSERT_EQUAL_INT(1, half_count);
    TEST_ASSERT_EQUAL_INT(0, full_count);
    TEST_ASSERT_EQUAL_UINT(SerialBusReader::BUFFER_SIZE/2, reader.index());
}

void test_full_callback_triggered_and_wraps(void) {
    FakeSerial fake;
    SerialBusReader reader(fake);
    reader.begin();
    reader.setHalfBufferCallback(onHalf);
    reader.setFullBufferCallback(onFull);

    const int N = SerialBusReader::BUFFER_SIZE;
    uint8_t data[N];
    for (int i = 0; i < N; i++) data[i] = i + 100;
    fake.addResponse(data, N);

    reader.readIntoBuffer();

    TEST_ASSERT_EQUAL_INT(1, half_count);
    TEST_ASSERT_EQUAL_INT(1, full_count);
    TEST_ASSERT_EQUAL_UINT(0, reader.index());
}

void test_partial_after_wrap(void) {
    FakeSerial fake;
    SerialBusReader reader(fake);
    reader.begin();
    reader.setHalfBufferCallback(onHalf);
    reader.setFullBufferCallback(onFull);

    const int N = SerialBusReader::BUFFER_SIZE + 10;
    uint8_t data[N];
    for (int i = 0; i < N; i++) data[i] = i + 200;
    fake.addResponse(data, N);

    reader.readIntoBuffer();

    TEST_ASSERT_EQUAL_INT(1, half_count);
    TEST_ASSERT_EQUAL_INT(1, full_count);
    TEST_ASSERT_EQUAL_UINT(10, reader.index());
}

// -------- MotorControl Tests --------

// Copy of AT matrix for testing
static const int8_t AT_test[3][8] = {
  { 0,  1,  0,  1,  0, -1,  0, -1},
  { 0,  1,  0, -1,  0, -1,  0,  1},
  { 1,  0,  1,  0,  1,  0,  1,  0}
};

// Helper to compute expected PWM
static uint8_t golden(int16_t raw) {
  if      (raw >  127) raw =  127;
  else if (raw < -128) raw = -128;
  return uint8_t(raw + 128);
}

void test_zero_RFcmd_neutral_PWM_on_all_motors(void) {
  uint8_t rf[3] = {0,0,0};
  uint8_t cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    TEST_ASSERT_EQUAL_UINT8(128, cmd[i]);
  }
}

void test_pure_forward_rf0_AT0_neutral(void) {
  uint8_t rf[3] = {10,0,0};
  uint8_t cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum = int16_t(rf[0]) * AT_test[0][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void test_pure_lateral_rf1_AT1_neutral(void) {
  uint8_t rf[3] = {0,20,0};
  uint8_t cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum = int16_t(rf[1]) * AT_test[1][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void test_pure_elevation_rf2_AT2_neutral(void) {
  uint8_t rf[3] = {0,0,30};
  uint8_t cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum = int16_t(rf[2]) * AT_test[2][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

void test_combined_RFcmd_sum_all_three_neutral(void) {
  uint8_t rf[3] = {5,7,9};
  uint8_t cmd[8];
  MotorControl::mapRFtoMotors(rf, cmd, (int8_t (*)[8])AT_test);
  for (uint8_t i = 0; i < 8; ++i) {
    int16_t sum =
      int16_t(rf[0]) * AT_test[0][i] +
      int16_t(rf[1]) * AT_test[1][i] +
      int16_t(rf[2]) * AT_test[2][i];
    TEST_ASSERT_EQUAL_UINT8(golden(sum), cmd[i]);
  }
}

// -------- Runner --------
void runUnityTests(void) {
  UNITY_BEGIN();
  // SerialBusReader tests
  RUN_TEST(test_half_callback_triggered);
  RUN_TEST(test_full_callback_triggered_and_wraps);
  RUN_TEST(test_partial_after_wrap);
  // MotorControl tests
  RUN_TEST(test_zero_RFcmd_neutral_PWM_on_all_motors);
  RUN_TEST(test_pure_forward_rf0_AT0_neutral);
  RUN_TEST(test_pure_lateral_rf1_AT1_neutral);
  RUN_TEST(test_pure_elevation_rf2_AT2_neutral);
  RUN_TEST(test_combined_RFcmd_sum_all_three_neutral);
  UNITY_END();
}

extern "C" void setup(void) {
  // allow USB serial to initialize
  delay(2000);
  runUnityTests();
}

extern "C" void loop(void) {
  // no-op
}
