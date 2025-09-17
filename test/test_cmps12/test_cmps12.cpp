// test/test_cmps12.cpp

#include <Arduino.h>
#include <unity.h>
#include "CMPS12.h"
#include <cstring>  // memcpy

// A minimal FakeSerial for testing
class FakeSerial : public Stream {
public:
    static const int MAX_BUF = 32;
    uint8_t rx_buf[MAX_BUF];
    uint8_t rx_len = 0, rx_pos = 0;
    uint8_t tx_buf[MAX_BUF];
    uint8_t tx_len = 0;

    FakeSerial() { rx_len = rx_pos = tx_len = 0; }

    void addResponse(const uint8_t *data, uint8_t len) {
        rx_len = min(len, MAX_BUF);
        memcpy(rx_buf, data, rx_len);
        rx_pos = 0;
    }

    void clearTx() { tx_len = 0; }

    // Stream overrides
    int available() override { return rx_pos < rx_len; }
    int read() override    { return available() ? rx_buf[rx_pos++] : -1; }
    int peek() override    { return available() ? rx_buf[rx_pos] : -1; }
    void flush() override  { /* no‑op */ }
    size_t write(uint8_t b) override {
        if (tx_len < MAX_BUF) tx_buf[tx_len++] = b;
        return 1;
    }

    // Helpers
    uint8_t getTxByte(uint8_t idx) const {
        return idx < tx_len ? tx_buf[idx] : 0xFF;
    }
    uint8_t getTxLen() const { return tx_len; }
};


void test_getVersion(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t resp[] = { 0x42 };
    fake.addResponse(resp, 1);

    uint8_t v = compass.getVersion();
    TEST_ASSERT_EQUAL_UINT8(0x42, v);
    TEST_ASSERT_EQUAL_UINT8(0x11, fake.getTxByte(0));
}

void test_getBearing8(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t resp[] = { 0x80 };
    fake.addResponse(resp, 1);

    uint8_t b = compass.getBearing8();
    TEST_ASSERT_EQUAL_UINT8(0x80, b);
    TEST_ASSERT_EQUAL_UINT8(0x12, fake.getTxByte(0));
}

void test_getBearing16(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t resp[] = { 0x0D, 0x34 };
    fake.addResponse(resp, 2);

    uint16_t b = compass.getBearing16();
    TEST_ASSERT_EQUAL_UINT16(0x0D34, b);
    TEST_ASSERT_EQUAL_UINT8(0x13, fake.getTxByte(0));
}

void test_getPitchAndRoll(void) {
    // Pitch
    {
      FakeSerial fake;
      CMPS12 compass(fake);
      compass.begin();
      const uint8_t pr[] = { 0x10 };
      fake.addResponse(pr, 1);
      int8_t pitch = compass.getPitch();
      TEST_ASSERT_EQUAL_INT8(0x10, pitch);
      TEST_ASSERT_EQUAL_UINT8(0x14, fake.getTxByte(0));
    }
    // Roll
    {
      FakeSerial fake;
      CMPS12 compass(fake);
      compass.begin();
      const uint8_t rr[] = { 0xF0 };
      fake.addResponse(rr, 1);
      int8_t roll = compass.getRoll();
      TEST_ASSERT_EQUAL_INT8(int8_t(0xF0), roll);
      TEST_ASSERT_EQUAL_UINT8(0x15, fake.getTxByte(0));
    }
}

void test_getMagRaw(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t resp[] = { 0x00, 0x01, 0xFF, 0xFE, 0x00, 0x10 };
    fake.addResponse(resp, 6);

    int16_t x, y, z;
    compass.getMagRaw(x, y, z);
    TEST_ASSERT_EQUAL_INT16(1,   x);
    TEST_ASSERT_EQUAL_INT16(-2,  y);
    TEST_ASSERT_EQUAL_INT16(16,  z);
    TEST_ASSERT_EQUAL_UINT8(0x19, fake.getTxByte(0));
}

void test_storeCalibrationProfile_success(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t ack[] = { 0x55, 0x55, 0x55 };
    fake.addResponse(ack, 3);

    bool ok = compass.storeCalibrationProfile();
    TEST_ASSERT_TRUE(ok);
    TEST_ASSERT_EQUAL_UINT8(3, fake.getTxLen());
    TEST_ASSERT_EQUAL_UINT8(0xF0, fake.getTxByte(0));
    TEST_ASSERT_EQUAL_UINT8(0xF5, fake.getTxByte(1));
    TEST_ASSERT_EQUAL_UINT8(0xF6, fake.getTxByte(2));
}

void test_storeCalibrationProfile_fail(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t ack[] = { 0x55, 0x00, 0x55 };
    fake.addResponse(ack, 3);

    bool ok = compass.storeCalibrationProfile();
    TEST_ASSERT_FALSE(ok);
}

void test_deleteCalibrationProfile_success(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t ack[] = { 0x55, 0x55, 0x55 };
    fake.addResponse(ack, 3);

    bool ok = compass.deleteCalibrationProfile();
    TEST_ASSERT_TRUE(ok);
    TEST_ASSERT_EQUAL_UINT8(0xE0, fake.getTxByte(0));
    TEST_ASSERT_EQUAL_UINT8(0xE5, fake.getTxByte(1));
    TEST_ASSERT_EQUAL_UINT8(0xE2, fake.getTxByte(2));
}

void test_deleteCalibrationProfile_fail(void) {
    FakeSerial fake;
    CMPS12 compass(fake);
    compass.begin();
    const uint8_t ack[] = { 0x55, 0x55, 0x00 };
    fake.addResponse(ack, 3);

    bool ok = compass.deleteCalibrationProfile();
    TEST_ASSERT_FALSE(ok);
}

void runUnityTests(void) {
    UNITY_BEGIN();
    RUN_TEST(test_getVersion);
    RUN_TEST(test_getBearing8);
    RUN_TEST(test_getBearing16);
    RUN_TEST(test_getPitchAndRoll);
    RUN_TEST(test_getMagRaw);
    RUN_TEST(test_storeCalibrationProfile_success);
    RUN_TEST(test_storeCalibrationProfile_fail);
    RUN_TEST(test_deleteCalibrationProfile_success);
    RUN_TEST(test_deleteCalibrationProfile_fail);
    UNITY_END();
}

extern "C" void setup(void) {
    delay(2000);
    runUnityTests();
}

extern "C" void loop(void) {
    // nothing
}
