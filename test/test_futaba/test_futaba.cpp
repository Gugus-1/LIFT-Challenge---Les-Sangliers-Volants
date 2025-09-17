// test/test_serialbusreader.cpp

#include <Arduino.h>
#include <unity.h>
#include "futaba.h"

// FakeSerial for testing SerialBusReader
class FakeSerial : public Stream {
public:
    static const int MAX_BUF = 256;
    uint8_t rx_buf[MAX_BUF];
    int rx_len = 0;
    int rx_pos = 0;

    FakeSerial() : rx_len(0), rx_pos(0) {}

    void addResponse(const uint8_t* data, int len) {
        rx_len = min(len, MAX_BUF);
        memcpy(rx_buf, data, rx_len);
        rx_pos = 0;
    }

    // Stream overrides
    int available() override {
        return rx_pos < rx_len;
    }
    int read() override {
        return available() ? rx_buf[rx_pos++] : -1;
    }
    int peek() override {
        return available() ? rx_buf[rx_pos] : -1;
    }
    void flush() override {}
    size_t write(uint8_t) override { return 1; }
};

// Counters for callbacks
static int half_count;
static int full_count;

void onHalf() {
    half_count++;
}

void onFull() {
    full_count++;
}

void setUp(void) {
    // Reset before each test
    half_count = 0;
    full_count = 0;
}

// Test that half-buffer callback fires at 25 bytes
void test_half_callback_triggered(void) {
    FakeSerial fake;
    SerialBusReader reader(fake);
    reader.begin();
    reader.setHalfBufferCallback(onHalf);
    reader.setFullBufferCallback(onFull);

    // Prepare 25 bytes of data
    const int N = SerialBusReader::BUFFER_SIZE / 2;
    uint8_t data[N];
    for (int i = 0; i < N; i++) data[i] = i;
    fake.addResponse(data, N);

    reader.readIntoBuffer();

    TEST_ASSERT_EQUAL_INT(1, half_count);
    TEST_ASSERT_EQUAL_INT(0, full_count);
    // Index should be at half point
    TEST_ASSERT_EQUAL_UINT(SerialBusReader::BUFFER_SIZE/2, reader.index());
}

// Test that full-buffer callback fires at 50 bytes and wraps index
void test_full_callback_triggered_and_wraps(void) {
    FakeSerial fake;
    SerialBusReader reader(fake);
    reader.begin();
    reader.setHalfBufferCallback(onHalf);
    reader.setFullBufferCallback(onFull);

    // Prepare 50 bytes of data
    const int N = SerialBusReader::BUFFER_SIZE;
    uint8_t data[N];
    for (int i = 0; i < N; i++) data[i] = i + 100;
    fake.addResponse(data, N);

    reader.readIntoBuffer();

    TEST_ASSERT_EQUAL_INT(1, half_count);
    TEST_ASSERT_EQUAL_INT(1, full_count);
    // Index should wrap to 0
    TEST_ASSERT_EQUAL_UINT(0, reader.index());
}

// Test partial buffer after wrap: feed 60 bytes
void test_partial_after_wrap(void) {
    FakeSerial fake;
    SerialBusReader reader(fake);
    reader.begin();
    reader.setHalfBufferCallback(onHalf);
    reader.setFullBufferCallback(onFull);

    // Prepare 60 bytes of data
    const int N = 60;
    uint8_t data[N];
    for (int i = 0; i < N; i++) data[i] = i + 200;
    fake.addResponse(data, N);

    reader.readIntoBuffer();

    TEST_ASSERT_EQUAL_INT(1, half_count);
    TEST_ASSERT_EQUAL_INT(1, full_count);
    // After 50 bytes, index resets then 10 more bytes
    TEST_ASSERT_EQUAL_UINT(10, reader.index());
}

void runUnityTests(void) {
    UNITY_BEGIN();
    RUN_TEST(test_half_callback_triggered);
    RUN_TEST(test_full_callback_triggered_and_wraps);
    RUN_TEST(test_partial_after_wrap);
    UNITY_END();
}

extern "C" void setup(void) {
    delay(2000);
    runUnityTests();
}

extern "C" void loop(void) {
    // nothing
}
