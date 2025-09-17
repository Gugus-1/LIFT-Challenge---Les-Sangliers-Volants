#include <Arduino.h>
#include <FreeRTOS.h>
#include <task.h>
#include <queue.h>
#include <semphr.h>
#include <motor_control.h>

// Configuration
static constexpr uint8_t    BUTTON_PIN       = 2;
static constexpr TickType_t TIMER1_PERIOD_MS = 500;
static constexpr TickType_t TIMER2_PERIOD_MS = 1000;
static constexpr TickType_t LOGGER_PERIOD_MS = 2000;
static constexpr size_t     LOG_QUEUE_DEPTH  = 20;
static constexpr size_t     LOG_MSG_LEN      = 64;

// Globals
static QueueHandle_t     logQueue;
static SemaphoreHandle_t buttonSem; 

// ISR - give the semaphore when button falls
void buttonISR()
{
  BaseType_t xWokenByISR = pdFALSE;
  xSemaphoreGiveFromISR(buttonSem, &xWokenByISR);
  portYIELD_FROM_ISR(xWokenByISR);
}

// TaskTimer1 - fires every TIMER1_PERIOD_MS
void TaskTimer1(void *pvParameters)
{
  (void)pvParameters;
  char msg[LOG_MSG_LEN];
  TickType_t xNextWake = xTaskGetTickCount();
  for (;;)
  {
    vTaskDelayUntil(&xNextWake, pdMS_TO_TICKS(TIMER1_PERIOD_MS));
    snprintf(msg, LOG_MSG_LEN, "Timer1 fired @ %lu", (unsigned long)xNextWake);
    xQueueSend(logQueue, msg, 0);
  }
}

// TaskTimer2 - fires every TIMER2_PERIOD_MS
void TaskTimer2(void *pvParameters)
{
  (void)pvParameters;
  char msg[LOG_MSG_LEN];
  TickType_t xNextWake = xTaskGetTickCount();
  for (;;)
  {
    vTaskDelayUntil(&xNextWake, pdMS_TO_TICKS(TIMER2_PERIOD_MS));
    snprintf(msg, LOG_MSG_LEN, "Timer2 fired @ %lu", (unsigned long)xNextWake);
    xQueueSend(logQueue, msg, 0);
  }
}

// TaskButton - waits on button semaphore
void TaskButton(void *pvParameters)
{
  (void)pvParameters;
  char msg[LOG_MSG_LEN];
  for (;;)
  {
    xSemaphoreTake(buttonSem, portMAX_DELAY);
    TickType_t now = xTaskGetTickCount();
    snprintf(msg, LOG_MSG_LEN, "Button pressed @ %lu", (unsigned long)now);
    xQueueSend(logQueue, msg, 0);
  }
}

// TaskLogger - wakes every LOGGER_PERIOD_MS and drains the queue
void TaskLogger(void *pvParameters)
{
  (void)pvParameters;
  char msg[LOG_MSG_LEN];
  for (;;)
  {
    vTaskDelay(pdMS_TO_TICKS(LOGGER_PERIOD_MS));
    while (xQueueReceive(logQueue, msg, 0) == pdPASS) {
      Serial.println(msg);
    }
  }
}

void setup()
{
  // Initialize USB CDC Serial
  Serial.begin(115200);
  while (!Serial) { }

  // Create the queue and binary semaphore
  logQueue  = xQueueCreate(LOG_QUEUE_DEPTH, LOG_MSG_LEN);
  buttonSem = xSemaphoreCreateBinary();

  // Configure the button pin and attach the ISR
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonISR, FALLING);

  // Create each FreeRTOS task: name, stack (words), params, priority, handle
  xTaskCreate(TaskTimer1,  "T1",  128, nullptr, 2, nullptr);
  xTaskCreate(TaskTimer2,  "T2",  128, nullptr, 1, nullptr);
  xTaskCreate(TaskButton,  "BTN", 128, nullptr, 3, nullptr);
  xTaskCreate(TaskLogger,  "LOG", 256, nullptr, 1, nullptr);

  // Start the scheduler (this does not return)
  vTaskStartScheduler();
}

void loop()
{
  // not used, scheduler is running
}