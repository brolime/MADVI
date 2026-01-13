#include <Arduino.h>

// put function declarations here:
int count1 = 0;
int count2 = 0;

void task1( void * params){
  while(1){
    Serial.println("Task 1 counter: ");
    Serial.println(count1++);
    vTaskDelay(1000 / portTICK_PERIOD_MS);
  }
}

void task2( void * params){
  while(1){
    Serial.println("Task 2 counter: ");
    Serial.println(count2++);
    vTaskDelay(1234 / portTICK_PERIOD_MS);
  }
}


void setup() {
  Serial.begin(9600);
  xTaskCreate( task1, "task1", 1000, NULL, 1, NULL);
  xTaskCreate( task2, "task2", 1000, NULL, 1, NULL);
}

void loop() {
}
