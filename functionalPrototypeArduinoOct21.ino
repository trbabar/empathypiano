// Arduino code: send button events over serial to Python
int buttonPins[] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
int numButtons = 9;
bool buttonStates[9];

void setup() {
  Serial.begin(9600);
  for (int i = 0; i < numButtons; i++) {
    pinMode(buttonPins[i], INPUT_PULLUP);
    buttonStates[i] = HIGH;
  }
}

void loop() {
  for (int i = 0; i < numButtons; i++) {
    int state = digitalRead(buttonPins[i]);
    if (state == LOW && buttonStates[i] == HIGH) {
      // Button pressed (LOW -> active)
      Serial.print("PRESS ");
      Serial.println(i);
      buttonStates[i] = LOW;
    }
    else if (state == HIGH && buttonStates[i] == LOW) {
      // Button released
      Serial.print("RELEASE ");
      Serial.println(i);
      buttonStates[i] = HIGH;
    }
  }
  delay(10);
}
