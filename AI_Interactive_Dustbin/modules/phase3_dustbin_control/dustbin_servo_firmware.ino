/*
=============================================================================
Project: AI Interactive Rejecting Dustbin
Module: Phase 3 - Microcontroller Firmware
File: dustbin_servo_firmware.ino
Target: Arduino Uno / Nano / Mega or ESP32
=============================================================================

This firmware runs on the microcontroller board and listens for ASCII serial
commands sent from the Python host controller via USB Serial.

Command Protocol:
  Python -> Board:
    PING\n         -> Handshake check
    OPEN\n         -> Rotates servo to OPEN angle (default 90 deg)
    CLOSE\n        -> Rotates servo to CLOSED angle (default 0 deg)
    REJECT\n       -> Executes rapid comedic snap-close rejection
    ANGLE <deg>\n  -> Sets servo to custom angle (0 - 180)
    STATUS\n       -> Queries current lid state

  Board -> Python:
    PONG\n
    ACK:OPEN\n
    ACK:CLOSE\n
    ACK:REJECT\n
    ACK:ANGLE <deg>\n
    STATE:<OPEN|CLOSED>\n
    ERR:UNKNOWN_COMMAND\n

Wiring:
  - Servo Signal (Orange/Yellow) -> Digital Pin 9 (Arduino) or GPIO 18 (ESP32)
  - Servo Power (Red)            -> 5V External Power or 5V Pin
  - Servo Ground (Brown/Black)   -> GND (Common Ground with Arduino)
=============================================================================
*/

#include <Servo.h>

const int SERVO_PIN = 9;
const int ANGLE_CLOSED = 0;
const int ANGLE_OPEN   = 90;

Servo dustbinServo;
String currentLidState = "CLOSED";

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  dustbinServo.attach(SERVO_PIN);
  dustbinServo.write(ANGLE_CLOSED);
  currentLidState = "CLOSED";

  Serial.println("READY:AI_DUSTBIN_SERVO");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) {
      return;
    }

    if (input == "PING") {
      Serial.println("PONG");
    }
    else if (input == "OPEN") {
      dustbinServo.write(ANGLE_OPEN);
      currentLidState = "OPEN";
      delay(200);
      Serial.println("ACK:OPEN");
    }
    else if (input == "CLOSE") {
      dustbinServo.write(ANGLE_CLOSED);
      currentLidState = "CLOSED";
      delay(200);
      Serial.println("ACK:CLOSE");
    }
    else if (input == "REJECT") {
      dustbinServo.write(45);
      delay(150);
      dustbinServo.write(60);
      delay(100);
      dustbinServo.write(30);
      delay(100);
      dustbinServo.write(ANGLE_CLOSED);
      delay(200);
      currentLidState = "CLOSED";
      Serial.println("ACK:REJECT");
    }
    else if (input.startsWith("ANGLE")) {
      int spaceIdx = input.indexOf(' ');
      if (spaceIdx > 0) {
        int angle = input.substring(spaceIdx + 1).toInt();
        angle = constrain(angle, 0, 180);
        dustbinServo.write(angle);
        delay(150);
        Serial.print("ACK:ANGLE ");
        Serial.println(angle);
      } else {
        Serial.println("ERR:INVALID_ANGLE");
      }
    }
    else if (input == "STATUS") {
      Serial.print("STATE:");
      Serial.println(currentLidState);
    }
    else {
      Serial.println("ERR:UNKNOWN_COMMAND");
    }
  }
}
