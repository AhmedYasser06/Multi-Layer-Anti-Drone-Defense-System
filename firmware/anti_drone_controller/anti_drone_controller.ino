/*
 * Multi-Layer Anti-Drone Defense System — Arduino Firmware
 * ----------------------------------------------------------
 * Role: dumb actuator/sensor node. ALL decisions (radar->angle logic,
 * YOLO fusion, tracking, NMSE priority, lock logic) live on the laptop.
 * This sketch only:
 *   1. Continuously sweeps the RADAR servo + ultrasonic sensor and
 *      streams raw (angle, distance) samples over serial.
 *   2. Accepts commands from the laptop to move the PAN/TILT gimbal,
 *      toggle the RF jammer relay, and toggle the laser.
 *   3. Pauses the radar sweep while the laptop has "claimed" control
 *      during an active track (so the sweep servo doesn't fight for
 *      the serial line at the same time as fine pointing corrections).
 *
 * SERIAL PROTOCOL (ASCII, newline-terminated, 115200 baud)
 * ----------------------------------------------------------
 * Laptop -> Arduino:
 *   PT,<pan_deg>,<tilt_deg>      Move pan-tilt gimbal (0-180 each)
 *   JAM,<0|1>                    RF jammer relay off/on
 *   LASER,<0|1>                  Laser diode off/on
 *   RADAR,<0|1>                  Pause(0)/resume(1) radar sweep
 *   PING                         Health check -> replies PONG
 *
 * Arduino -> Laptop:
 *   RADAR,<angle_deg>,<dist_cm>  One radar sample (only while sweeping)
 *   ACK,<cmd>                    Command acknowledged
 *   PONG                         Reply to PING
 *
 * Wiring (defaults, change pins below to match your build):
 *   Pan servo        -> D9
 *   Tilt servo        -> D10
 *   Radar sweep servo -> D6
 *   HC-SR04 TRIG      -> D7
 *   HC-SR04 ECHO      -> D8
 *   Jammer relay IN   -> D4
 *   Laser gate (NPN/MOSFET) -> D5
 */

#include <Servo.h>

// ---------- Pin map ----------
const uint8_t PIN_PAN_SERVO    = 9;
const uint8_t PIN_TILT_SERVO   = 10;
const uint8_t PIN_RADAR_SERVO  = 6;
const uint8_t PIN_TRIG         = 7;
const uint8_t PIN_ECHO         = 8;
const uint8_t PIN_JAMMER       = 4;
const uint8_t PIN_LASER        = 5;

// ---------- Servo limits ----------
const int PAN_MIN = 0,  PAN_MAX = 180;
const int TILT_MIN = 20, TILT_MAX = 160;   // mechanical limits of your bracket
const int RADAR_MIN = 0, RADAR_MAX = 180;

Servo panServo, tiltServo, radarServo;

// ---------- Radar sweep state ----------
bool radarSweeping = true;
int radarAngle = RADAR_MIN;
int radarStep = 2;                 // degrees per step -> resolution vs. speed tradeoff
unsigned long lastRadarMove = 0;
const unsigned long RADAR_STEP_INTERVAL_MS = 40;   // servo settle time per step

// ---------- Ultrasonic timing ----------
const unsigned long US_TIMEOUT_US = 25000UL;  // ~4.3 m max range cutoff

// ---------- Serial parsing ----------
String inputLine = "";

float readDistanceCm() {
  // HC-SR04 read. Swap this function's body for mmWave (e.g. IWR1843 UART
  // frame parsing) later without touching anything else in the sketch.
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  unsigned long duration = pulseIn(PIN_ECHO, HIGH, US_TIMEOUT_US);
  if (duration == 0) return -1.0;      // no echo / out of range
  return duration * 0.0343f / 2.0f;    // speed of sound cm/us, round trip
}

void handleCommand(String line) {
  line.trim();
  if (line.length() == 0) return;

  int c1 = line.indexOf(',');
  String cmd = (c1 == -1) ? line : line.substring(0, c1);
  cmd.toUpperCase();

  if (cmd == "PT") {
    int c2 = line.indexOf(',', c1 + 1);
    if (c2 == -1) return;
    int pan  = line.substring(c1 + 1, c2).toInt();
    int tilt = line.substring(c2 + 1).toInt();
    pan  = constrain(pan, PAN_MIN, PAN_MAX);
    tilt = constrain(tilt, TILT_MIN, TILT_MAX);
    panServo.write(pan);
    tiltServo.write(tilt);
    Serial.print("ACK,PT\n");

  } else if (cmd == "JAM") {
    int val = line.substring(c1 + 1).toInt();
    digitalWrite(PIN_JAMMER, val ? HIGH : LOW);
    Serial.print("ACK,JAM\n");

  } else if (cmd == "LASER") {
    int val = line.substring(c1 + 1).toInt();
    digitalWrite(PIN_LASER, val ? HIGH : LOW);
    Serial.print("ACK,LASER\n");

  } else if (cmd == "RADAR") {
    int val = line.substring(c1 + 1).toInt();
    radarSweeping = (val != 0);
    Serial.print("ACK,RADAR\n");

  } else if (cmd == "PING") {
    Serial.print("PONG\n");
  }
}

void setup() {
  Serial.begin(115200);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_JAMMER, OUTPUT);
  pinMode(PIN_LASER, OUTPUT);
  digitalWrite(PIN_JAMMER, LOW);
  digitalWrite(PIN_LASER, LOW);

  panServo.attach(PIN_PAN_SERVO);
  tiltServo.attach(PIN_TILT_SERVO);
  radarServo.attach(PIN_RADAR_SERVO);

  panServo.write(90);
  tiltServo.write(90);
  radarServo.write(RADAR_MIN);

  inputLine.reserve(32);
}

void loop() {
  // --- 1. Non-blocking serial command read ---
  while (Serial.available() > 0) {
    char ch = (char)Serial.read();
    if (ch == '\n') {
      handleCommand(inputLine);
      inputLine = "";
    } else if (ch != '\r') {
      inputLine += ch;
    }
  }

  // --- 2. Radar sweep (only when not paused by an active track) ---
  if (radarSweeping) {
    unsigned long now = millis();
    if (now - lastRadarMove >= RADAR_STEP_INTERVAL_MS) {
      lastRadarMove = now;

      radarServo.write(radarAngle);
      float dist = readDistanceCm();
      if (dist > 0) {
        Serial.print("RADAR,");
        Serial.print(radarAngle);
        Serial.print(",");
        Serial.print(dist, 1);
        Serial.print("\n");
      }

      radarAngle += radarStep;
      if (radarAngle >= RADAR_MAX || radarAngle <= RADAR_MIN) {
        radarStep = -radarStep;      // ping-pong sweep instead of snapping back
        radarAngle = constrain(radarAngle, RADAR_MIN, RADAR_MAX);
      }
    }
  }
}
