/*
 * ESP32 Traffic Light Controller
 * Receives HTTP commands from Flask server to control 8 LEDs (4 lanes, each with Red and Green)
 * 
 * Wiring:
 * Lane A: Red=GPIO23, Green=GPIO14
 * Lane B: Red=GPIO25, Green=GPIO26
 * Lane C: Red=GPIO27, Green=GPIO32
 * Lane D: Red=GPIO18, Green=GPIO19
 * All LED negatives (-) → GND
 */

#include <WiFi.h>
#include <WebServer.h>

// WiFi credentials - CHANGE THESE
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// ESP32 Server on port 80
WebServer server(80);

// LED Pin Definitions
const int LANE_A_RED = 23;
const int LANE_A_GREEN = 14;
const int LANE_B_RED = 25;
const int LANE_B_GREEN = 26;
const int LANE_C_RED = 27;
const int LANE_C_GREEN = 32;
const int LANE_D_RED = 18;
const int LANE_D_GREEN = 19;

// Current state of each lane (0=RED, 1=GREEN)
int lane_states[4] = {0, 0, 0, 0};

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize all LED pins as OUTPUT
  pinMode(LANE_A_RED, OUTPUT);
  pinMode(LANE_A_GREEN, OUTPUT);
  pinMode(LANE_B_RED, OUTPUT);
  pinMode(LANE_B_GREEN, OUTPUT);
  pinMode(LANE_C_RED, OUTPUT);
  pinMode(LANE_C_GREEN, OUTPUT);
  pinMode(LANE_D_RED, OUTPUT);
  pinMode(LANE_D_GREEN, OUTPUT);

  // Initialize all LEDs to RED (OFF)
  setLaneState(0, 0); // Lane A = RED
  setLaneState(1, 0); // Lane B = RED
  setLaneState(2, 0); // Lane C = RED
  setLaneState(3, 0); // Lane D = RED

  // Connect to WiFi
  Serial.println("\nConnecting to WiFi...");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nFailed to connect to WiFi. Please check credentials.");
    // Create Access Point mode as fallback
    WiFi.mode(WIFI_AP);
    WiFi.softAP("ESP32-Traffic-Light", "12345678");
    Serial.print("AP IP address: ");
    Serial.println(WiFi.softAPIP());
  }

  // Setup HTTP endpoints
  server.on("/", handleRoot);
  server.on("/control", handleControl);
  server.on("/status", handleStatus);
  server.on("/test", handleTest);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("HTTP server started");
  Serial.println("Available endpoints:");
  Serial.println("  GET  /status - Get current LED states");
  Serial.println("  POST /control - Control LEDs (body: lane=X&state=RED|GREEN)");
  Serial.println("  GET  /test - Test all LEDs");
}

void loop() {
  server.handleClient();
}

// Handle root endpoint
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><title>ESP32 Traffic Controller</title></head><body>";
  html += "<h1>ESP32 Traffic Light Controller</h1>";
  html += "<p>Status: Running</p>";
  html += "<p>IP: " + WiFi.localIP().toString() + "</p>";
  html += "<h2>Control Commands:</h2>";
  html += "<p>GET /status - Get current LED states</p>";
  html += "<p>POST /control?lane=X&state=RED|GREEN - Control LEDs</p>";
  html += "<p>GET /test - Test all LEDs</p>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

// Handle control endpoint
void handleControl() {
  if (server.method() == HTTP_POST || server.method() == HTTP_GET) {
    String laneStr = server.arg("lane");
    String stateStr = server.arg("state");
    
    if (laneStr == "" || stateStr == "") {
      server.send(400, "text/plain", "Error: Missing parameters. Use: /control?lane=X&state=RED|GREEN");
      return;
    }

    int lane = laneStr.toInt() - 1; // Convert lane 1-4 to index 0-3
    int state = (stateStr == "GREEN" || stateStr == "green") ? 1 : 0;
    
    if (lane < 0 || lane > 3) {
      server.send(400, "text/plain", "Error: Lane must be 1-4");
      return;
    }

    setLaneState(lane, state);
    String response = "Lane " + String(lane + 1) + " set to " + (state == 1 ? "GREEN" : "RED");
    server.send(200, "text/plain", response);
    Serial.println(response);
  } else {
    server.send(405, "text/plain", "Method not allowed");
  }
}

// Handle status endpoint
void handleStatus() {
  String json = "{";
  json += "\"lane1\":\"" + String(lane_states[0] == 1 ? "GREEN" : "RED") + "\",";
  json += "\"lane2\":\"" + String(lane_states[1] == 1 ? "GREEN" : "RED") + "\",";
  json += "\"lane3\":\"" + String(lane_states[2] == 1 ? "GREEN" : "RED") + "\",";
  json += "\"lane4\":\"" + String(lane_states[3] == 1 ? "GREEN" : "RED") + "\"";
  json += "}";
  server.send(200, "application/json", json);
}

// Handle test endpoint - cycles through all LEDs
void handleTest() {
  Serial.println("Testing all LEDs...");
  
  // Test all RED LEDs
  for (int i = 0; i < 4; i++) {
    setLaneState(i, 0); // RED
    delay(500);
  }
  delay(1000);
  
  // Test all GREEN LEDs
  for (int i = 0; i < 4; i++) {
    setLaneState(i, 1); // GREEN
    delay(500);
  }
  delay(1000);
  
  // Reset all to RED
  for (int i = 0; i < 4; i++) {
    setLaneState(i, 0); // RED
  }
  
  server.send(200, "text/plain", "LED test completed");
}

// Handle 404
void handleNotFound() {
  server.send(404, "text/plain", "Not found");
}

// Set lane state (0=RED, 1=GREEN)
void setLaneState(int lane, int state) {
  if (lane < 0 || lane > 3) return;
  
  lane_states[lane] = state;
  
  // Control LEDs based on lane
  switch(lane) {
    case 0: // Lane A
      digitalWrite(LANE_A_RED, state == 0 ? HIGH : LOW);
      digitalWrite(LANE_A_GREEN, state == 1 ? HIGH : LOW);
      break;
    case 1: // Lane B
      digitalWrite(LANE_B_RED, state == 0 ? HIGH : LOW);
      digitalWrite(LANE_B_GREEN, state == 1 ? HIGH : LOW);
      break;
    case 2: // Lane C
      digitalWrite(LANE_C_RED, state == 0 ? HIGH : LOW);
      digitalWrite(LANE_C_GREEN, state == 1 ? HIGH : LOW);
      break;
    case 3: // Lane D
      digitalWrite(LANE_D_RED, state == 0 ? HIGH : LOW);
      digitalWrite(LANE_D_GREEN, state == 1 ? HIGH : LOW);
      break;
  }
  
  Serial.printf("Lane %d: %s\n", lane + 1, state == 1 ? "GREEN" : "RED");
}

