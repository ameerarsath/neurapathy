package com.smartshoe.api.controller;

import com.smartshoe.api.websocket.SmartShoeWebSocketHandler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import java.time.LocalDateTime;
import java.util.Map;

/**
 * REST Controller for WebSocket testing and management
 */
@RestController
@RequestMapping("/api/websocket")
@CrossOrigin(origins = "*")
public class WebSocketTestController {

    @Autowired
    private SmartShoeWebSocketHandler webSocketHandler;

    /**
     * Get WebSocket connection status
     */
    @GetMapping("/status")
    public Map<String, Object> getWebSocketStatus() {
        return Map.of(
            "status", "active",
            "activeConnections", webSocketHandler.getActiveSessionCount(),
            "serverTime", LocalDateTime.now().toString(),
            "endpoints", Map.of(
                "websocket", "/ws",
                "alternative", "/websocket"
            )
        );
    }

    /**
     * Send test message to all connected WebSocket clients
     */
    @PostMapping("/broadcast/test")
    public Map<String, Object> broadcastTestMessage(@RequestBody(required = false) Map<String, Object> payload) {
        String message = payload != null && payload.containsKey("message") 
            ? (String) payload.get("message") 
            : "Test message from WebSocket Test Controller";
            
        webSocketHandler.broadcastMessage("test_broadcast", Map.of(
            "message", message,
            "source", "REST API",
            "timestamp", LocalDateTime.now().toString()
        ));
        
        return Map.of(
            "success", true,
            "message", "Test message broadcasted",
            "sentTo", webSocketHandler.getActiveSessionCount() + " connections",
            "payload", message
        );
    }

    /**
     * Send test device alert
     */
    @PostMapping("/broadcast/device-alert")
    public Map<String, Object> broadcastDeviceAlert(@RequestBody Map<String, String> payload) {
        String deviceId = payload.getOrDefault("deviceId", "SH-TEST-001");
        String alertType = payload.getOrDefault("alertType", "BATTERY_LOW");
        String message = payload.getOrDefault("message", "Test device alert");
        
        webSocketHandler.broadcastDeviceAlert(deviceId, alertType, message);
        
        return Map.of(
            "success", true,
            "alertType", "device_alert",
            "deviceId", deviceId,
            "sentTo", webSocketHandler.getActiveSessionCount() + " connections"
        );
    }

    /**
     * Send test medical alert
     */
    @PostMapping("/broadcast/medical-alert")
    public Map<String, Object> broadcastMedicalAlert(@RequestBody Map<String, String> payload) {
        String patientId = payload.getOrDefault("patientId", "PAT-TEST-001");
        String alertType = payload.getOrDefault("alertType", "THRESHOLD_EXCEEDED");
        String message = payload.getOrDefault("message", "Test medical alert");
        
        webSocketHandler.broadcastMedicalAlert(patientId, alertType, message);
        
        return Map.of(
            "success", true,
            "alertType", "medical_alert",
            "patientId", patientId,
            "sentTo", webSocketHandler.getActiveSessionCount() + " connections"
        );
    }

    /**
     * Send test results
     */
    @PostMapping("/broadcast/test-result")
    public Map<String, Object> broadcastTestResult(@RequestBody Map<String, Object> payload) {
        String patientId = (String) payload.getOrDefault("patientId", "PAT-TEST-001");
        String testType = (String) payload.getOrDefault("testType", "VIBRATION_TEST");
        Map<String, Object> results = Map.of(
            "value", payload.getOrDefault("value", 45.2),
            "unit", payload.getOrDefault("unit", "Hz"),
            "severity", payload.getOrDefault("severity", "MODERATE"),
            "quality", payload.getOrDefault("quality", 92.5)
        );
        
        webSocketHandler.broadcastTestResult(patientId, testType, results);
        
        return Map.of(
            "success", true,
            "resultType", "test_result",
            "patientId", patientId,
            "testType", testType,
            "sentTo", webSocketHandler.getActiveSessionCount() + " connections"
        );
    }

    /**
     * Get active WebSocket sessions info
     */
    @GetMapping("/sessions")
    public Map<String, Object> getActiveSessions() {
        return Map.of(
            "activeConnections", webSocketHandler.getActiveSessionCount(),
            "sessions", webSocketHandler.getSessionInfo(),
            "timestamp", LocalDateTime.now().toString()
        );
    }
}