package com.smartshoe.api.websocket;

import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArraySet;

/**
 * WebSocket Handler for Smart Shoe Real-time Communication
 * 
 * Handles WebSocket connections for real-time data streaming including:
 * - Device sensor data
 * - Medical alerts and notifications
 * - System status updates
 * - Real-time test results
 */
@Component
public class SmartShoeWebSocketHandler extends TextWebSocketHandler {

    private static final Logger logger = LoggerFactory.getLogger(SmartShoeWebSocketHandler.class);
    
    // Store active WebSocket sessions
    private final CopyOnWriteArraySet<WebSocketSession> sessions = new CopyOnWriteArraySet<>();
    
    // Store session metadata
    private final Map<String, SessionInfo> sessionInfo = new ConcurrentHashMap<>();
    
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public void afterConnectionEstablished(WebSocketSession session) throws Exception {
        sessions.add(session);
        
        // Store session info
        SessionInfo info = new SessionInfo(
            session.getId(),
            LocalDateTime.now(),
            session.getRemoteAddress() != null ? session.getRemoteAddress().toString() : "unknown"
        );
        sessionInfo.put(session.getId(), info);
        
        logger.info("WebSocket connection established: {} from {}", 
                   session.getId(), 
                   session.getRemoteAddress());
        
        // Send welcome message
        sendWelcomeMessage(session);
        
        // Notify all sessions about new connection
        broadcastSystemMessage("user_connected", Map.of(
            "sessionId", session.getId(),
            "totalConnections", sessions.size(),
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    @Override
    public void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        String payload = message.getPayload();
        logger.debug("Received WebSocket message from {}: {}", session.getId(), payload);
        
        try {
            // Parse incoming message
            @SuppressWarnings("unchecked")
            Map<String, Object> messageData = objectMapper.readValue(payload, Map.class);
            String messageType = (String) messageData.get("type");
            
            // Handle different message types
            switch (messageType != null ? messageType : "unknown") {
                case "ping":
                    handlePingMessage(session, messageData);
                    break;
                case "subscribe":
                    handleSubscribeMessage(session, messageData);
                    break;
                case "unsubscribe":
                    handleUnsubscribeMessage(session, messageData);
                    break;
                case "device_data":
                    handleDeviceDataMessage(session, messageData);
                    break;
                default:
                    logger.warn("Unknown message type received: {}", messageType);
                    sendErrorMessage(session, "Unknown message type: " + messageType);
            }
            
        } catch (Exception e) {
            logger.error("Error processing WebSocket message from {}: {}", session.getId(), e.getMessage());
            sendErrorMessage(session, "Error processing message: " + e.getMessage());
        }
    }

    @Override
    public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
        logger.error("WebSocket transport error for session {}: {}", 
                    session.getId(), exception.getMessage());
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus closeStatus) throws Exception {
        sessions.remove(session);
        sessionInfo.remove(session.getId());
        
        logger.info("WebSocket connection closed: {} with status: {}", 
                   session.getId(), closeStatus);
        
        // Notify remaining sessions about disconnection
        broadcastSystemMessage("user_disconnected", Map.of(
            "sessionId", session.getId(),
            "totalConnections", sessions.size(),
            "closeStatus", closeStatus.toString(),
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    // Message handling methods
    private void handlePingMessage(WebSocketSession session, Map<String, Object> messageData) throws IOException {
        // Respond with pong
        sendMessage(session, Map.of(
            "type", "pong",
            "timestamp", LocalDateTime.now().toString(),
            "originalTimestamp", messageData.get("timestamp")
        ));
    }

    private void handleSubscribeMessage(WebSocketSession session, Map<String, Object> messageData) throws IOException {
        String channel = (String) messageData.get("channel");
        logger.info("Session {} subscribing to channel: {}", session.getId(), channel);
        
        // Store subscription info (you can extend this to handle actual channel management)
        sendMessage(session, Map.of(
            "type", "subscription_confirmed",
            "channel", channel,
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    private void handleUnsubscribeMessage(WebSocketSession session, Map<String, Object> messageData) throws IOException {
        String channel = (String) messageData.get("channel");
        logger.info("Session {} unsubscribing from channel: {}", session.getId(), channel);
        
        sendMessage(session, Map.of(
            "type", "unsubscription_confirmed",
            "channel", channel,
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    private void handleDeviceDataMessage(WebSocketSession session, Map<String, Object> messageData) throws IOException {
        // Echo device data to all connected sessions (in a real app, you'd process this data)
        broadcastMessage("device_data_update", messageData);
    }

    // Utility methods for sending messages
    private void sendWelcomeMessage(WebSocketSession session) throws IOException {
        sendMessage(session, Map.of(
            "type", "welcome",
            "message", "Connected to Smart Shoe WebSocket Server",
            "sessionId", session.getId(),
            "serverTime", LocalDateTime.now().toString(),
            "features", Map.of(
                "realTimeData", true,
                "alerts", true,
                "deviceStatus", true,
                "testResults", true
            )
        ));
    }

    private void sendErrorMessage(WebSocketSession session, String error) throws IOException {
        sendMessage(session, Map.of(
            "type", "error",
            "message", error,
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    private void sendMessage(WebSocketSession session, Map<String, Object> message) throws IOException {
        if (session.isOpen()) {
            String json = objectMapper.writeValueAsString(message);
            session.sendMessage(new TextMessage(json));
        }
    }

    // Public methods for broadcasting messages from other parts of the application
    public void broadcastMessage(String type, Object data) {
        Map<String, Object> message = Map.of(
            "type", type,
            "data", data,
            "timestamp", LocalDateTime.now().toString()
        );
        
        sessions.forEach(session -> {
            try {
                if (session.isOpen()) {
                    sendMessage(session, message);
                }
            } catch (IOException e) {
                logger.error("Error broadcasting message to session {}: {}", session.getId(), e.getMessage());
            }
        });
    }

    public void broadcastSystemMessage(String type, Object data) {
        broadcastMessage("system_" + type, data);
    }

    public void broadcastDeviceAlert(String deviceId, String alertType, String message) {
        broadcastMessage("device_alert", Map.of(
            "deviceId", deviceId,
            "alertType", alertType,
            "message", message,
            "severity", "HIGH",
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    public void broadcastMedicalAlert(String patientId, String alertType, String message) {
        broadcastMessage("medical_alert", Map.of(
            "patientId", patientId,
            "alertType", alertType,
            "message", message,
            "severity", "CRITICAL",
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    public void broadcastTestResult(String patientId, String testType, Object results) {
        broadcastMessage("test_result", Map.of(
            "patientId", patientId,
            "testType", testType,
            "results", results,
            "timestamp", LocalDateTime.now().toString()
        ));
    }

    // Getters for monitoring
    public int getActiveSessionCount() {
        return sessions.size();
    }

    public Map<String, SessionInfo> getSessionInfo() {
        return Map.copyOf(sessionInfo);
    }

    // Inner class for session metadata
    public static class SessionInfo {
        public final String sessionId;
        public final LocalDateTime connectedAt;
        public final String remoteAddress;

        public SessionInfo(String sessionId, LocalDateTime connectedAt, String remoteAddress) {
            this.sessionId = sessionId;
            this.connectedAt = connectedAt;
            this.remoteAddress = remoteAddress;
        }
    }
}