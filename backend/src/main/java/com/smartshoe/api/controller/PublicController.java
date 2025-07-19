package com.smartshoe.api.controller;

import org.springframework.web.bind.annotation.*;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Public API Controller - No authentication required
 * These endpoints are accessible without login credentials
 */
@RestController
@RequestMapping("/api")
public class PublicController {

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "UP");
        response.put("message", "Smart Shoe Backend is running successfully!");
        response.put("timestamp", LocalDateTime.now().toString());
        return response;
    }

    @GetMapping("/status")
    public Map<String, Object> status() {
        Map<String, Object> status = new HashMap<>();
        status.put("application", "Smart Shoe Backend");
        status.put("version", "3.0.0");
        status.put("status", "running");
        status.put("timestamp", LocalDateTime.now().toString());
        status.put("java_version", System.getProperty("java.version"));
        status.put("spring_profiles", System.getProperty("spring.profiles.active", "default"));
        status.put("authentication", "enabled");
        return status;
    }

    @GetMapping("/test")
    public Map<String, Object> test() {
        Map<String, Object> response = new HashMap<>();
        response.put("message", "API Test endpoint working!");
        response.put("timestamp", LocalDateTime.now().toString());
        response.put("endpoints", Map.of(
            "public", "No authentication required",
            "secured", "Authentication required - use credentials below"
        ));
        return response;
    }

    @GetMapping("/credentials")
    public Map<String, Object> getCredentials() {
        Map<String, Object> credentials = new HashMap<>();
        credentials.put("message", "Authentication required");
        credentials.put("info", "Please obtain credentials through proper user registration and authentication");
        credentials.put("registration", "Contact system administrator for account setup");
        credentials.put("authentication", "Use proper login flow - no hardcoded credentials available");
        return credentials;
    }

    @GetMapping("/endpoints")
    public Map<String, Object> getEndpoints() {
        Map<String, Object> endpoints = new HashMap<>();
        
        Map<String, String> publicEndpoints = new HashMap<>();
        publicEndpoints.put("/api/health", "Health check - no auth required");
        publicEndpoints.put("/api/status", "Application status - no auth required");
        publicEndpoints.put("/api/test", "Test endpoint - no auth required");
        publicEndpoints.put("/api/credentials", "View default credentials - no auth required");
        publicEndpoints.put("/api/endpoints", "List all endpoints - no auth required");
        publicEndpoints.put("/h2-console", "H2 Database console - no auth required");
        publicEndpoints.put("/actuator/health", "Actuator health - no auth required");
        
        Map<String, String> securedEndpoints = new HashMap<>();
        securedEndpoints.put("/api/patients/**", "Patient management - requires auth");
        securedEndpoints.put("/api/devices/**", "Device management - requires auth");
        securedEndpoints.put("/api/medical-readings/**", "Medical data - requires auth");
        
        endpoints.put("public", publicEndpoints);
        endpoints.put("secured", securedEndpoints);
        endpoints.put("authentication", "Basic Authentication (username:password)");
        
        return endpoints;
    }
}