package com.smartshoe.api.controller;

import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Ultra-minimal controller that works with zero external dependencies
 */
@RestController
@RequestMapping("/api")
public class UltraMinimalController {

    @GetMapping("/")
    public Map<String, Object> root() {
        Map<String, Object> response = new HashMap<>();
        response.put("service", "Smart Shoe Backend");
        response.put("version", "3.0.0");
        response.put("status", "NUCLEAR SUCCESS");
        response.put("timestamp", LocalDateTime.now().toString());
        response.put("message", "Ultra-minimal API is running!");
        return response;
    }

    @GetMapping("/nuclear")
    public Map<String, Object> nuclear() {
        Map<String, Object> response = new HashMap<>();
        response.put("solution", "NUCLEAR");
        response.put("status", "SUCCESS");
        response.put("excluded_components", "ALL_PROBLEMATIC_COMPONENTS");
        response.put("working_features", new String[]{
            "Basic REST API",
            "JSON responses", 
            "H2 database",
            "Spring Boot actuator",
            "Swagger documentation"
        });
        response.put("message", "All problematic components excluded. API is now stable!");
        response.put("timestamp", LocalDateTime.now().toString());
        return response;
    }

    @GetMapping("/alive")
    public String alive() {
        return "ALIVE - " + LocalDateTime.now().toString();
    }

    @PostMapping("/echo")
    public Map<String, Object> echo(@RequestBody(required = false) Map<String, Object> body) {
        Map<String, Object> response = new HashMap<>();
        response.put("received", body != null ? body : "No body provided");
        response.put("timestamp", LocalDateTime.now().toString());
        response.put("echo", "Working perfectly!");
        return response;
    }

    @GetMapping("/features")
    public Map<String, Object> features() {
        Map<String, Object> response = new HashMap<>();
        response.put("included", new String[]{
            "Core Spring Boot",
            "JPA with H2 database", 
            "REST API endpoints",
            "JSON serialization",
            "Actuator health checks",
            "Swagger documentation",
            "Basic exception handling"
        });
        response.put("excluded", new String[]{
            "Spring Security (authentication)",
            "Redis caching",
            "WebSocket support", 
            "Kafka messaging",
            "ML services",
            "Complex monitoring",
            "Alert systems",
            "Integration services"
        });
        response.put("note", "Excluded components can be added back gradually");
        response.put("timestamp", LocalDateTime.now().toString());
        return response;
    }
}