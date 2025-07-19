package com.smartshoe.api.controller;

import com.smartshoe.api.service.TwoFactorAuthService;
import com.smartshoe.api.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * Test controller for 2FA functionality
 */
@RestController
@RequestMapping("/api/test")
@Tag(name = "Test", description = "Test endpoints for 2FA functionality")
public class TestController {

    private final TwoFactorAuthService twoFactorAuthService;
    private final UserService userService;

    public TestController(TwoFactorAuthService twoFactorAuthService, UserService userService) {
        this.twoFactorAuthService = twoFactorAuthService;
        this.userService = userService;
    }

    /**
     * Test 2FA secret generation
     */
    @GetMapping("/2fa/generate-secret")
    @Operation(summary = "Generate 2FA secret", description = "Generate a test 2FA secret")
    public ResponseEntity<Map<String, Object>> generateSecret() {
        try {
            String secret = twoFactorAuthService.generateSecretKey();
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("secret", secret);
            response.put("qrUrl", "otpauth://totp/SmartShoe:test@example.com?secret=" + secret + "&issuer=SmartShoe");
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Test user statistics
     */
    @GetMapping("/users/stats")
    @Operation(summary = "Get user statistics", description = "Get user statistics for testing")
    public ResponseEntity<Map<String, Object>> getUserStats() {
        try {
            UserService.UserStatistics stats = userService.getUserStatistics();
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("statistics", stats);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Health check for authentication system
     */
    @GetMapping("/auth/health")
    @Operation(summary = "Auth health check", description = "Check authentication system health")
    public ResponseEntity<Map<String, Object>> authHealth() {
        try {
            long userCount = userService.getAllUsers().size();
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("userCount", userCount);
            response.put("authSystemReady", true);
            
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("authSystemReady", false);
            response.put("error", e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
}