package com.smartshoe.api.controller;

import com.smartshoe.api.entity.User;
import com.smartshoe.api.service.TwoFactorAuthService;
import com.smartshoe.api.service.UserService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * REST Controller for Authentication and Two-Factor Authentication operations
 */
@RestController
@RequestMapping("/api/auth")
@Tag(name = "Authentication", description = "Authentication and Two-Factor Authentication API")
public class AuthController {

    private final UserService userService;
    private final TwoFactorAuthService twoFactorAuthService;
    private final PasswordEncoder passwordEncoder;

    public AuthController(UserService userService, TwoFactorAuthService twoFactorAuthService, PasswordEncoder passwordEncoder) {
        this.userService = userService;
        this.twoFactorAuthService = twoFactorAuthService;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Login endpoint with 2FA support
     */
    @PostMapping("/login")
    @Operation(summary = "Login with optional 2FA", description = "Login with username/password and optional 2FA code")
    public ResponseEntity<Map<String, Object>> login(@Valid @RequestBody LoginRequest request) {
        try {
            // Check if user can login
            if (!userService.canUserLogin(request.getUsername())) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "Account is locked or disabled"
                ));
            }

            // Get user
            Optional<User> userOpt = userService.getUserByUsername(request.getUsername());
            if (userOpt.isEmpty()) {
                userService.handleFailedLogin(request.getUsername());
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "Invalid username or password"
                ));
            }

            User user = userOpt.get();

            // Verify password
            if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
                userService.handleFailedLogin(request.getUsername());
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "Invalid username or password"
                ));
            }

            // Check if 2FA is required
            if (user.getTwoFactorEnabled() != null && user.getTwoFactorEnabled()) {
                if (request.getTotpCode() == null || request.getTotpCode().isEmpty()) {
                    return ResponseEntity.ok(Map.of(
                            "success", false,
                            "requiresTwoFactor", true,
                            "message", "Two-factor authentication code required"
                    ));
                }

                // Verify 2FA code
                boolean isValidTOTP = twoFactorAuthService.verifyTOTP(request.getUsername(), request.getTotpCode());
                boolean isValidBackup = false;
                
                if (!isValidTOTP && request.getBackupCode() != null && !request.getBackupCode().isEmpty()) {
                    isValidBackup = twoFactorAuthService.verifyBackupCode(request.getUsername(), request.getBackupCode());
                }

                if (!isValidTOTP && !isValidBackup) {
                    userService.handleFailedLogin(request.getUsername());
                    return ResponseEntity.badRequest().body(Map.of(
                            "success", false,
                            "message", "Invalid two-factor authentication code"
                    ));
                }
            }

            // Successful login
            userService.handleSuccessfulLogin(request.getUsername());

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Login successful");
            response.put("user", Map.of(
                    "id", user.getId(),
                    "username", user.getUsername(),
                    "email", user.getEmail(),
                    "firstName", user.getFirstName(),
                    "lastName", user.getLastName(),
                    "primaryRole", user.getPrimaryRole(),
                    "roles", user.getRoles(),
                    "twoFactorEnabled", user.getTwoFactorEnabled()
            ));

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Login failed: " + e.getMessage()
            ));
        }
    }

    /**
     * Enable two-factor authentication
     */
    @PostMapping("/2fa/enable")
    @Operation(summary = "Enable 2FA", description = "Enable two-factor authentication for the current user")
    public ResponseEntity<Map<String, Object>> enableTwoFactor() {
        try {
            String username = getCurrentUsername();
            if (username == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not authenticated"
                ));
            }

            TwoFactorAuthService.EnableTwoFactorResult result = twoFactorAuthService.enableTwoFactor(username);

            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Two-factor authentication setup initiated");
            response.put("secretKey", result.getSecretKey());
            response.put("qrCodeUrl", result.getQrCodeUrl());
            response.put("backupCodes", result.getBackupCodes());

            return ResponseEntity.ok(response);

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Failed to enable 2FA: " + e.getMessage()
            ));
        }
    }

    /**
     * Verify 2FA setup
     */
    @PostMapping("/2fa/verify-setup")
    @Operation(summary = "Verify 2FA setup", description = "Verify 2FA setup with TOTP code")
    public ResponseEntity<Map<String, Object>> verifyTwoFactorSetup(@Valid @RequestBody VerifyTwoFactorRequest request) {
        try {
            String username = getCurrentUsername();
            if (username == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not authenticated"
                ));
            }

            boolean isValid = twoFactorAuthService.verifyTwoFactorSetup(username, request.getTotpCode());

            if (isValid) {
                return ResponseEntity.ok(Map.of(
                        "success", true,
                        "message", "Two-factor authentication enabled successfully"
                ));
            } else {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "Invalid verification code"
                ));
            }

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Failed to verify 2FA setup: " + e.getMessage()
            ));
        }
    }

    /**
     * Disable two-factor authentication
     */
    @PostMapping("/2fa/disable")
    @Operation(summary = "Disable 2FA", description = "Disable two-factor authentication for the current user")
    public ResponseEntity<Map<String, Object>> disableTwoFactor(@Valid @RequestBody DisableTwoFactorRequest request) {
        try {
            String username = getCurrentUsername();
            if (username == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not authenticated"
                ));
            }

            // Verify current password
            Optional<User> userOpt = userService.getUserByUsername(username);
            if (userOpt.isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not found"
                ));
            }

            User user = userOpt.get();
            if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "Invalid password"
                ));
            }

            twoFactorAuthService.disableTwoFactor(username);

            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "Two-factor authentication disabled successfully"
            ));

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Failed to disable 2FA: " + e.getMessage()
            ));
        }
    }

    /**
     * Regenerate backup codes
     */
    @PostMapping("/2fa/regenerate-backup-codes")
    @Operation(summary = "Regenerate backup codes", description = "Generate new backup codes for 2FA")
    public ResponseEntity<Map<String, Object>> regenerateBackupCodes() {
        try {
            String username = getCurrentUsername();
            if (username == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not authenticated"
                ));
            }

            List<String> backupCodes = twoFactorAuthService.regenerateBackupCodes(username);

            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "Backup codes regenerated successfully",
                    "backupCodes", backupCodes
            ));

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Failed to regenerate backup codes: " + e.getMessage()
            ));
        }
    }

    /**
     * Get 2FA status
     */
    @GetMapping("/2fa/status")
    @Operation(summary = "Get 2FA status", description = "Get current user's 2FA status")
    public ResponseEntity<Map<String, Object>> getTwoFactorStatus() {
        try {
            String username = getCurrentUsername();
            if (username == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not authenticated"
                ));
            }

            boolean isEnabled = twoFactorAuthService.isTwoFactorEnabled(username);

            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "twoFactorEnabled", isEnabled
            ));

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Failed to get 2FA status: " + e.getMessage()
            ));
        }
    }

    /**
     * Change password
     */
    @PostMapping("/change-password")
    @Operation(summary = "Change password", description = "Change user password")
    public ResponseEntity<Map<String, Object>> changePassword(@Valid @RequestBody ChangePasswordRequest request) {
        try {
            String username = getCurrentUsername();
            if (username == null) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not authenticated"
                ));
            }

            // Verify current password
            Optional<User> userOpt = userService.getUserByUsername(username);
            if (userOpt.isEmpty()) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "User not found"
                ));
            }

            User user = userOpt.get();
            if (!passwordEncoder.matches(request.getCurrentPassword(), user.getPassword())) {
                return ResponseEntity.badRequest().body(Map.of(
                        "success", false,
                        "message", "Current password is incorrect"
                ));
            }

            userService.updatePassword(username, request.getNewPassword());

            return ResponseEntity.ok(Map.of(
                    "success", true,
                    "message", "Password changed successfully"
            ));

        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Failed to change password: " + e.getMessage()
            ));
        }
    }

    /**
     * Get current authenticated username
     */
    private String getCurrentUsername() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        return authentication != null ? authentication.getName() : null;
    }

    /**
     * Request DTOs
     */
    public static class LoginRequest {
        @NotBlank
        private String username;
        
        @NotBlank
        private String password;
        
        private String totpCode;
        private String backupCode;

        // Getters and setters
        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getTotpCode() { return totpCode; }
        public void setTotpCode(String totpCode) { this.totpCode = totpCode; }
        public String getBackupCode() { return backupCode; }
        public void setBackupCode(String backupCode) { this.backupCode = backupCode; }
    }

    public static class VerifyTwoFactorRequest {
        @NotBlank
        @Size(min = 6, max = 6)
        private String totpCode;

        public String getTotpCode() { return totpCode; }
        public void setTotpCode(String totpCode) { this.totpCode = totpCode; }
    }

    public static class DisableTwoFactorRequest {
        @NotBlank
        private String password;

        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    public static class ChangePasswordRequest {
        @NotBlank
        private String currentPassword;
        
        @NotBlank
        @Size(min = 6, max = 255)
        private String newPassword;

        public String getCurrentPassword() { return currentPassword; }
        public void setCurrentPassword(String currentPassword) { this.currentPassword = currentPassword; }
        public String getNewPassword() { return newPassword; }
        public void setNewPassword(String newPassword) { this.newPassword = newPassword; }
    }
}