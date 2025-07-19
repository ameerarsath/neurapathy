package com.smartshoe.api.service;

import com.smartshoe.api.entity.User;
import com.smartshoe.api.repository.UserRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.ByteBuffer;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.security.SecureRandom;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * Service for Two-Factor Authentication (2FA) operations using TOTP
 */
@Service
@Transactional
public class TwoFactorAuthService {

    private static final String ALGORITHM = "HmacSHA1";
    private static final int DIGITS = 6;
    private static final int TIME_STEP = 30; // 30 seconds
    private static final int WINDOW = 1; // Allow 1 time step tolerance
    private static final String ISSUER = "Smart Shoe Platform";

    private final UserRepository userRepository;

    public TwoFactorAuthService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    /**
     * Generate a new secret key for 2FA
     */
    public String generateSecretKey() {
        SecureRandom random = new SecureRandom();
        byte[] bytes = new byte[20]; // 160-bit key
        random.nextBytes(bytes);
        return Base64.getEncoder().encodeToString(bytes);
    }

    /**
     * Enable 2FA for a user
     */
    public EnableTwoFactorResult enableTwoFactor(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        if (user.getTwoFactorEnabled()) {
            throw new RuntimeException("Two-factor authentication is already enabled for this user");
        }

        String secretKey = generateSecretKey();
        user.setTwoFactorSecret(secretKey);
        user.setTwoFactorEnabled(false); // Will be enabled after verification

        // Generate backup codes
        List<String> backupCodes = generateBackupCodes();
        user.setBackupCodes(String.join(",", backupCodes));

        userRepository.save(user);

        String qrCodeUrl = generateQRCodeUrl(user.getUsername(), secretKey);

        EnableTwoFactorResult result = new EnableTwoFactorResult();
        result.setSecretKey(secretKey);
        result.setQrCodeUrl(qrCodeUrl);
        result.setBackupCodes(backupCodes);
        return result;
    }

    /**
     * Verify 2FA setup with TOTP code
     */
    public boolean verifyTwoFactorSetup(String username, String totpCode) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        if (user.getTwoFactorSecret() == null) {
            throw new RuntimeException("Two-factor authentication setup not initiated");
        }

        if (verifyTOTPWithSecret(user.getTwoFactorSecret(), totpCode)) {
            user.setTwoFactorEnabled(true);
            userRepository.save(user);
            return true;
        }

        return false;
    }

    /**
     * Verify TOTP code during login
     */
    public boolean verifyTOTP(String username, String totpCode) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        if (!user.getTwoFactorEnabled() || user.getTwoFactorSecret() == null) {
            throw new RuntimeException("Two-factor authentication is not enabled for this user");
        }

        return verifyTOTPWithSecret(user.getTwoFactorSecret(), totpCode);
    }

    /**
     * Verify backup code
     */
    public boolean verifyBackupCode(String username, String backupCode) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        if (!user.getTwoFactorEnabled() || user.getBackupCodes() == null) {
            return false;
        }

        String[] codes = user.getBackupCodes().split(",");
        List<String> remainingCodes = new ArrayList<>();

        boolean found = false;
        for (String code : codes) {
            if (code.trim().equals(backupCode.trim())) {
                found = true;
                // Don't add used backup code to remaining codes
            } else {
                remainingCodes.add(code);
            }
        }

        if (found) {
            user.setBackupCodes(String.join(",", remainingCodes));
            userRepository.save(user);
            return true;
        }

        return false;
    }

    /**
     * Disable 2FA for a user
     */
    public void disableTwoFactor(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.setTwoFactorEnabled(false);
        user.setTwoFactorSecret(null);
        user.setBackupCodes(null);

        userRepository.save(user);
    }

    /**
     * Generate new backup codes
     */
    public List<String> regenerateBackupCodes(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        if (!user.getTwoFactorEnabled()) {
            throw new RuntimeException("Two-factor authentication is not enabled for this user");
        }

        List<String> backupCodes = generateBackupCodes();
        user.setBackupCodes(String.join(",", backupCodes));
        userRepository.save(user);

        return backupCodes;
    }

    /**
     * Check if user has 2FA enabled
     */
    public boolean isTwoFactorEnabled(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        return user.getTwoFactorEnabled() != null && user.getTwoFactorEnabled();
    }

    /**
     * Generate QR code URL for Google Authenticator
     */
    private String generateQRCodeUrl(String username, String secretKey) {
        String encodedUsername = username.replace(" ", "%20");
        String encodedIssuer = ISSUER.replace(" ", "%20");
        
        return String.format(
                "otpauth://totp/%s:%s?secret=%s&issuer=%s&algorithm=SHA1&digits=%d&period=%d",
                encodedIssuer, encodedUsername, secretKey, encodedIssuer, DIGITS, TIME_STEP
        );
    }

    /**
     * Generate backup codes
     */
    private List<String> generateBackupCodes() {
        List<String> codes = new ArrayList<>();
        SecureRandom random = new SecureRandom();
        
        for (int i = 0; i < 10; i++) {
            // Generate 8-digit backup code
            String code = String.format("%08d", random.nextInt(100000000));
            codes.add(code);
        }
        
        return codes;
    }

    /**
     * Verify TOTP code against secret
     */
    private boolean verifyTOTPWithSecret(String secretKey, String totpCode) {
        try {
            byte[] key = Base64.getDecoder().decode(secretKey);
            long timeStep = Instant.now().getEpochSecond() / TIME_STEP;

            // Check current time step and adjacent time steps for clock skew tolerance
            for (int i = -WINDOW; i <= WINDOW; i++) {
                long adjustedTimeStep = timeStep + i;
                String generatedCode = generateTOTP(key, adjustedTimeStep);
                
                if (generatedCode.equals(totpCode)) {
                    return true;
                }
            }

            return false;
        } catch (Exception e) {
            throw new RuntimeException("Error verifying TOTP code", e);
        }
    }

    /**
     * Generate TOTP code for given key and time step
     */
    private String generateTOTP(byte[] key, long timeStep) throws NoSuchAlgorithmException, InvalidKeyException {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.putLong(timeStep);
        byte[] timeBytes = buffer.array();

        Mac mac = Mac.getInstance(ALGORITHM);
        mac.init(new SecretKeySpec(key, ALGORITHM));
        byte[] hash = mac.doFinal(timeBytes);

        int offset = hash[hash.length - 1] & 0x0F;
        int truncatedHash = ((hash[offset] & 0x7F) << 24) |
                           ((hash[offset + 1] & 0xFF) << 16) |
                           ((hash[offset + 2] & 0xFF) << 8) |
                           (hash[offset + 3] & 0xFF);

        int code = truncatedHash % (int) Math.pow(10, DIGITS);
        return String.format("%0" + DIGITS + "d", code);
    }

    /**
     * Result object for enabling 2FA
     */
    public static class EnableTwoFactorResult {
        private String secretKey;
        private String qrCodeUrl;
        private List<String> backupCodes;

        public static EnableTwoFactorResult builder() {
            return new EnableTwoFactorResult();
        }

        public EnableTwoFactorResult secretKey(String secretKey) {
            this.secretKey = secretKey;
            return this;
        }

        public EnableTwoFactorResult qrCodeUrl(String qrCodeUrl) {
            this.qrCodeUrl = qrCodeUrl;
            return this;
        }

        public EnableTwoFactorResult backupCodes(List<String> backupCodes) {
            this.backupCodes = backupCodes;
            return this;
        }

        public EnableTwoFactorResult build() {
            return this;
        }

        // Getters and setters
        public String getSecretKey() { return secretKey; }
        public void setSecretKey(String secretKey) { this.secretKey = secretKey; }
        public String getQrCodeUrl() { return qrCodeUrl; }
        public void setQrCodeUrl(String qrCodeUrl) { this.qrCodeUrl = qrCodeUrl; }
        public List<String> getBackupCodes() { return backupCodes; }
        public void setBackupCodes(List<String> backupCodes) { this.backupCodes = backupCodes; }
    }
}