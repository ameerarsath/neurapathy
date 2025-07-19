package com.smartshoe.api.service;

import com.smartshoe.api.entity.User;
import com.smartshoe.api.repository.UserRepository;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * Service layer for User entity operations
 */
@Service
@Transactional
public class UserService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public UserService(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    /**
     * Create a new user
     */
    public User createUser(String username, String password, String email, String firstName, String lastName, Set<User.Role> roles) {
        if (userRepository.existsByUsername(username)) {
            throw new RuntimeException("Username already exists: " + username);
        }

        if (email != null && userRepository.existsByEmail(email)) {
            throw new RuntimeException("Email already exists: " + email);
        }

        User user = new User();
        user.setUsername(username);
        user.setPassword(passwordEncoder.encode(password));
        user.setEmail(email);
        user.setFirstName(firstName);
        user.setLastName(lastName);
        user.setRoles(roles);
        user.setPasswordChangedAt(LocalDateTime.now());
        user.setEnabled(true);
        user.setAccountNonExpired(true);
        user.setAccountNonLocked(true);
        user.setCredentialsNonExpired(true);
        user.setTwoFactorEnabled(false);
        user.setFailedLoginAttempts(0);

        return userRepository.save(user);
    }

    /**
     * Get user by username
     */
    public Optional<User> getUserByUsername(String username) {
        return userRepository.findByUsername(username);
    }

    /**
     * Get user by ID
     */
    public Optional<User> getUserById(Long id) {
        return userRepository.findById(id);
    }

    /**
     * Update user password
     */
    public User updatePassword(String username, String newPassword) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.setPassword(passwordEncoder.encode(newPassword));
        user.setPasswordChangedAt(LocalDateTime.now());
        user.setCredentialsNonExpired(true);

        return userRepository.save(user);
    }

    /**
     * Update user profile
     */
    public User updateProfile(String username, String email, String firstName, String lastName) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        if (email != null && !email.equals(user.getEmail())) {
            if (userRepository.existsByEmail(email)) {
                throw new RuntimeException("Email already exists: " + email);
            }
            user.setEmail(email);
        }

        user.setFirstName(firstName);
        user.setLastName(lastName);

        return userRepository.save(user);
    }

    /**
     * Handle successful login
     */
    public User handleSuccessfulLogin(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.resetFailedLoginAttempts();
        return userRepository.save(user);
    }

    /**
     * Handle failed login
     */
    public User handleFailedLogin(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.incrementFailedLoginAttempts();
        return userRepository.save(user);
    }

    /**
     * Check if user can login (not locked, enabled, etc.)
     */
    public boolean canUserLogin(String username) {
        Optional<User> userOpt = userRepository.findByUsername(username);
        if (userOpt.isEmpty()) {
            return false;
        }

        User user = userOpt.get();
        return user.getEnabled() && 
               user.getAccountNonLocked() && 
               user.getAccountNonExpired() && 
               !user.isAccountLocked();
    }

    /**
     * Enable/disable user account
     */
    public User setUserEnabled(String username, boolean enabled) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.setEnabled(enabled);
        return userRepository.save(user);
    }

    /**
     * Lock user account
     */
    public User lockUser(String username, LocalDateTime lockUntil) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.setAccountNonLocked(false);
        user.setLockedUntil(lockUntil);
        return userRepository.save(user);
    }

    /**
     * Unlock user account
     */
    public User unlockUser(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        user.setAccountNonLocked(true);
        user.setLockedUntil(null);
        user.setFailedLoginAttempts(0);
        return userRepository.save(user);
    }

    /**
     * Delete user
     */
    public void deleteUser(String username) {
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found: " + username));

        userRepository.delete(user);
    }

    /**
     * Get all users
     */
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    /**
     * Get users by role
     */
    public List<User> getUsersByRole(User.Role role) {
        return userRepository.findUsersByRole(role);
    }

    /**
     * Get locked users
     */
    public List<User> getLockedUsers() {
        return userRepository.findLockedUsers(LocalDateTime.now());
    }

    /**
     * Get users with expired passwords
     */
    public List<User> getUsersWithExpiredPasswords() {
        LocalDateTime expiredBefore = LocalDateTime.now().minusDays(90);
        return userRepository.findUsersWithExpiredPasswords(expiredBefore);
    }

    /**
     * Get user statistics
     */
    public UserStatistics getUserStatistics() {
        long totalUsers = userRepository.count();
        long adminUsers = userRepository.countUsersByRole(User.Role.ADMIN);
        long providerUsers = userRepository.countUsersByRole(User.Role.PROVIDER);
        long patientUsers = userRepository.countUsersByRole(User.Role.PATIENT);
        long usersWithTwoFactor = userRepository.countUsersWithTwoFactorEnabled();
        long activeUsers = userRepository.countActiveUsers(LocalDateTime.now().minusDays(30));

        UserStatistics stats = new UserStatistics();
        stats.setTotalUsers(totalUsers);
        stats.setAdminUsers(adminUsers);
        stats.setProviderUsers(providerUsers);
        stats.setPatientUsers(patientUsers);
        stats.setUsersWithTwoFactor(usersWithTwoFactor);
        stats.setActiveUsers(activeUsers);
        return stats;
    }

    /**
     * Initialize default users (for development)
     */
    @Transactional
    public void initializeDefaultUsers() {
        if (userRepository.count() == 0) {
            // Create admin user
            createUser("admin", "admin123", "admin@smartshoe.com", "Admin", "User", Set.of(User.Role.ADMIN));
            
            // Create doctor user
            createUser("doctor", "doctor123", "doctor@smartshoe.com", "Dr. John", "Smith", Set.of(User.Role.PROVIDER));
            
            // Create patient user
            createUser("patient", "patient123", "patient@smartshoe.com", "Jane", "Doe", Set.of(User.Role.PATIENT));
            
            // Create demo user
            createUser("demo", "demo", "demo@smartshoe.com", "Demo", "User", Set.of(User.Role.USER));
        }
    }

    /**
     * User statistics DTO
     */
    public static class UserStatistics {
        private long totalUsers;
        private long adminUsers;
        private long providerUsers;
        private long patientUsers;
        private long usersWithTwoFactor;
        private long activeUsers;

        public static UserStatistics builder() {
            return new UserStatistics();
        }

        public UserStatistics totalUsers(long totalUsers) {
            this.totalUsers = totalUsers;
            return this;
        }

        public UserStatistics adminUsers(long adminUsers) {
            this.adminUsers = adminUsers;
            return this;
        }

        public UserStatistics providerUsers(long providerUsers) {
            this.providerUsers = providerUsers;
            return this;
        }

        public UserStatistics patientUsers(long patientUsers) {
            this.patientUsers = patientUsers;
            return this;
        }

        public UserStatistics usersWithTwoFactor(long usersWithTwoFactor) {
            this.usersWithTwoFactor = usersWithTwoFactor;
            return this;
        }

        public UserStatistics activeUsers(long activeUsers) {
            this.activeUsers = activeUsers;
            return this;
        }

        public UserStatistics build() {
            return this;
        }

        // Getters and setters
        public long getTotalUsers() { return totalUsers; }
        public void setTotalUsers(long totalUsers) { this.totalUsers = totalUsers; }
        public long getAdminUsers() { return adminUsers; }
        public void setAdminUsers(long adminUsers) { this.adminUsers = adminUsers; }
        public long getProviderUsers() { return providerUsers; }
        public void setProviderUsers(long providerUsers) { this.providerUsers = providerUsers; }
        public long getPatientUsers() { return patientUsers; }
        public void setPatientUsers(long patientUsers) { this.patientUsers = patientUsers; }
        public long getUsersWithTwoFactor() { return usersWithTwoFactor; }
        public void setUsersWithTwoFactor(long usersWithTwoFactor) { this.usersWithTwoFactor = usersWithTwoFactor; }
        public long getActiveUsers() { return activeUsers; }
        public void setActiveUsers(long activeUsers) { this.activeUsers = activeUsers; }
    }
}