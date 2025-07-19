package com.smartshoe.api.repository;

import com.smartshoe.api.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for User entity operations
 */
@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    /**
     * Find user by username
     */
    Optional<User> findByUsername(String username);

    /**
     * Find user by email
     */
    Optional<User> findByEmail(String email);

    /**
     * Check if username exists
     */
    boolean existsByUsername(String username);

    /**
     * Check if email exists
     */
    boolean existsByEmail(String email);

    /**
     * Find users with 2FA enabled
     */
    @Query("SELECT u FROM User u WHERE u.twoFactorEnabled = true")
    List<User> findUsersWithTwoFactorEnabled();

    /**
     * Find users with failed login attempts greater than threshold
     */
    @Query("SELECT u FROM User u WHERE u.failedLoginAttempts >= :threshold")
    List<User> findUsersWithFailedLoginAttempts(@Param("threshold") int threshold);

    /**
     * Find locked users
     */
    @Query("SELECT u FROM User u WHERE u.lockedUntil IS NOT NULL AND u.lockedUntil > :now")
    List<User> findLockedUsers(@Param("now") LocalDateTime now);

    /**
     * Find users with expired passwords
     */
    @Query("SELECT u FROM User u WHERE u.passwordChangedAt IS NULL OR u.passwordChangedAt < :expiredBefore")
    List<User> findUsersWithExpiredPasswords(@Param("expiredBefore") LocalDateTime expiredBefore);

    /**
     * Find users by role
     */
    @Query("SELECT u FROM User u JOIN u.roles r WHERE r = :role")
    List<User> findUsersByRole(@Param("role") User.Role role);

    /**
     * Find enabled users
     */
    List<User> findByEnabledTrue();

    /**
     * Find users created after specific date
     */
    List<User> findByCreatedAtAfter(LocalDateTime date);

    /**
     * Find users by last login date range
     */
    List<User> findByLastLoginAtBetween(LocalDateTime start, LocalDateTime end);

    /**
     * Count users by role
     */
    @Query("SELECT COUNT(u) FROM User u JOIN u.roles r WHERE r = :role")
    long countUsersByRole(@Param("role") User.Role role);

    /**
     * Count users with 2FA enabled
     */
    @Query("SELECT COUNT(u) FROM User u WHERE u.twoFactorEnabled = true")
    long countUsersWithTwoFactorEnabled();

    /**
     * Count active users (logged in within last 30 days)
     */
    @Query("SELECT COUNT(u) FROM User u WHERE u.lastLoginAt >= :thirtyDaysAgo")
    long countActiveUsers(@Param("thirtyDaysAgo") LocalDateTime thirtyDaysAgo);
}