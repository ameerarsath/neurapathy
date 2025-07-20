package com.smartshoe.api.config;

import com.smartshoe.api.service.CustomUserDetailsService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfigurationSource;

/**
 * Security Configuration for Smart Shoe API
 * Merged configuration that handles both CORS and authentication
 * Supports both in-memory users (for compatibility) and database users with 2FA
 */
@Configuration
@EnableWebSecurity
public class WebSecurityConfig {

    @Autowired
    private CorsConfigurationSource corsConfigurationSource;
    
    private final CustomUserDetailsService customUserDetailsService;

    public WebSecurityConfig(CustomUserDetailsService customUserDetailsService) {
        this.customUserDetailsService = customUserDetailsService;
    }

    /**
     * Configure HTTP Security with CORS integration
     * 
     * @param http HttpSecurity object to configure
     * @return SecurityFilterChain
     * @throws Exception if configuration fails
     */
    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            // Enable CORS with our custom configuration
            .cors(cors -> cors.configurationSource(corsConfigurationSource))
            
            // Disable CSRF for API endpoints (typical for REST APIs)
            .csrf(csrf -> csrf.disable())
            
            // Configure session management (stateless for REST API)
            .sessionManagement(session -> 
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            )
            
            // Configure authorization rules
            .authorizeHttpRequests(authz -> authz
                // Allow preflight OPTIONS requests
                .requestMatchers("OPTIONS", "/**").permitAll()
                
                // Public endpoints - no authentication required
                .requestMatchers(
                    "/api/health",
                    "/api/status", 
                    "/api/test",
                    "/api/credentials",
                    "/api/endpoints",
                    "/api/dashboard/**",
                    "/api/auth/login",
                    "/api/auth/register",
                    "/actuator/**",
                    "/h2-console/**",
                    "/swagger-ui/**",
                    "/v3/api-docs/**",
                    "/api/ml/**",
                    "/ws/**",
                    "/websocket/**"
                ).permitAll()
                
                // Public API endpoints for easy testing
                .requestMatchers("/api/public/**").permitAll()
                
                // All other requests require authentication
                .anyRequest().authenticated()
            )
            .httpBasic(basic -> {}) // Enable basic authentication
            .userDetailsService(customUserDetailsService); // Use custom user details service

        // Disable frame options for H2 Console (development only)
        http.headers(headers -> headers.frameOptions(frameOptions -> frameOptions.sameOrigin()));

        return http.build();
    }

    @Bean
    public UserDetailsService userDetailsService() {
        // Create default users for testing
        UserDetails admin = User.builder()
                .username("admin")
                .password(passwordEncoder().encode("admin123"))
                .roles("ADMIN", "PROVIDER")
                .build();

        UserDetails doctor = User.builder()
                .username("doctor")
                .password(passwordEncoder().encode("doctor123"))
                .roles("PROVIDER")
                .build();

        UserDetails patient = User.builder()
                .username("patient")
                .password(passwordEncoder().encode("patient123"))
                .roles("PATIENT")
                .build();

        UserDetails demo = User.builder()
                .username("demo")
                .password(passwordEncoder().encode("demo"))
                .roles("USER")
                .build();

        return new InMemoryUserDetailsManager(admin, doctor, patient, demo);
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}