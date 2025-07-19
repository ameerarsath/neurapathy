package com.smartshoe.api.config;

import com.smartshoe.api.service.CustomUserDetailsService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import java.util.Arrays;

/**
 * Security Configuration for Smart Shoe API
 * Supports both in-memory users (for compatibility) and database users with 2FA
 */
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    private final CustomUserDetailsService customUserDetailsService;

    public SecurityConfig(CustomUserDetailsService customUserDetailsService) {
        this.customUserDetailsService = customUserDetailsService;
    }

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .cors(cors -> cors.configurationSource(corsConfigurationSource())) // Enable CORS
            .csrf(csrf -> csrf.disable()) // Disable CSRF for API testing
            .authorizeHttpRequests(authz -> authz
                // Public endpoints - no authentication required
                .requestMatchers(
                    "/api/health",
                    "/api/status", 
                    "/api/test",
                    "/api/credentials",
                    "/api/endpoints",
                    "/api/dashboard/**",
                    "/api/auth/login",
                    "/actuator/**",
                    "/h2-console/**",
                    "/swagger-ui/**",
                    "/v3/api-docs/**",
                    "/api/ml/**"
                ).permitAll()
                
                // Public API endpoints for easy testing
                .requestMatchers("/api/public/**").permitAll()
                
                // All other requests require authentication
                .anyRequest().authenticated()
            )
            .httpBasic(basic -> {}) // Enable basic authentication
            .userDetailsService(customUserDetailsService) // Use custom user details service
            .headers(headers -> headers.frameOptions(frameOptions -> frameOptions.sameOrigin())); // Allow H2 console iframe from same origin

        return http.build();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        // Only allow specific origins for security
        configuration.setAllowedOriginPatterns(Arrays.asList(
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173"
        ));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        configuration.setAllowCredentials(true);
        configuration.setExposedHeaders(Arrays.asList("Authorization"));
        
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
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