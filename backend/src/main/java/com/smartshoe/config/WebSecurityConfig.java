package com.smartshoe.config;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.web.cors.CorsConfigurationSource;

/**
 * Web Security Configuration with CORS Integration
 * 
 * This configuration ensures that CORS settings are properly integrated
 * with Spring Security and that preflight OPTIONS requests are handled correctly.
 */
@Configuration
@EnableWebSecurity
public class WebSecurityConfig {

    @Autowired
    private CorsConfigurationSource corsConfigurationSource;

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
                
                // Public endpoints
                .requestMatchers("/api/health/**").permitAll()
                .requestMatchers("/api/status/**").permitAll()
                .requestMatchers("/api/test/**").permitAll()
                .requestMatchers("/api/credentials/**").permitAll()
                .requestMatchers("/api/endpoints/**").permitAll()
                .requestMatchers("/api/auth/login").permitAll()
                .requestMatchers("/api/auth/register").permitAll()
                
                // ML API endpoints (if they should be public)
                .requestMatchers("/api/ml/**").permitAll()
                
                // Actuator endpoints
                .requestMatchers("/actuator/**").permitAll()
                
                // H2 Console (development only)
                .requestMatchers("/h2-console/**").permitAll()
                
                // Swagger/OpenAPI documentation
                .requestMatchers("/swagger-ui/**").permitAll()
                .requestMatchers("/v3/api-docs/**").permitAll()
                .requestMatchers("/swagger-resources/**").permitAll()
                .requestMatchers("/webjars/**").permitAll()
                
                // All other endpoints require authentication
                .anyRequest().authenticated()
            );

        // Disable frame options for H2 Console (development only)
        http.headers(headers -> headers.frameOptions(frameOptions -> frameOptions.disable()));

        return http.build();
    }
}