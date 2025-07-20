package com.smartshoe.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

/**
 * CORS Configuration - ALLOW ALL ORIGINS
 * 
 * This configuration disables CORS restrictions completely.
 * All origins, methods, and headers are allowed.
 * 
 * WARNING: This is for development/testing only. Not recommended for production.
 */
// @Configuration - DISABLED to avoid conflicts with CorsConfig
// Uncomment this line and comment out CorsConfig to enable complete CORS bypass
public class CorsAllowAllConfig implements WebMvcConfigurer {

    /**
     * Allow ALL origins, methods, and headers - No CORS restrictions
     */
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOriginPatterns("*")     // Allow ANY origin
                .allowedMethods("*")            // Allow ANY HTTP method
                .allowedHeaders("*")            // Allow ANY headers
                .allowCredentials(false)        // Must be false with "*" origins
                .maxAge(3600);                  // Cache preflight for 1 hour
    }

    /**
     * Bean-based CORS configuration - Allow everything
     * DISABLED - Uncomment @Bean to enable
     */
    // @Bean
    public CorsConfigurationSource corsConfigurationSourceDisabled() {
        CorsConfiguration configuration = new CorsConfiguration();
        
        // Allow ALL origins
        configuration.addAllowedOriginPattern("*");
        
        // Allow ALL methods
        configuration.addAllowedMethod("*");
        
        // Allow ALL headers
        configuration.addAllowedHeader("*");
        
        // Expose common headers
        configuration.addExposedHeader("*");
        
        // Credentials must be false when using "*"
        configuration.setAllowCredentials(false);
        
        // Cache preflight for 1 hour
        configuration.setMaxAge(3600L);
        
        // Apply to all paths
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        return source;
    }
}