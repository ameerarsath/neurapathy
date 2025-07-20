package com.smartshoe.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

import java.util.Arrays;
import java.util.Collections;

/**
 * CORS (Cross-Origin Resource Sharing) Configuration
 * 
 * This configuration allows the frontend applications to access the backend API
 * from different origins (domains, ports, or protocols).
 * 
 * IMPORTANT: For production deployment, ensure only trusted origins are allowed.
 */
@Configuration
public class CorsConfig implements WebMvcConfigurer {

    /**
     * Global CORS configuration for all endpoints
     * This method configures CORS settings that apply to all controllers
     */
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/**")
                .allowedOriginPatterns("*") // Allow all origins in development
                .allowedOrigins(
                    // Production origins
                    "http://13.201.120.175",
                    "http://13.201.120.175:3000",
                    "https://13.201.120.175",
                    "https://13.201.120.175:3000",
                    
                    // Development origins
                    "http://localhost:3000",
                    "http://localhost:5173",
                    "http://localhost:8080",
                    "http://127.0.0.1:3000",
                    "http://127.0.0.1:5173",
                    
                    // Mobile app origins (for Capacitor)
                    "capacitor://localhost",
                    "ionic://localhost",
                    "http://localhost",
                    "https://localhost"
                )
                .allowedMethods(
                    "GET", 
                    "POST", 
                    "PUT", 
                    "PATCH", 
                    "DELETE", 
                    "OPTIONS", 
                    "HEAD"
                )
                .allowedHeaders(
                    "Authorization",
                    "Content-Type",
                    "Accept",
                    "Origin",
                    "Access-Control-Request-Method",
                    "Access-Control-Request-Headers",
                    "X-Requested-With",
                    "X-App-Version",
                    "X-App-Environment",
                    "Cache-Control",
                    "Pragma"
                )
                .exposedHeaders(
                    "Access-Control-Allow-Origin",
                    "Access-Control-Allow-Credentials",
                    "Authorization",
                    "Content-Disposition"
                )
                .allowCredentials(true)
                .maxAge(3600); // Cache preflight response for 1 hour
    }

    /**
     * CORS Configuration Source Bean
     * Alternative/additional CORS configuration method
     * This provides more fine-grained control over CORS settings
     */
    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        
        // Production and development origins
        configuration.setAllowedOrigins(Arrays.asList(
            // Production origins
            "http://13.201.120.175",
            "http://13.201.120.175:3000",
            "https://13.201.120.175",
            "https://13.201.120.175:3000",
            
            // Development origins
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173"
        ));
        
        // Mobile app origins (Capacitor)
        configuration.setAllowedOriginPatterns(Arrays.asList(
            "capacitor://*",
            "ionic://*",
            "file://*"
        ));
        
        // Allowed HTTP methods
        configuration.setAllowedMethods(Arrays.asList(
            "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"
        ));
        
        // Allowed headers
        configuration.setAllowedHeaders(Arrays.asList(
            "*" // Allow all headers for simplicity
        ));
        
        // Exposed headers (headers that the client can access)
        configuration.setExposedHeaders(Arrays.asList(
            "Authorization",
            "Content-Disposition",
            "Access-Control-Allow-Origin",
            "Access-Control-Allow-Credentials"
        ));
        
        // Allow credentials (cookies, authorization headers)
        configuration.setAllowCredentials(true);
        
        // Cache preflight response for 1 hour
        configuration.setMaxAge(3600L);
        
        // Apply CORS configuration to all paths
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        
        return source;
    }

    /**
     * Strict CORS Configuration for Production
     * Use this method in production by replacing the above methods
     * This provides a more secure CORS configuration
     */
    // @Bean
    // public CorsConfigurationSource productionCorsConfigurationSource() {
    //     CorsConfiguration configuration = new CorsConfiguration();
    //     
    //     // Only allow specific production origins
    //     configuration.setAllowedOrigins(Arrays.asList(
    //         "http://13.201.120.175",
    //         "https://13.201.120.175",
    //         "https://your-production-domain.com"
    //     ));
    //     
    //     configuration.setAllowedMethods(Arrays.asList(
    //         "GET", "POST", "PUT", "DELETE", "OPTIONS"
    //     ));
    //     
    //     configuration.setAllowedHeaders(Arrays.asList(
    //         "Authorization",
    //         "Content-Type",
    //         "Accept"
    //     ));
    //     
    //     configuration.setAllowCredentials(true);
    //     configuration.setMaxAge(3600L);
    //     
    //     UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
    //     source.registerCorsConfiguration("/**", configuration);
    //     
    //     return source;
    // }
}