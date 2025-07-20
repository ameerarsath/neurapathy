package com.smartshoe.config;

import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

import jakarta.servlet.*;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import java.io.IOException;

/**
 * Preflight CORS Filter
 * 
 * This filter specifically handles OPTIONS preflight requests to ensure
 * proper CORS headers are returned before Spring Security processes the request.
 * 
 * This is particularly important for authentication endpoints where the
 * preflight request needs to succeed before the actual login request.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class PreflightCorsFilter implements Filter {

    @Override
    public void doFilter(ServletRequest req, ServletResponse res, FilterChain chain)
            throws IOException, ServletException {
        
        HttpServletRequest request = (HttpServletRequest) req;
        HttpServletResponse response = (HttpServletResponse) res;

        // Get the origin from the request
        String origin = request.getHeader("Origin");
        
        // List of allowed origins
        String[] allowedOrigins = {
            "http://13.201.120.175",
            "http://13.201.120.175:3000",
            "https://13.201.120.175",
            "https://13.201.120.175:3000",
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "capacitor://localhost",
            "ionic://localhost"
        };
        
        // Check if origin is allowed
        boolean isAllowedOrigin = false;
        if (origin != null) {
            for (String allowedOrigin : allowedOrigins) {
                if (allowedOrigin.equals(origin)) {
                    isAllowedOrigin = true;
                    break;
                }
            }
            // Also allow Capacitor origins (mobile apps)
            if (origin.startsWith("capacitor://") || origin.startsWith("ionic://") || origin.startsWith("file://")) {
                isAllowedOrigin = true;
            }
        }
        
        // Set CORS headers if origin is allowed
        if (isAllowedOrigin) {
            response.setHeader("Access-Control-Allow-Origin", origin);
        }
        
        response.setHeader("Access-Control-Allow-Credentials", "true");
        response.setHeader("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD");
        response.setHeader("Access-Control-Allow-Headers", 
            "Authorization, Content-Type, Accept, Origin, Access-Control-Request-Method, " +
            "Access-Control-Request-Headers, X-Requested-With, X-App-Version, X-App-Environment, " +
            "Cache-Control, Pragma");
        response.setHeader("Access-Control-Expose-Headers", 
            "Authorization, Content-Disposition, Access-Control-Allow-Origin, Access-Control-Allow-Credentials");
        response.setHeader("Access-Control-Max-Age", "3600");

        // Handle preflight OPTIONS requests
        if ("OPTIONS".equalsIgnoreCase(request.getMethod())) {
            response.setStatus(HttpServletResponse.SC_OK);
            return; // Don't continue the filter chain for OPTIONS requests
        }

        // Continue with the next filter in the chain
        chain.doFilter(req, res);
    }

    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        // Initialization logic if needed
    }

    @Override
    public void destroy() {
        // Cleanup logic if needed
    }
}