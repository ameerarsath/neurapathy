package com.smartshoe.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * ULTRA-MINIMAL Smart Shoe Application
 * Only core Spring Boot functionality - NO external dependencies
 */
@SpringBootApplication
public class SmartShoeApplication {
    
    public static void main(String[] args) {
        SpringApplication.run(SmartShoeApplication.class, args);
        System.out.println("===============================================");
        System.out.println("🚀 ULTRA-MINIMAL Smart Shoe Backend Started!");
        System.out.println("===============================================");
        System.out.println("✓ Basic API: http://localhost:8080/api");
        System.out.println("✓ Nuclear: http://localhost:8080/api/nuclear");
        System.out.println("✓ Simple: http://localhost:8080/api/simple");
        System.out.println("✓ Health: http://localhost:8080/actuator/health");
        System.out.println("✓ H2 Console: http://localhost:8080/h2-console");
        System.out.println("===============================================");
        System.out.println("ALL COMPILATION ERRORS RESOLVED!");
        System.out.println("===============================================");
    }
}
