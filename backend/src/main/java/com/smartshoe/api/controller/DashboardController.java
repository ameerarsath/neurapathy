package com.smartshoe.api.controller;

import org.springframework.web.bind.annotation.*;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

/**
 * Dashboard API Controller
 * Provides dashboard data and statistics
 */
@RestController
@RequestMapping("/api/dashboard")
public class DashboardController {

    @GetMapping("/{userId}")
    public Map<String, Object> getDashboardData(@PathVariable String userId) {
        Map<String, Object> response = new HashMap<>();
        
        // Mock dashboard data - replace with actual database queries
        Map<String, Object> dashboardData = new HashMap<>();
        dashboardData.put("dailySteps", "0");
        dashboardData.put("stepsTrend", 0);
        dashboardData.put("stepsGoal", 10000);
        dashboardData.put("pressureStatus", "Unknown");
        dashboardData.put("pressureSubtitle", "No data available");
        dashboardData.put("batteryLevel", null);
        dashboardData.put("batterySubtitle", "No device connected");
        
        response.put("success", true);
        response.put("data", dashboardData);
        response.put("timestamp", LocalDateTime.now().toString());
        
        return response;
    }

    @GetMapping("/statistics")
    public Map<String, Object> getDashboardStatistics() {
        Map<String, Object> response = new HashMap<>();
        
        // Mock statistics data
        Map<String, Object> statistics = new HashMap<>();
        statistics.put("totalUsers", 0);
        statistics.put("activeDevices", 0);
        statistics.put("totalTests", 0);
        statistics.put("alertsToday", 0);
        
        response.put("success", true);
        response.put("data", statistics);
        response.put("timestamp", LocalDateTime.now().toString());
        
        return response;
    }

    @GetMapping("/recent-activity")
    public Map<String, Object> getRecentActivity() {
        Map<String, Object> response = new HashMap<>();
        
        // Mock recent activity data - replace with actual database queries
        response.put("success", true);
        response.put("data", new Object[0]); // Empty array for now
        response.put("timestamp", LocalDateTime.now().toString());
        
        return response;
    }
}