package com.smartshoe.api.util;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Utility class for input validation
 */
public class ValidationUtils {
    
    // Email validation pattern
    private static final Pattern EMAIL_PATTERN = 
        Pattern.compile("^[A-Za-z0-9+_.-]+@(.+)$");
    
    // Phone validation pattern (supports various formats)
    private static final Pattern PHONE_PATTERN = 
        Pattern.compile("^\\+?[1-9]\\d{1,14}$");
    
    // JSON validation for simple key-value pairs
    private static final Pattern JSON_PATTERN = 
        Pattern.compile("^\\{.*\\}$");
    
    /**
     * Validate email format
     */
    public static boolean isValidEmail(String email) {
        return email != null && EMAIL_PATTERN.matcher(email).matches();
    }
    
    /**
     * Validate phone number format
     */
    public static boolean isValidPhone(String phone) {
        return phone != null && PHONE_PATTERN.matcher(phone).matches();
    }
    
    /**
     * Validate JSON string format (basic check)
     */
    public static boolean isValidJSON(String json) {
        return json != null && JSON_PATTERN.matcher(json.trim()).matches();
    }
    
    /**
     * Validate required parameters in request map
     */
    public static boolean hasRequiredParameters(Map<String, Object> request, String... requiredParams) {
        if (request == null) {
            return false;
        }
        
        for (String param : requiredParams) {
            if (!request.containsKey(param) || request.get(param) == null) {
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * Safely convert object to Long
     */
    public static Long safeLongValue(Object value) {
        if (value == null) {
            return null;
        }
        
        try {
            if (value instanceof Number) {
                return ((Number) value).longValue();
            } else {
                return Long.valueOf(value.toString());
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid number format: " + value);
        }
    }
    
    /**
     * Safely convert object to Integer
     */
    public static Integer safeIntegerValue(Object value) {
        if (value == null) {
            return null;
        }
        
        try {
            if (value instanceof Number) {
                return ((Number) value).intValue();
            } else {
                return Integer.valueOf(value.toString());
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid number format: " + value);
        }
    }
    
    /**
     * Safely convert object to Boolean
     */
    public static Boolean safeBooleanValue(Object value) {
        if (value == null) {
            return null;
        }
        
        if (value instanceof Boolean) {
            return (Boolean) value;
        } else {
            return Boolean.valueOf(value.toString());
        }
    }
    
    /**
     * Safely convert object to String
     */
    public static String safeStringValue(Object value) {
        if (value == null) {
            return null;
        }
        
        return value.toString().trim();
    }
    
    /**
     * Validate string length
     */
    public static boolean isValidLength(String str, int minLength, int maxLength) {
        if (str == null) {
            return false;
        }
        
        int length = str.length();
        return length >= minLength && length <= maxLength;
    }
    
    /**
     * Sanitize string input to prevent XSS
     */
    public static String sanitizeString(String input) {
        if (input == null) {
            return null;
        }
        
        return input.replaceAll("[<>\"'&]", "");
    }
}