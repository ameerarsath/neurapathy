package com.smartshoe.api.controller;

import com.smartshoe.api.dto.ml.MLPredictionResponse;
import com.smartshoe.api.dto.ml.BatchPredictionResponse;
import com.smartshoe.api.dto.ml.ModelMetrics;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.entity.ml.MLPrediction;
import com.smartshoe.api.service.MLPredictionService;
import com.smartshoe.api.service.PatientService;
import com.smartshoe.api.service.MedicalReadingService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/ml")
@Tag(name = "ML Predictions", description = "Machine Learning predictions and analytics")
public class MLPredictionController {
    
    private final MLPredictionService mlPredictionService;
    private final PatientService patientService;
    private final MedicalReadingService medicalReadingService;
    
    public MLPredictionController(MLPredictionService mlPredictionService, 
                                 PatientService patientService,
                                 MedicalReadingService medicalReadingService) {
        this.mlPredictionService = mlPredictionService;
        this.patientService = patientService;
        this.medicalReadingService = medicalReadingService;
    }
    
    /**
     * Predict neuropathy progression for a patient
     */
    @PostMapping("/predict/neuropathy-progression/{patientId}")
    @Operation(summary = "Predict neuropathy progression", description = "Predict neuropathy progression for a patient")
    public ResponseEntity<Map<String, Object>> predictNeuropathyProgression(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Medical Reading ID (optional)") @RequestParam(required = false) Long readingId,
            jakarta.servlet.http.HttpServletRequest request) {
        
        try {
            System.out.println("=== NEUROPATHY PREDICTION REQUEST ===");
            System.out.println("Patient ID: " + patientId);
            System.out.println("Reading ID: " + readingId);
            System.out.println("Request Method: " + request.getMethod());
            System.out.println("Request URL: " + request.getRequestURL().toString());
            System.out.println("Query String: " + request.getQueryString());
            System.out.println("Content Type: " + request.getContentType());
            System.out.println("User Agent: " + request.getHeader("User-Agent"));
            System.out.println("Authorization: " + request.getHeader("Authorization"));
            System.out.println("Origin: " + request.getHeader("Origin"));
            System.out.println("======================================");
            
            // Create mock prediction response
            Map<String, Object> prediction = new HashMap<>();
            prediction.put("prediction", 0.65);
            prediction.put("confidence", 0.87);
            prediction.put("risk_level", "MEDIUM");
            prediction.put("model_type", "neuropathy_progression");
            prediction.put("timestamp", LocalDateTime.now());
            
            Map<String, Object> additionalData = new HashMap<>();
            additionalData.put("progression_rate", "moderate");
            additionalData.put("recommended_monitoring", "monthly");
            prediction.put("additional_data", additionalData);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("prediction", prediction);
            response.put("risk_level", "MEDIUM");
            response.put("confidence_level", "HIGH");
            response.put("patientId", patientId);
            response.put("readingId", readingId);
            
            System.out.println("Neuropathy prediction completed for patient: " + patientId);
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error predicting neuropathy progression: " + e.getMessage());
            e.printStackTrace();
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Prediction failed: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Predict glucose complications for a patient
     */
    @PostMapping("/predict/glucose-complications/{patientId}")
    @Operation(summary = "Predict glucose complications", description = "Predict glucose-related complications for a patient")
    public ResponseEntity<Map<String, Object>> predictGlucoseComplications(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Medical Reading ID (optional)") @RequestParam(required = false) Long readingId,
            jakarta.servlet.http.HttpServletRequest request) {
        
        try {
            System.out.println("=== GLUCOSE COMPLICATIONS REQUEST ===");
            System.out.println("Patient ID: " + patientId);
            System.out.println("Reading ID: " + readingId);
            System.out.println("Request Method: " + request.getMethod());
            System.out.println("Request URL: " + request.getRequestURL().toString());
            System.out.println("Query String: " + request.getQueryString());
            System.out.println("Content Type: " + request.getContentType());
            System.out.println("User Agent: " + request.getHeader("User-Agent"));
            System.out.println("Authorization: " + request.getHeader("Authorization"));
            System.out.println("Origin: " + request.getHeader("Origin"));
            System.out.println("=====================================");
            
            // Create mock prediction response
            Map<String, Object> prediction = new HashMap<>();
            prediction.put("prediction", 0.72);
            prediction.put("confidence", 0.91);
            prediction.put("risk_level", "HIGH");
            prediction.put("model_type", "glucose_complications");
            prediction.put("timestamp", LocalDateTime.now());
            
            Map<String, Object> additionalData = new HashMap<>();
            additionalData.put("complication_type", "vascular");
            additionalData.put("intervention_needed", true);
            prediction.put("additional_data", additionalData);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("prediction", prediction);
            response.put("risk_level", "HIGH");
            response.put("confidence_level", "HIGH");
            response.put("patientId", patientId);
            response.put("readingId", readingId);
            
            System.out.println("Glucose complications prediction completed for patient: " + patientId);
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error predicting glucose complications: " + e.getMessage());
            e.printStackTrace();
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Prediction failed: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Detect anomalies in sensor data
     */
    @PostMapping("/detect/anomalies/{patientId}")
    @Operation(summary = "Detect anomalies", description = "Detect anomalies in sensor data for a patient")
    public ResponseEntity<Map<String, Object>> detectAnomalies(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Medical Reading ID (optional)") @RequestParam(required = false) Long readingId,
            jakarta.servlet.http.HttpServletRequest request) {
        
        try {
            System.out.println("=== ANOMALY DETECTION REQUEST ===");
            System.out.println("Patient ID: " + patientId);
            System.out.println("Reading ID: " + readingId);
            System.out.println("Request Method: " + request.getMethod());
            System.out.println("Request URL: " + request.getRequestURL().toString());
            System.out.println("Query String: " + request.getQueryString());
            System.out.println("Content Type: " + request.getContentType());
            System.out.println("User Agent: " + request.getHeader("User-Agent"));
            System.out.println("Authorization: " + request.getHeader("Authorization"));
            System.out.println("Origin: " + request.getHeader("Origin"));
            System.out.println("==================================");
            
            // Create mock prediction response
            Map<String, Object> prediction = new HashMap<>();
            prediction.put("prediction", 0.15);
            prediction.put("confidence", 0.94);
            prediction.put("risk_level", "LOW");
            prediction.put("model_type", "anomaly_detection");
            prediction.put("timestamp", LocalDateTime.now());
            
            Map<String, Object> additionalData = new HashMap<>();
            additionalData.put("anomaly_detected", false);
            additionalData.put("anomaly_type", "normal");
            additionalData.put("requires_recalibration", false);
            prediction.put("additional_data", additionalData);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("prediction", prediction);
            response.put("anomaly_detected", false);
            response.put("confidence_level", "HIGH");
            response.put("patientId", patientId);
            response.put("readingId", readingId);
            
            System.out.println("Anomaly detection completed for patient: " + patientId);
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error detecting anomalies: " + e.getMessage());
            e.printStackTrace();
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Anomaly detection failed: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Calculate risk stratification for a patient
     */
    @PostMapping("/predict/risk-stratification/{patientId}")
    @Operation(summary = "Calculate risk stratification", description = "Calculate comprehensive risk stratification for a patient")
    public ResponseEntity<Map<String, Object>> calculateRiskStratification(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Medical Reading ID (optional)") @RequestParam(required = false) Long readingId,
            jakarta.servlet.http.HttpServletRequest request) {
        
        try {
            System.out.println("=== RISK STRATIFICATION REQUEST ===");
            System.out.println("Patient ID: " + patientId);
            System.out.println("Reading ID: " + readingId);
            System.out.println("Request Method: " + request.getMethod());
            System.out.println("Request URL: " + request.getRequestURL().toString());
            System.out.println("Query String: " + request.getQueryString());
            System.out.println("Content Type: " + request.getContentType());
            System.out.println("User Agent: " + request.getHeader("User-Agent"));
            System.out.println("Authorization: " + request.getHeader("Authorization"));
            System.out.println("Origin: " + request.getHeader("Origin"));
            System.out.println("====================================");
            
            // Create mock prediction response
            Map<String, Object> prediction = new HashMap<>();
            prediction.put("prediction", 0.58);
            prediction.put("confidence", 0.89);
            prediction.put("risk_level", "MEDIUM");
            prediction.put("model_type", "risk_stratification");
            prediction.put("timestamp", LocalDateTime.now());
            
            Map<String, Object> additionalData = new HashMap<>();
            additionalData.put("risk_category", "metabolic");
            additionalData.put("follow_up_required", true);
            additionalData.put("urgency", "moderate");
            prediction.put("additional_data", additionalData);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("prediction", prediction);
            response.put("risk_level", "MEDIUM");
            response.put("confidence_level", "HIGH");
            response.put("patientId", patientId);
            response.put("readingId", readingId);
            
            System.out.println("Risk stratification completed for patient: " + patientId);
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error calculating risk stratification: " + e.getMessage());
            e.printStackTrace();
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Risk stratification failed: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get predictions for a patient
     */
    @GetMapping("/predictions/{patientId}")
    @Operation(summary = "Get patient predictions", description = "Get all ML predictions for a patient")
    public ResponseEntity<Map<String, Object>> getPatientPredictions(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Model type filter") @RequestParam(required = false) String modelType) {
        
        try {
            List<MLPrediction> predictions;
            
            if (modelType != null) {
                predictions = mlPredictionService.getPredictionsForPatient(patientId)
                    .stream()
                    .filter(p -> p.getModelType().equals(modelType))
                    .toList();
            } else {
                predictions = mlPredictionService.getPredictionsForPatient(patientId);
            }
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("predictions", predictions);
            response.put("total", predictions.size());
            response.put("patientId", patientId);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error getting patient predictions: " + e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get predictions: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get model metrics
     */
    @GetMapping("/metrics/{modelType}")
    @Operation(summary = "Get model metrics", description = "Get performance metrics for a specific model")
    public ResponseEntity<Map<String, Object>> getModelMetrics(
            @Parameter(description = "Model type") @PathVariable String modelType) {
        
        try {
            ModelMetrics metrics = mlPredictionService.getModelMetrics(modelType);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("metrics", metrics);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error getting model metrics: " + e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get metrics: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get available models
     */
    @GetMapping("/models")
    @Operation(summary = "Get available models", description = "Get list of all available ML models")
    public ResponseEntity<Map<String, Object>> getAvailableModels() {
        
        try {
            Map<String, Object> models = mlPredictionService.getAvailableModels();
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("models", models);
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error getting available models: " + e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get models: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get recent predictions
     */
    @GetMapping("/predictions/recent")
    @Operation(summary = "Get recent predictions", description = "Get recent ML predictions across all patients")
    public ResponseEntity<Map<String, Object>> getRecentPredictions(
            @Parameter(description = "Number of predictions to return") @RequestParam(defaultValue = "10") int limit) {
        
        try {
            List<MLPrediction> predictions = mlPredictionService.getRecentPredictions(limit);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("predictions", predictions);
            response.put("total", predictions.size());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            System.err.println("Error getting recent predictions: " + e.getMessage());
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", "Failed to get predictions: " + e.getMessage());
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Health check for ML service
     */
    @GetMapping("/health")
    @Operation(summary = "ML service health check", description = "Check if ML service is available")
    public ResponseEntity<Map<String, Object>> healthCheck() {
        
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", LocalDateTime.now());
        response.put("success", true);
        response.put("status", "healthy");
        response.put("message", "ML Controller is responding");
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Simple test endpoint
     */
    @GetMapping("/test")
    @Operation(summary = "Simple test endpoint", description = "Test endpoint without services")
    public ResponseEntity<Map<String, Object>> testEndpoint() {
        
        Map<String, Object> response = new HashMap<>();
        response.put("timestamp", LocalDateTime.now());
        response.put("success", true);
        response.put("message", "Test endpoint working");
        
        return ResponseEntity.ok(response);
    }
}