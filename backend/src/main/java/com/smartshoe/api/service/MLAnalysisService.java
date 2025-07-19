package com.smartshoe.api.service;

import com.smartshoe.api.dto.ml.MLPredictionResponse;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.entity.Alert;
import com.smartshoe.api.entity.ml.MLPrediction;
import com.smartshoe.api.config.MLConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.scheduling.annotation.Async;

import java.time.LocalDateTime;
import java.util.concurrent.CompletableFuture;
import java.util.List;
import java.util.ArrayList;

@Service
public class MLAnalysisService {
    
    private final MLPredictionService mlPredictionService;
    private final AlertService alertService;
    private final MLConfig mlConfig;
    
    @Autowired
    public MLAnalysisService(MLPredictionService mlPredictionService, 
                            AlertService alertService,
                            MLConfig mlConfig) {
        this.mlPredictionService = mlPredictionService;
        this.alertService = alertService;
        this.mlConfig = mlConfig;
    }
    
    /**
     * Perform comprehensive ML analysis for a new medical reading
     * This method runs asynchronously to avoid blocking the main thread
     */
    @Async
    public CompletableFuture<List<MLPredictionResponse>> analyzeNewReading(Patient patient, MedicalReading reading) {
        List<MLPredictionResponse> predictions = new ArrayList<>();
        
        try {
            // Check if ML service is enabled
            if (!mlConfig.getApi().isEnabled()) {
                System.out.println("ML service is disabled, skipping analysis");
                return CompletableFuture.completedFuture(predictions);
            }
            
            // Run all enabled ML models
            if (isModelEnabled("neuropathy-progression")) {
                try {
                    MLPredictionResponse neuropathyPrediction = mlPredictionService.predictNeuropathyProgression(patient, reading);
                    predictions.add(neuropathyPrediction);
                    
                    // Generate alert if high risk
                    if (neuropathyPrediction.getPrediction() > 0.7) {
                        createHighRiskAlert(patient, "Neuropathy Progression", neuropathyPrediction);
                    }
                } catch (Exception e) {
                    System.err.println("Error in neuropathy progression analysis: " + e.getMessage());
                }
            }
            
            if (isModelEnabled("glucose-complications")) {
                try {
                    MLPredictionResponse glucosePrediction = mlPredictionService.predictGlucoseComplications(patient, reading);
                    predictions.add(glucosePrediction);
                    
                    // Generate alert if high risk
                    if (glucosePrediction.getPrediction() > 0.7) {
                        createHighRiskAlert(patient, "Glucose Complications", glucosePrediction);
                    }
                } catch (Exception e) {
                    System.err.println("Error in glucose complications analysis: " + e.getMessage());
                }
            }
            
            if (isModelEnabled("anomaly-detection")) {
                try {
                    MLPredictionResponse anomalyPrediction = mlPredictionService.detectAnomalies(patient, reading);
                    predictions.add(anomalyPrediction);
                    
                    // Generate alert if anomaly detected
                    if (anomalyPrediction.getPrediction() > 0.5) {
                        createAnomalyAlert(patient, "Sensor Anomaly", anomalyPrediction);
                    }
                } catch (Exception e) {
                    System.err.println("Error in anomaly detection analysis: " + e.getMessage());
                }
            }
            
            if (isModelEnabled("risk-stratification")) {
                try {
                    MLPredictionResponse riskPrediction = mlPredictionService.calculateRiskStratification(patient, reading);
                    predictions.add(riskPrediction);
                    
                    // Generate alert if high risk
                    if (riskPrediction.getPrediction() > 0.7) {
                        createHighRiskAlert(patient, "Risk Stratification", riskPrediction);
                    }
                } catch (Exception e) {
                    System.err.println("Error in risk stratification analysis: " + e.getMessage());
                }
            }
            
            System.out.println("Completed ML analysis for patient " + patient.getId() + " - " + predictions.size() + " predictions generated");
            
        } catch (Exception e) {
            System.err.println("Error in ML analysis: " + e.getMessage());
        }
        
        return CompletableFuture.completedFuture(predictions);
    }
    
    /**
     * Run ML analysis for a patient's historical data
     */
    @Async
    public CompletableFuture<List<MLPredictionResponse>> analyzePatientHistory(Patient patient, List<MedicalReading> readings) {
        List<MLPredictionResponse> allPredictions = new ArrayList<>();
        
        try {
            for (MedicalReading reading : readings) {
                List<MLPredictionResponse> predictions = analyzeNewReading(patient, reading).get();
                allPredictions.addAll(predictions);
            }
            
            System.out.println("Completed historical ML analysis for patient " + patient.getId() + " - " + allPredictions.size() + " predictions generated");
            
        } catch (Exception e) {
            System.err.println("Error in historical ML analysis: " + e.getMessage());
        }
        
        return CompletableFuture.completedFuture(allPredictions);
    }
    
    /**
     * Get ML insights for a patient
     */
    public PatientMLInsights getPatientInsights(Long patientId) {
        try {
            List<MLPrediction> predictions = mlPredictionService.getPredictionsForPatient(patientId);
            
            PatientMLInsights insights = new PatientMLInsights();
            insights.setPatientId(patientId);
            insights.setTotalPredictions(predictions.size());
            insights.setLastAnalysisDate(predictions.isEmpty() ? null : 
                predictions.get(0).getTimestamp());
            
            // Calculate risk levels
            long highRiskCount = predictions.stream()
                .filter(p -> p.getPrediction() > 0.7)
                .count();
            insights.setHighRiskPredictions((int) highRiskCount);
            
            // Calculate average confidence
            double avgConfidence = predictions.stream()
                .mapToDouble(p -> p.getConfidence() != null ? p.getConfidence() : 0.0)
                .average()
                .orElse(0.0);
            insights.setAverageConfidence(avgConfidence);
            
            // Get latest predictions by model type
            predictions.stream()
                .collect(java.util.stream.Collectors.groupingBy(
                    MLPrediction::getModelType,
                    java.util.stream.Collectors.maxBy(
                        java.util.Comparator.comparing(MLPrediction::getTimestamp)
                    )
                ))
                .forEach((modelType, latestPrediction) -> {
                    if (latestPrediction.isPresent()) {
                        insights.getLatestPredictions().put(modelType, latestPrediction.get());
                    }
                });
            
            return insights;
            
        } catch (Exception e) {
            System.err.println("Error getting patient insights: " + e.getMessage());
            return new PatientMLInsights();
        }
    }
    
    /**
     * Check if a model is enabled
     */
    private boolean isModelEnabled(String modelType) {
        if (mlConfig.getApi().getModels() == null) {
            return true; // Default to enabled if not configured
        }
        
        MLConfig.ModelConfig modelConfig = mlConfig.getApi().getModels().get(modelType);
        return modelConfig == null || modelConfig.isEnabled();
    }
    
    /**
     * Create high risk alert
     */
    private void createHighRiskAlert(Patient patient, String type, MLPredictionResponse prediction) {
        try {
            Alert alert = new Alert();
            alert.setPatientId(patient.getId());
            alert.setAlertType("NEUROPATHY_HIGH_RISK");
            alert.setSeverity(Alert.AlertSeverity.HIGH);
            alert.setTitle("High Risk " + type + " Detected");
            alert.setMessage(String.format("ML analysis indicates high risk for %s (Risk: %.2f, Confidence: %.2f)", 
                type, prediction.getPrediction(), prediction.getConfidence()));
            alert.setRequiresAction(true);
            alert.setIsAcknowledged(false);
            
            alertService.createAlert(alert);
            
        } catch (Exception e) {
            System.err.println("Error creating high risk alert: " + e.getMessage());
        }
    }
    
    /**
     * Create anomaly alert
     */
    private void createAnomalyAlert(Patient patient, String type, MLPredictionResponse prediction) {
        try {
            Alert alert = new Alert();
            alert.setPatientId(patient.getId());
            alert.setAlertType("SENSOR_MALFUNCTION");
            alert.setSeverity(Alert.AlertSeverity.MEDIUM);
            alert.setTitle(type + " Detected");
            alert.setMessage(String.format("Anomaly detected in sensor data (Score: %.2f, Confidence: %.2f)", 
                prediction.getPrediction(), prediction.getConfidence()));
            alert.setRequiresAction(true);
            alert.setIsAcknowledged(false);
            
            alertService.createAlert(alert);
            
        } catch (Exception e) {
            System.err.println("Error creating anomaly alert: " + e.getMessage());
        }
    }
    
    /**
     * Patient ML insights data class
     */
    public static class PatientMLInsights {
        private Long patientId;
        private int totalPredictions;
        private LocalDateTime lastAnalysisDate;
        private int highRiskPredictions;
        private double averageConfidence;
        private java.util.Map<String, MLPrediction> latestPredictions = new java.util.HashMap<>();
        
        // Getters and setters
        public Long getPatientId() { return patientId; }
        public void setPatientId(Long patientId) { this.patientId = patientId; }
        
        public int getTotalPredictions() { return totalPredictions; }
        public void setTotalPredictions(int totalPredictions) { this.totalPredictions = totalPredictions; }
        
        public LocalDateTime getLastAnalysisDate() { return lastAnalysisDate; }
        public void setLastAnalysisDate(LocalDateTime lastAnalysisDate) { this.lastAnalysisDate = lastAnalysisDate; }
        
        public int getHighRiskPredictions() { return highRiskPredictions; }
        public void setHighRiskPredictions(int highRiskPredictions) { this.highRiskPredictions = highRiskPredictions; }
        
        public double getAverageConfidence() { return averageConfidence; }
        public void setAverageConfidence(double averageConfidence) { this.averageConfidence = averageConfidence; }
        
        public java.util.Map<String, MLPrediction> getLatestPredictions() { return latestPredictions; }
        public void setLatestPredictions(java.util.Map<String, MLPrediction> latestPredictions) { this.latestPredictions = latestPredictions; }
    }
}