package com.smartshoe.api.dto.ml;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.Map;

public class MLPredictionResponse {
    
    @JsonProperty("patient_id")
    private String patientId;
    
    @JsonProperty("model_type")
    private String modelType;
    
    @JsonProperty("prediction")
    private Double prediction;
    
    @JsonProperty("confidence")
    private Double confidence;
    
    @JsonProperty("model_version")
    private String modelVersion;
    
    @JsonProperty("feature_importance")
    private Map<String, Double> featureImportance;
    
    @JsonProperty("additional_data")
    private Map<String, Object> additionalData;
    
    @JsonProperty("timestamp")
    private LocalDateTime timestamp;
    
    @JsonProperty("processing_time_ms")
    private Double processingTimeMs;
    
    @JsonProperty("cache_hit")
    private Boolean cacheHit;
    
    // Helper methods for interpretation
    public String getRiskLevel() {
        if (prediction == null) return "UNKNOWN";
        if (prediction > 0.7) return "HIGH";
        if (prediction > 0.4) return "MEDIUM";
        return "LOW";
    }
    
    public String getConfidenceLevel() {
        if (confidence == null) return "UNKNOWN";
        if (confidence > 0.8) return "HIGH";
        if (confidence > 0.6) return "MEDIUM";
        return "LOW";
    }
    
    // Getters and Setters
    public String getPatientId() { return patientId; }
    public void setPatientId(String patientId) { this.patientId = patientId; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public Double getPrediction() { return prediction; }
    public void setPrediction(Double prediction) { this.prediction = prediction; }
    public Double getConfidence() { return confidence; }
    public void setConfidence(Double confidence) { this.confidence = confidence; }
    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }
    public Map<String, Double> getFeatureImportance() { return featureImportance; }
    public void setFeatureImportance(Map<String, Double> featureImportance) { this.featureImportance = featureImportance; }
    public Map<String, Object> getAdditionalData() { return additionalData; }
    public void setAdditionalData(Map<String, Object> additionalData) { this.additionalData = additionalData; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    public Double getProcessingTimeMs() { return processingTimeMs; }
    public void setProcessingTimeMs(Double processingTimeMs) { this.processingTimeMs = processingTimeMs; }
    public Boolean getCacheHit() { return cacheHit; }
    public void setCacheHit(Boolean cacheHit) { this.cacheHit = cacheHit; }
}