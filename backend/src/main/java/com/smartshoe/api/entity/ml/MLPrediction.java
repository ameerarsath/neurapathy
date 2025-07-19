package com.smartshoe.api.entity.ml;

import com.smartshoe.api.entity.AuditableEntity;
import jakarta.persistence.*;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.time.LocalDateTime;

@Entity
@Table(name = "ml_predictions")
public class MLPrediction extends AuditableEntity {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "patient_id", nullable = false)
    private Long patientId;
    
    @Column(name = "model_type", nullable = false)
    private String modelType;
    
    @Column(name = "prediction_value", nullable = false)
    private Double prediction;
    
    @Column(name = "confidence")
    private Double confidence;
    
    @Column(name = "model_version")
    private String modelVersion;
    
    @Column(name = "feature_importance", columnDefinition = "TEXT")
    private String featureImportance;
    
    @Column(name = "additional_data", columnDefinition = "TEXT")
    private String additionalData;
    
    @Column(name = "timestamp", nullable = false)
    private LocalDateTime timestamp;
    
    @Column(name = "processing_time_ms")
    private Double processingTimeMs;
    
    @Column(name = "cache_hit")
    private Boolean cacheHit = false;
    
    // Helper methods
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
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getPatientId() { return patientId; }
    public void setPatientId(Long patientId) { this.patientId = patientId; }
    public String getModelType() { return modelType; }
    public void setModelType(String modelType) { this.modelType = modelType; }
    public Double getPrediction() { return prediction; }
    public void setPrediction(Double prediction) { this.prediction = prediction; }
    public Double getConfidence() { return confidence; }
    public void setConfidence(Double confidence) { this.confidence = confidence; }
    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }
    public String getFeatureImportance() { return featureImportance; }
    public void setFeatureImportance(String featureImportance) { this.featureImportance = featureImportance; }
    public String getAdditionalData() { return additionalData; }
    public void setAdditionalData(String additionalData) { this.additionalData = additionalData; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
    public Double getProcessingTimeMs() { return processingTimeMs; }
    public void setProcessingTimeMs(Double processingTimeMs) { this.processingTimeMs = processingTimeMs; }
    public Boolean getCacheHit() { return cacheHit; }
    public void setCacheHit(Boolean cacheHit) { this.cacheHit = cacheHit; }
}