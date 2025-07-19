package com.smartshoe.api.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

/**
 * Entity representing medical alerts for patients
 */
@Entity
@Table(name = "alerts")
@Getter
@Setter
public class Alert extends AuditableEntity {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @Column(name = "patient_id", nullable = false)
    private Long patientId;
    
    @Column(name = "alert_type", nullable = false)
    private String alertType;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "severity", nullable = false)
    private AlertSeverity severity;
    
    @Column(name = "title", nullable = false)
    private String title;
    
    @Column(name = "message", columnDefinition = "TEXT")
    private String message;
    
    @Column(name = "is_acknowledged", nullable = false)
    private Boolean isAcknowledged = false;
    
    @Column(name = "acknowledged_by")
    private String acknowledgedBy;
    
    @Column(name = "acknowledged_at")
    private java.time.LocalDateTime acknowledgedAt;
    
    @Column(name = "requires_action", nullable = false)
    private Boolean requiresAction = false;
    
    @Column(name = "metadata", columnDefinition = "TEXT")
    private String metadata;
    
    public enum AlertSeverity {
        LOW, MEDIUM, HIGH, CRITICAL
    }
    
    public enum AlertType {
        NEUROPATHY_HIGH_RISK,
        GLUCOSE_ANOMALY,
        SENSOR_MALFUNCTION,
        EMERGENCY_CONTACT,
        PROVIDER_REVIEW_REQUIRED
    }
    
    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Long getPatientId() { return patientId; }
    public void setPatientId(Long patientId) { this.patientId = patientId; }
    public String getAlertType() { return alertType; }
    public void setAlertType(String alertType) { this.alertType = alertType; }
    public AlertSeverity getSeverity() { return severity; }
    public void setSeverity(AlertSeverity severity) { this.severity = severity; }
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
    public Boolean getIsAcknowledged() { return isAcknowledged; }
    public void setIsAcknowledged(Boolean isAcknowledged) { this.isAcknowledged = isAcknowledged; }
    public String getAcknowledgedBy() { return acknowledgedBy; }
    public void setAcknowledgedBy(String acknowledgedBy) { this.acknowledgedBy = acknowledgedBy; }
    public java.time.LocalDateTime getAcknowledgedAt() { return acknowledgedAt; }
    public void setAcknowledgedAt(java.time.LocalDateTime acknowledgedAt) { this.acknowledgedAt = acknowledgedAt; }
    public Boolean getRequiresAction() { return requiresAction; }
    public void setRequiresAction(Boolean requiresAction) { this.requiresAction = requiresAction; }
    public String getMetadata() { return metadata; }
    public void setMetadata(String metadata) { this.metadata = metadata; }
}