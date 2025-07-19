package com.smartshoe.api.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import org.hibernate.annotations.CreationTimestamp;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.LocalDateTime;

/**
 * Medical Reading Entity - Sensor data and medical measurements
 */
@Entity
@Table(name = "medical_readings")
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
public class MedicalReading {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id", nullable = false)
    @NotNull(message = "Patient is required")
    @JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
    private Patient patient;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "device_id", nullable = false)
    @NotNull(message = "Device is required")
    @JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
    private Device device;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "reading_type", nullable = false)
    @NotNull(message = "Reading type is required")
    private ReadingType readingType;
    
    @Column(name = "sensor_value")
    @DecimalMin(value = "0.0", message = "Value must be positive")
    private Double value;
    
    @Column(name = "unit", length = 20)
    private String unit;
    
    @Column(name = "temperature_data", columnDefinition = "TEXT")
    private String temperatureData; // JSON string for temperature sensors
    
    @Column(name = "vibration_data", columnDefinition = "TEXT")
    private String vibrationData; // JSON string for vibration test data
    
    @Enumerated(EnumType.STRING)
    @Column(name = "foot_side")
    private FootSide footSide;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "severity_level")
    private SeverityLevel severityLevel;
    
    @Column(name = "notes", columnDefinition = "TEXT")
    private String notes;
    
    @Column(name = "provider_notes", columnDefinition = "TEXT")
    private String providerNotes;
    
    @Column(name = "signal_strength")
    private Integer signalStrength;
    
    @Column(name = "has_motion_artifacts")
    private Boolean hasMotionArtifacts = false;
    
    @Column(name = "is_baseline")
    private Boolean isBaseline = false;
    
    @Column(name = "quality_score")
    @DecimalMin(value = "0.0", message = "Quality score must be between 0 and 100")
    @DecimalMax(value = "100.0", message = "Quality score must be between 0 and 100")
    private Double qualityScore;
    
    @CreationTimestamp
    @Column(name = "recorded_at", nullable = false, updatable = false)
    private LocalDateTime recordedAt;
    
    public enum ReadingType {
        VIBRATION,
        TEMPERATURE,
        PAIN_ASSESSMENT,
        BLOOD_GLUCOSE,
        FOOT_SCAN,
        NEUROPATHY_SCREENING
    }
    
    public enum FootSide {
        LEFT,
        RIGHT,
        BOTH
    }
    
    public enum SeverityLevel {
        NORMAL,
        MILD,
        MODERATE,
        SEVERE,
        CRITICAL
    }
    
    // Constructors
    public MedicalReading() {}
    
    public MedicalReading(Patient patient, Device device, ReadingType readingType, Double value, String unit) {
        this.patient = patient;
        this.device = device;
        this.readingType = readingType;
        this.value = value;
        this.unit = unit;
        this.hasMotionArtifacts = false;
        this.isBaseline = false;
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public Patient getPatient() {
        return patient;
    }
    
    public void setPatient(Patient patient) {
        this.patient = patient;
    }
    
    public Device getDevice() {
        return device;
    }
    
    public void setDevice(Device device) {
        this.device = device;
    }
    
    public ReadingType getReadingType() {
        return readingType;
    }
    
    public void setReadingType(ReadingType readingType) {
        this.readingType = readingType;
    }
    
    public Double getValue() {
        return value;
    }
    
    public void setValue(Double value) {
        this.value = value;
    }
    
    public String getUnit() {
        return unit;
    }
    
    public void setUnit(String unit) {
        this.unit = unit;
    }
    
    
    public String getTemperatureData() {
        return temperatureData;
    }
    
    public void setTemperatureData(String temperatureData) {
        this.temperatureData = temperatureData;
    }
    
    public String getVibrationData() {
        return vibrationData;
    }
    
    public void setVibrationData(String vibrationData) {
        this.vibrationData = vibrationData;
    }
    
    public FootSide getFootSide() {
        return footSide;
    }
    
    public void setFootSide(FootSide footSide) {
        this.footSide = footSide;
    }
    
    public SeverityLevel getSeverityLevel() {
        return severityLevel;
    }
    
    public void setSeverityLevel(SeverityLevel severityLevel) {
        this.severityLevel = severityLevel;
    }
    
    public String getNotes() {
        return notes;
    }
    
    public void setNotes(String notes) {
        this.notes = notes;
    }
    
    public String getProviderNotes() {
        return providerNotes;
    }
    
    public void setProviderNotes(String providerNotes) {
        this.providerNotes = providerNotes;
    }
    
    public Integer getSignalStrength() {
        return signalStrength;
    }
    
    public void setSignalStrength(Integer signalStrength) {
        this.signalStrength = signalStrength;
    }
    
    public Boolean getHasMotionArtifacts() {
        return hasMotionArtifacts;
    }
    
    public void setHasMotionArtifacts(Boolean hasMotionArtifacts) {
        this.hasMotionArtifacts = hasMotionArtifacts;
    }
    
    public Boolean getIsBaseline() {
        return isBaseline;
    }
    
    public void setIsBaseline(Boolean isBaseline) {
        this.isBaseline = isBaseline;
    }
    
    public Double getQualityScore() {
        return qualityScore;
    }
    
    public void setQualityScore(Double qualityScore) {
        this.qualityScore = qualityScore;
    }
    
    public LocalDateTime getRecordedAt() {
        return recordedAt;
    }
    
    public void setRecordedAt(LocalDateTime recordedAt) {
        this.recordedAt = recordedAt;
    }
    
    // Helper methods
    public boolean isAbnormal() {
        return severityLevel != null && severityLevel != SeverityLevel.NORMAL;
    }
    
    public boolean requiresAttention() {
        return severityLevel == SeverityLevel.SEVERE || 
               severityLevel == SeverityLevel.CRITICAL;
    }
    
    public boolean isHighQuality() {
        return qualityScore != null && qualityScore >= 80.0;
    }
    
    // Builder-like pattern for easier construction
    public static MedicalReading builder() {
        return new MedicalReading();
    }
    
    public MedicalReading patient(Patient patient) {
        this.patient = patient;
        return this;
    }
    
    public MedicalReading device(Device device) {
        this.device = device;
        return this;
    }
    
    public MedicalReading readingType(ReadingType readingType) {
        this.readingType = readingType;
        return this;
    }
    
    public MedicalReading value(Double value) {
        this.value = value;
        return this;
    }
    
    public MedicalReading unit(String unit) {
        this.unit = unit;
        return this;
    }
    
    public MedicalReading recordedAt(LocalDateTime recordedAt) {
        this.recordedAt = recordedAt;
        return this;
    }
    
    public MedicalReading build() {
        return this;
    }
}