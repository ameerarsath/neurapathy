package com.smartshoe.api.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import org.hibernate.annotations.CreationTimestamp;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.LocalDateTime;

/**
 * Test Stimulus Entity - Individual stimulus and patient response during neuropathy testing
 */
@Entity
@Table(name = "test_stimuli")
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
public class TestStimulus {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "neuropathy_test_id", nullable = false)
    @NotNull(message = "Neuropathy test is required")
    @JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
    private NeuropathyTest neuropathyTest;
    
    @Column(name = "stimulus_sequence")
    @Min(value = 1, message = "Stimulus sequence must be positive")
    @Max(value = 100, message = "Stimulus sequence cannot exceed 100")
    private Integer stimulusSequence;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "stimulus_type", nullable = false)
    @NotNull(message = "Stimulus type is required")
    private StimulusType stimulusType;
    
    @Column(name = "stimulus_intensity")
    @DecimalMin(value = "0.0", message = "Stimulus intensity must be non-negative")
    @DecimalMax(value = "1.0", message = "Stimulus intensity must not exceed 1.0")
    private Double stimulusIntensity; // Actual intensity delivered by device (0.0 to 1.0)
    
    @Column(name = "stimulus_location")
    @Size(max = 500, message = "Stimulus location JSON must not exceed 500 characters")
    private String stimulusLocation; // JSON: {"x": 25, "y": 50, "region": "heel"}
    
    @Column(name = "stimulus_duration_ms")
    @Min(value = 0, message = "Stimulus duration must be non-negative")
    @Max(value = 30000, message = "Stimulus duration cannot exceed 30 seconds")
    private Integer stimulusDurationMs;
    
    @Column(name = "no_stimulus_trial")
    private Boolean noStimulusTrial = false; // True when no stimulus is intentionally given
    
    // Patient Response Fields
    @Column(name = "patient_felt_sensation")
    private Boolean patientFeltSensation;
    
    @Column(name = "perceived_intensity")
    @Min(value = 0, message = "Intensity must be between 0 and 10")
    @Max(value = 10, message = "Intensity must be between 0 and 10")
    private Integer perceivedIntensity; // Patient's reported intensity (1-10)
    
    @Column(name = "perceived_location")
    private String perceivedLocation; // JSON: {"x": 30, "y": 55, "region": "heel"}
    
    @Enumerated(EnumType.STRING)
    @Column(name = "perceived_type")
    private StimulusType perceivedType;
    
    @Column(name = "response_time_ms")
    private Long responseTimeMs;
    
    @Column(name = "response_confidence")
    @Min(value = 1, message = "Confidence must be between 1 and 5")
    @Max(value = 5, message = "Confidence must be between 1 and 5")
    private Integer responseConfidence; // 1-5 scale
    
    @CreationTimestamp
    @Column(name = "stimulus_time", nullable = false, updatable = false)
    private LocalDateTime stimulusTime;
    
    @Column(name = "response_time")
    private LocalDateTime responseTime;
    
    @Column(name = "notes", columnDefinition = "TEXT")
    private String notes;
    
    public enum StimulusType {
        VIBRATION,
        TEMPERATURE_HOT,
        TEMPERATURE_COLD,
        PINPRICK,
        NONE // For control trials
    }
    
    // Constructors
    public TestStimulus() {}
    
    public TestStimulus(NeuropathyTest neuropathyTest, StimulusType stimulusType, 
                       Double stimulusIntensity, String stimulusLocation) {
        this.neuropathyTest = neuropathyTest;
        this.stimulusType = stimulusType;
        this.stimulusIntensity = stimulusIntensity;
        this.stimulusLocation = stimulusLocation;
        this.noStimulusTrial = false;
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public NeuropathyTest getNeuropathyTest() {
        return neuropathyTest;
    }
    
    public void setNeuropathyTest(NeuropathyTest neuropathyTest) {
        this.neuropathyTest = neuropathyTest;
    }
    
    public Integer getStimulusSequence() {
        return stimulusSequence;
    }
    
    public void setStimulusSequence(Integer stimulusSequence) {
        this.stimulusSequence = stimulusSequence;
    }
    
    public StimulusType getStimulusType() {
        return stimulusType;
    }
    
    public void setStimulusType(StimulusType stimulusType) {
        this.stimulusType = stimulusType;
    }
    
    public Double getStimulusIntensity() {
        return stimulusIntensity;
    }
    
    public void setStimulusIntensity(Double stimulusIntensity) {
        this.stimulusIntensity = stimulusIntensity;
    }
    
    public String getStimulusLocation() {
        return stimulusLocation;
    }
    
    public void setStimulusLocation(String stimulusLocation) {
        this.stimulusLocation = stimulusLocation;
    }
    
    public Integer getStimulusDurationMs() {
        return stimulusDurationMs;
    }
    
    public void setStimulusDurationMs(Integer stimulusDurationMs) {
        this.stimulusDurationMs = stimulusDurationMs;
    }
    
    public Boolean getNoStimulusTrial() {
        return noStimulusTrial;
    }
    
    public void setNoStimulusTrial(Boolean noStimulusTrial) {
        this.noStimulusTrial = noStimulusTrial;
    }
    
    public Boolean getPatientFeltSensation() {
        return patientFeltSensation;
    }
    
    public void setPatientFeltSensation(Boolean patientFeltSensation) {
        this.patientFeltSensation = patientFeltSensation;
    }
    
    public Integer getPerceivedIntensity() {
        return perceivedIntensity;
    }
    
    public void setPerceivedIntensity(Integer perceivedIntensity) {
        this.perceivedIntensity = perceivedIntensity;
    }
    
    public String getPerceivedLocation() {
        return perceivedLocation;
    }
    
    public void setPerceivedLocation(String perceivedLocation) {
        this.perceivedLocation = perceivedLocation;
    }
    
    public StimulusType getPerceivedType() {
        return perceivedType;
    }
    
    public void setPerceivedType(StimulusType perceivedType) {
        this.perceivedType = perceivedType;
    }
    
    public Long getResponseTimeMs() {
        return responseTimeMs;
    }
    
    public void setResponseTimeMs(Long responseTimeMs) {
        this.responseTimeMs = responseTimeMs;
    }
    
    public Integer getResponseConfidence() {
        return responseConfidence;
    }
    
    public void setResponseConfidence(Integer responseConfidence) {
        this.responseConfidence = responseConfidence;
    }
    
    public LocalDateTime getStimulusTime() {
        return stimulusTime;
    }
    
    public void setStimulusTime(LocalDateTime stimulusTime) {
        this.stimulusTime = stimulusTime;
    }
    
    public LocalDateTime getResponseTime() {
        return responseTime;
    }
    
    public void setResponseTime(LocalDateTime responseTime) {
        this.responseTime = responseTime;
    }
    
    public String getNotes() {
        return notes;
    }
    
    public void setNotes(String notes) {
        this.notes = notes;
    }
    
    // Helper methods
    public boolean isCorrectDetection() {
        if (noStimulusTrial) {
            return patientFeltSensation == null || !patientFeltSensation;
        }
        return patientFeltSensation != null && patientFeltSensation;
    }
    
    public boolean isTypeMatchCorrect() {
        return perceivedType != null && perceivedType.equals(stimulusType);
    }
    
    public double getIntensityAccuracy() {
        if (stimulusIntensity == null || perceivedIntensity == null) {
            return 0.0;
        }
        // Convert device intensity to 1-10 scale for comparison
        double deviceIntensityScaled = stimulusIntensity * 10.0;
        double difference = Math.abs(deviceIntensityScaled - perceivedIntensity);
        return Math.max(0.0, 1.0 - (difference / 10.0));
    }
}