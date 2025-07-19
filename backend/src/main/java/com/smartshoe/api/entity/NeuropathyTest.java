package com.smartshoe.api.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import org.hibernate.annotations.CreationTimestamp;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Neuropathy Test Entity - Represents a complete neuropathy testing session
 */
@Entity
@Table(name = "neuropathy_tests")
@JsonIgnoreProperties({"hibernateLazyInitializer", "handler"})
public class NeuropathyTest {
    
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
    @Column(name = "test_status", nullable = false)
    @NotNull(message = "Test status is required")
    private TestStatus testStatus = TestStatus.PENDING;
    
    @Column(name = "total_stimuli")
    private Integer totalStimuli;
    
    @Column(name = "completed_stimuli")
    private Integer completedStimuli = 0;
    
    @Column(name = "test_duration_minutes")
    private Integer testDurationMinutes;
    
    @Column(name = "instructions_shown")
    private Boolean instructionsShown = false;
    
    @Column(name = "baseline_test")
    private Boolean baselineTest = false;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "foot_side")
    private FootSide footSide;
    
    @CreationTimestamp
    @Column(name = "started_at", nullable = false, updatable = false)
    private LocalDateTime startedAt;
    
    @Column(name = "completed_at")
    private LocalDateTime completedAt;
    
    @OneToMany(mappedBy = "neuropathyTest", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<TestStimulus> stimuli;
    
    @Column(name = "notes", columnDefinition = "TEXT")
    private String notes;
    
    @Column(name = "physician_notes", columnDefinition = "TEXT")
    private String physicianNotes;
    
    public enum TestStatus {
        PENDING,
        IN_PROGRESS,
        COMPLETED,
        CANCELLED,
        FAILED
    }
    
    public enum FootSide {
        LEFT,
        RIGHT,
        BOTH
    }
    
    // Constructors
    public NeuropathyTest() {}
    
    public NeuropathyTest(Patient patient, Device device, FootSide footSide) {
        this.patient = patient;
        this.device = device;
        this.footSide = footSide;
        this.testStatus = TestStatus.PENDING;
        this.completedStimuli = 0;
        this.instructionsShown = false;
        this.baselineTest = false;
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
    
    public TestStatus getTestStatus() {
        return testStatus;
    }
    
    public void setTestStatus(TestStatus testStatus) {
        this.testStatus = testStatus;
    }
    
    public Integer getTotalStimuli() {
        return totalStimuli;
    }
    
    public void setTotalStimuli(Integer totalStimuli) {
        this.totalStimuli = totalStimuli;
    }
    
    public Integer getCompletedStimuli() {
        return completedStimuli;
    }
    
    public void setCompletedStimuli(Integer completedStimuli) {
        this.completedStimuli = completedStimuli;
    }
    
    public Integer getTestDurationMinutes() {
        return testDurationMinutes;
    }
    
    public void setTestDurationMinutes(Integer testDurationMinutes) {
        this.testDurationMinutes = testDurationMinutes;
    }
    
    public Boolean getInstructionsShown() {
        return instructionsShown;
    }
    
    public void setInstructionsShown(Boolean instructionsShown) {
        this.instructionsShown = instructionsShown;
    }
    
    public Boolean getBaselineTest() {
        return baselineTest;
    }
    
    public void setBaselineTest(Boolean baselineTest) {
        this.baselineTest = baselineTest;
    }
    
    public FootSide getFootSide() {
        return footSide;
    }
    
    public void setFootSide(FootSide footSide) {
        this.footSide = footSide;
    }
    
    public LocalDateTime getStartedAt() {
        return startedAt;
    }
    
    public void setStartedAt(LocalDateTime startedAt) {
        this.startedAt = startedAt;
    }
    
    public LocalDateTime getCompletedAt() {
        return completedAt;
    }
    
    public void setCompletedAt(LocalDateTime completedAt) {
        this.completedAt = completedAt;
    }
    
    public List<TestStimulus> getStimuli() {
        return stimuli;
    }
    
    public void setStimuli(List<TestStimulus> stimuli) {
        this.stimuli = stimuli;
    }
    
    public String getNotes() {
        return notes;
    }
    
    public void setNotes(String notes) {
        this.notes = notes;
    }
    
    public String getPhysicianNotes() {
        return physicianNotes;
    }
    
    public void setPhysicianNotes(String physicianNotes) {
        this.physicianNotes = physicianNotes;
    }
    
    // Helper methods
    public boolean isCompleted() {
        return testStatus == TestStatus.COMPLETED;
    }
    
    public boolean isInProgress() {
        return testStatus == TestStatus.IN_PROGRESS;
    }
    
    public double getCompletionPercentage() {
        if (totalStimuli == null || totalStimuli == 0) {
            return 0.0;
        }
        return (completedStimuli.doubleValue() / totalStimuli.doubleValue()) * 100.0;
    }
    
    public void markCompleted() {
        this.testStatus = TestStatus.COMPLETED;
        this.completedAt = LocalDateTime.now();
    }
}