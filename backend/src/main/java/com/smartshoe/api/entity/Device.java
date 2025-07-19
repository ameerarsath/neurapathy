package com.smartshoe.api.entity;

import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.time.LocalDateTime;

/**
 * Device Entity - Smart shoe device information
 */
@Entity
@Table(name = "devices")
public class Device {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @NotBlank(message = "Device serial number is required")
    @Size(max = 100, message = "Serial number must not exceed 100 characters")
    @Column(name = "serial_number", nullable = false, unique = true, length = 100)
    private String serialNumber;
    
    @NotBlank(message = "Device model is required")
    @Size(max = 50, message = "Model must not exceed 50 characters")
    @Column(name = "model", nullable = false, length = 50)
    private String model;
    
    @NotBlank(message = "Firmware version is required")
    @Column(name = "firmware_version", nullable = false)
    private String firmwareVersion;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "patient_id")
    @JsonIgnoreProperties({"hibernateLazyInitializer", "handler", "devices"})
    private Patient patient;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "status")
    private DeviceStatus status = DeviceStatus.INACTIVE;
    
    @Enumerated(EnumType.STRING)
    @Column(name = "device_type")
    private DeviceType deviceType = DeviceType.SMART_SHOE;
    
    @Column(name = "battery_level")
    @Min(value = 0, message = "Battery level must be between 0 and 100")
    @Max(value = 100, message = "Battery level must be between 0 and 100")
    private Integer batteryLevel;
    
    @Column(name = "last_sync")
    private LocalDateTime lastSync;
    
    @Column(name = "is_calibrated")
    private Boolean isCalibrated = false;
    
    @Column(name = "calibration_date")
    private LocalDateTime calibrationDate;
    
    @Column(name = "is_active")
    private Boolean isActive = true;
    
    @CreationTimestamp
    @Column(name = "created_at", nullable = false, updatable = false)
    private LocalDateTime createdAt;
    
    @UpdateTimestamp
    @Column(name = "updated_at")
    private LocalDateTime updatedAt;
    
    public enum DeviceStatus {
        ACTIVE,
        INACTIVE,
        MAINTENANCE,
        ERROR,
        LOW_BATTERY
    }
    
    public enum DeviceType {
        SMART_SHOE,
        SENSOR_INSOLE,
        TEMPERATURE_SENSOR
    }
    
    // Constructors
    public Device() {}
    
    public Device(String serialNumber, String model, String firmwareVersion) {
        this.serialNumber = serialNumber;
        this.model = model;
        this.firmwareVersion = firmwareVersion;
        this.status = DeviceStatus.INACTIVE;
        this.deviceType = DeviceType.SMART_SHOE;
        this.isCalibrated = false;
        this.isActive = true;
    }
    
    // Getters and Setters
    public Long getId() {
        return id;
    }
    
    public void setId(Long id) {
        this.id = id;
    }
    
    public String getSerialNumber() {
        return serialNumber;
    }
    
    public void setSerialNumber(String serialNumber) {
        this.serialNumber = serialNumber;
    }
    
    public String getModel() {
        return model;
    }
    
    public void setModel(String model) {
        this.model = model;
    }
    
    public String getFirmwareVersion() {
        return firmwareVersion;
    }
    
    public void setFirmwareVersion(String firmwareVersion) {
        this.firmwareVersion = firmwareVersion;
    }
    
    public Patient getPatient() {
        return patient;
    }
    
    public void setPatient(Patient patient) {
        this.patient = patient;
    }
    
    public DeviceStatus getStatus() {
        return status;
    }
    
    public void setStatus(DeviceStatus status) {
        this.status = status;
    }
    
    public DeviceType getDeviceType() {
        return deviceType;
    }
    
    public void setDeviceType(DeviceType deviceType) {
        this.deviceType = deviceType;
    }
    
    public Integer getBatteryLevel() {
        return batteryLevel;
    }
    
    public void setBatteryLevel(Integer batteryLevel) {
        this.batteryLevel = batteryLevel;
    }
    
    public LocalDateTime getLastSync() {
        return lastSync;
    }
    
    public void setLastSync(LocalDateTime lastSync) {
        this.lastSync = lastSync;
    }
    
    public Boolean getIsCalibrated() {
        return isCalibrated;
    }
    
    public void setIsCalibrated(Boolean isCalibrated) {
        this.isCalibrated = isCalibrated;
    }
    
    public LocalDateTime getCalibrationDate() {
        return calibrationDate;
    }
    
    public void setCalibrationDate(LocalDateTime calibrationDate) {
        this.calibrationDate = calibrationDate;
    }
    
    public Boolean getIsActive() {
        return isActive;
    }
    
    public void setIsActive(Boolean isActive) {
        this.isActive = isActive;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(LocalDateTime createdAt) {
        this.createdAt = createdAt;
    }
    
    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
    
    public void setUpdatedAt(LocalDateTime updatedAt) {
        this.updatedAt = updatedAt;
    }
    
    // Helper methods
    public boolean isLowBattery() {
        return batteryLevel != null && batteryLevel < 20;
    }
    
    public boolean requiresCalibration() {
        return !isCalibrated || 
               (calibrationDate != null && 
                calibrationDate.isBefore(LocalDateTime.now().minusDays(30)));
    }
    
    public boolean isOnline() {
        return lastSync != null && 
               lastSync.isAfter(LocalDateTime.now().minusMinutes(5));
    }
}