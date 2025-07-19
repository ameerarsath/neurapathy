package com.smartshoe.api.repository;

import com.smartshoe.api.entity.Device;
import com.smartshoe.api.entity.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for Device entity operations
 */
@Repository
public interface DeviceRepository extends JpaRepository<Device, Long> {
    
    /**
     * Find device by serial number
     */
    Optional<Device> findBySerialNumber(String serialNumber);
    
    /**
     * Find active devices
     */
    List<Device> findByIsActiveTrue();
    
    /**
     * Find devices by patient
     */
    List<Device> findByPatientAndIsActiveTrue(Patient patient);
    
    /**
     * Find devices by patient ID
     */
    List<Device> findByPatientIdAndIsActiveTrue(Long patientId);
    
    /**
     * Find devices by status
     */
    List<Device> findByStatusAndIsActiveTrue(Device.DeviceStatus status);
    
    /**
     * Find devices with low battery
     */
    @Query("SELECT d FROM Device d WHERE d.batteryLevel < :threshold AND d.isActive = true")
    List<Device> findDevicesWithLowBattery(@Param("threshold") int threshold);
    
    /**
     * Find devices requiring calibration
     */
    @Query("SELECT d FROM Device d WHERE " +
           "(d.isCalibrated = false OR d.calibrationDate < :threshold) " +
           "AND d.isActive = true")
    List<Device> findDevicesRequiringCalibration(@Param("threshold") LocalDateTime threshold);
    
    /**
     * Find offline devices (not synced recently)
     */
    @Query("SELECT d FROM Device d WHERE " +
           "(d.lastSync IS NULL OR d.lastSync < :threshold) " +
           "AND d.isActive = true")
    List<Device> findOfflineDevices(@Param("threshold") LocalDateTime threshold);
    
    /**
     * Find devices by model and firmware version
     */
    List<Device> findByModelAndFirmwareVersionAndIsActiveTrue(String model, String firmwareVersion);
    
    /**
     * Check if serial number exists
     */
    boolean existsBySerialNumber(String serialNumber);
    
    /**
     * Count active devices
     */
    long countByIsActiveTrue();
    
    /**
     * Count devices by status
     */
    long countByStatusAndIsActiveTrue(Device.DeviceStatus status);
    
    /**
     * Count devices assigned to patients
     */
    long countByPatientIsNotNullAndIsActiveTrue();
}