package com.smartshoe.api.service;

import com.smartshoe.api.entity.Device;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.repository.DeviceRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Service layer for Device entity operations
 */
@Service
@Transactional(readOnly = true)
public class DeviceService {
    
    private final DeviceRepository deviceRepository;
    private final PatientService patientService;
    
    public DeviceService(DeviceRepository deviceRepository, PatientService patientService) {
        this.deviceRepository = deviceRepository;
        this.patientService = patientService;
    }
    
    /**
     * Register a new device
     */
    @Transactional
    public Device registerDevice(Device device) {
        System.out.println("Registering new device with serial number: " + device.getSerialNumber());
        
        if (deviceRepository.existsBySerialNumber(device.getSerialNumber())) {
            throw new RuntimeException("Device with serial number " + device.getSerialNumber() + " already exists");
        }
        
        device.setIsActive(true);
        device.setStatus(Device.DeviceStatus.INACTIVE);
        Device savedDevice = deviceRepository.save(device);
        
        System.out.println("Registered device with ID: " + savedDevice.getId());
        return savedDevice;
    }
    
    /**
     * Update device information
     */
    @Transactional
    public Device updateDevice(Long id, Device deviceDetails) {
        System.out.println("Updating device with ID: " + id);
        
        Device device = getDeviceById(id);
        
        // Update fields if provided
        if (deviceDetails.getModel() != null && !deviceDetails.getModel().trim().isEmpty()) {
            device.setModel(deviceDetails.getModel());
        }
        if (deviceDetails.getFirmwareVersion() != null && !deviceDetails.getFirmwareVersion().trim().isEmpty()) {
            device.setFirmwareVersion(deviceDetails.getFirmwareVersion());
        }
        if (deviceDetails.getBatteryLevel() != null) {
            device.setBatteryLevel(deviceDetails.getBatteryLevel());
        }
        if (deviceDetails.getStatus() != null) {
            device.setStatus(deviceDetails.getStatus());
        }
        if (deviceDetails.getDeviceType() != null) {
            device.setDeviceType(deviceDetails.getDeviceType());
        }
        
        Device updatedDevice = deviceRepository.save(device);
        System.out.println("Updated device with ID: " + updatedDevice.getId());
        
        return updatedDevice;
    }
    
    /**
     * Assign device to patient
     */
    @Transactional
    public Device assignDeviceToPatient(Long deviceId, Long patientId) {
        System.out.println("Assigning device " + deviceId + " to patient " + patientId);
        
        Device device = getDeviceById(deviceId);
        Patient patient = patientService.getPatientById(patientId);
        
        device.setPatient(patient);
        device.setStatus(Device.DeviceStatus.ACTIVE);
        
        Device assignedDevice = deviceRepository.save(device);
        System.out.println("Assigned device " + deviceId + " to patient " + patientId);
        
        return assignedDevice;
    }
    
    /**
     * Unassign device from patient
     */
    @Transactional
    public Device unassignDeviceFromPatient(Long deviceId) {
        System.out.println("Unassigning device with ID: " + deviceId);
        
        Device device = getDeviceById(deviceId);
        device.setPatient(null);
        device.setStatus(Device.DeviceStatus.INACTIVE);
        
        Device unassignedDevice = deviceRepository.save(device);
        System.out.println("Unassigned device with ID: " + deviceId);
        
        return unassignedDevice;
    }
    
    /**
     * Update device battery level
     */
    @Transactional
    public Device updateBatteryLevel(Long deviceId, int batteryLevel) {
        Device device = getDeviceById(deviceId);
        device.setBatteryLevel(batteryLevel);
        
        // Update status based on battery level
        if (batteryLevel < 20) {
            device.setStatus(Device.DeviceStatus.LOW_BATTERY);
        } else if (device.getStatus() == Device.DeviceStatus.LOW_BATTERY) {
            device.setStatus(Device.DeviceStatus.ACTIVE);
        }
        
        return deviceRepository.save(device);
    }
    
    /**
     * Update device sync status
     */
    @Transactional
    public Device updateLastSync(Long deviceId) {
        Device device = getDeviceById(deviceId);
        device.setLastSync(LocalDateTime.now());
        
        return deviceRepository.save(device);
    }
    
    /**
     * Calibrate device
     */
    @Transactional
    public Device calibrateDevice(Long deviceId) {
        System.out.println("Calibrating device with ID: " + deviceId);
        
        Device device = getDeviceById(deviceId);
        device.setIsCalibrated(true);
        device.setCalibrationDate(LocalDateTime.now());
        
        Device calibratedDevice = deviceRepository.save(device);
        System.out.println("Calibrated device with ID: " + deviceId);
        
        return calibratedDevice;
    }
    
    /**
     * Get device by ID
     */
    public Device getDeviceById(Long id) {
        return deviceRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Device not found with ID: " + id));
    }
    
    /**
     * Get device by serial number
     */
    public Optional<Device> getDeviceBySerialNumber(String serialNumber) {
        return deviceRepository.findBySerialNumber(serialNumber);
    }
    
    /**
     * Get all active devices
     */
    public List<Device> getAllActiveDevices() {
        return deviceRepository.findByIsActiveTrue();
    }
    
    /**
     * Get devices by patient
     */
    public List<Device> getDevicesByPatient(Long patientId) {
        return deviceRepository.findByPatientIdAndIsActiveTrue(patientId);
    }
    
    /**
     * Get devices by status
     */
    public List<Device> getDevicesByStatus(Device.DeviceStatus status) {
        return deviceRepository.findByStatusAndIsActiveTrue(status);
    }
    
    /**
     * Get devices with low battery
     */
    public List<Device> getDevicesWithLowBattery() {
        return deviceRepository.findDevicesWithLowBattery(20);
    }
    
    /**
     * Get devices requiring calibration
     */
    public List<Device> getDevicesRequiringCalibration() {
        LocalDateTime threshold = LocalDateTime.now().minusDays(30);
        return deviceRepository.findDevicesRequiringCalibration(threshold);
    }
    
    /**
     * Get offline devices
     */
    public List<Device> getOfflineDevices() {
        LocalDateTime threshold = LocalDateTime.now().minusMinutes(5);
        return deviceRepository.findOfflineDevices(threshold);
    }
    
    /**
     * Deactivate device
     */
    @Transactional
    public void deactivateDevice(Long id) {
        System.out.println("Deactivating device with ID: " + id);
        
        Device device = getDeviceById(id);
        device.setIsActive(false);
        device.setStatus(Device.DeviceStatus.INACTIVE);
        deviceRepository.save(device);
        
        System.out.println("Deactivated device with ID: " + id);
    }
    
    /**
     * Get device statistics
     */
    public DeviceStatistics getDeviceStatistics() {
        long totalActive = deviceRepository.countByIsActiveTrue();
        long activeCount = deviceRepository.countByStatusAndIsActiveTrue(Device.DeviceStatus.ACTIVE);
        long assignedCount = deviceRepository.countByPatientIsNotNullAndIsActiveTrue();
        
        return DeviceStatistics.builder()
                .totalActiveDevices(totalActive)
                .activeDevices(activeCount)
                .assignedDevices(assignedCount)
                .unassignedDevices(totalActive - assignedCount)
                .build();
    }
    
    /**
     * Statistics DTO
     */
    public static class DeviceStatistics {
        private long totalActiveDevices;
        private long activeDevices;
        private long assignedDevices;
        private long unassignedDevices;
        
        public DeviceStatistics() {}
        
        public static DeviceStatistics builder() {
            return new DeviceStatistics();
        }
        
        public DeviceStatistics totalActiveDevices(long totalActiveDevices) {
            this.totalActiveDevices = totalActiveDevices;
            return this;
        }
        
        public DeviceStatistics activeDevices(long activeDevices) {
            this.activeDevices = activeDevices;
            return this;
        }
        
        public DeviceStatistics assignedDevices(long assignedDevices) {
            this.assignedDevices = assignedDevices;
            return this;
        }
        
        public DeviceStatistics unassignedDevices(long unassignedDevices) {
            this.unassignedDevices = unassignedDevices;
            return this;
        }
        
        public DeviceStatistics build() {
            return this;
        }
        
        public long getTotalActiveDevices() { return totalActiveDevices; }
        public void setTotalActiveDevices(long totalActiveDevices) { this.totalActiveDevices = totalActiveDevices; }
        public long getActiveDevices() { return activeDevices; }
        public void setActiveDevices(long activeDevices) { this.activeDevices = activeDevices; }
        public long getAssignedDevices() { return assignedDevices; }
        public void setAssignedDevices(long assignedDevices) { this.assignedDevices = assignedDevices; }
        public long getUnassignedDevices() { return unassignedDevices; }
        public void setUnassignedDevices(long unassignedDevices) { this.unassignedDevices = unassignedDevices; }
    }
}