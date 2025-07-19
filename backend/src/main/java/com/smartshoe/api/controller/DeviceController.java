package com.smartshoe.api.controller;

import com.smartshoe.api.entity.Device;
import com.smartshoe.api.service.DeviceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for Device management operations
 */
@RestController
@RequestMapping("/api/devices")
@Tag(name = "Device Management", description = "API for managing smart shoe devices")
public class DeviceController {
    
    private final DeviceService deviceService;
    
    public DeviceController(DeviceService deviceService) {
        this.deviceService = deviceService;
    }
    
    /**
     * Register a new device
     */
    @PostMapping
    @Operation(summary = "Register a new device", description = "Register a new smart shoe device in the system")
    public ResponseEntity<Map<String, Object>> registerDevice(@Valid @RequestBody Device device) {
        System.out.println("Registering new device with serial number: " + device.getSerialNumber());
        
        try {
            Device savedDevice = deviceService.registerDevice(device);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Device registered successfully");
            response.put("device", savedDevice);
            
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error registering device: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get device by ID
     */
    @GetMapping("/{id}")
    @Operation(summary = "Get device by ID", description = "Retrieve device information by device ID")
    public ResponseEntity<Map<String, Object>> getDeviceById(
            @Parameter(description = "Device ID") @PathVariable Long id) {
        
        try {
            Device device = deviceService.getDeviceById(id);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("device", device);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error finding device with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.notFound().build();
        }
    }
    
    /**
     * Get device by serial number
     */
    @GetMapping("/serial/{serialNumber}")
    @Operation(summary = "Get device by serial number", description = "Retrieve device information by serial number")
    public ResponseEntity<Map<String, Object>> getDeviceBySerialNumber(
            @Parameter(description = "Device serial number") @PathVariable String serialNumber) {
        
        return deviceService.getDeviceBySerialNumber(serialNumber)
                .map(device -> {
                    Map<String, Object> response = new HashMap<>();
                    response.put("success", true);
                    response.put("device", device);
                    return ResponseEntity.ok(response);
                })
                .orElseGet(() -> {
                    Map<String, Object> response = new HashMap<>();
                    response.put("success", false);
                    response.put("message", "Device not found with serial number: " + serialNumber);
                    return ResponseEntity.notFound().build();
                });
    }
    
    /**
     * Get all active devices
     */
    @GetMapping
    @Operation(summary = "Get all active devices", description = "Retrieve all active devices in the system")
    public ResponseEntity<Map<String, Object>> getAllActiveDevices() {
        
        List<Device> devices = deviceService.getAllActiveDevices();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("devices", devices);
        response.put("total", devices.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get devices by patient
     */
    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get devices by patient", description = "Retrieve all devices assigned to a specific patient")
    public ResponseEntity<Map<String, Object>> getDevicesByPatient(
            @Parameter(description = "Patient ID") @PathVariable Long patientId) {
        
        List<Device> devices = deviceService.getDevicesByPatient(patientId);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("devices", devices);
        response.put("patientId", patientId);
        response.put("total", devices.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get devices by status
     */
    @GetMapping("/status/{status}")
    @Operation(summary = "Get devices by status", description = "Retrieve devices filtered by status")
    public ResponseEntity<Map<String, Object>> getDevicesByStatus(
            @Parameter(description = "Device status") @PathVariable Device.DeviceStatus status) {
        
        List<Device> devices = deviceService.getDevicesByStatus(status);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("devices", devices);
        response.put("status", status);
        response.put("total", devices.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get devices with low battery
     */
    @GetMapping("/low-battery")
    @Operation(summary = "Get devices with low battery", description = "Retrieve devices with battery level below 20%")
    public ResponseEntity<Map<String, Object>> getDevicesWithLowBattery() {
        
        List<Device> devices = deviceService.getDevicesWithLowBattery();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("devices", devices);
        response.put("total", devices.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get devices requiring calibration
     */
    @GetMapping("/require-calibration")
    @Operation(summary = "Get devices requiring calibration", description = "Retrieve devices that need calibration")
    public ResponseEntity<Map<String, Object>> getDevicesRequiringCalibration() {
        
        List<Device> devices = deviceService.getDevicesRequiringCalibration();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("devices", devices);
        response.put("total", devices.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get offline devices
     */
    @GetMapping("/offline")
    @Operation(summary = "Get offline devices", description = "Retrieve devices that haven't synced recently")
    public ResponseEntity<Map<String, Object>> getOfflineDevices() {
        
        List<Device> devices = deviceService.getOfflineDevices();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("devices", devices);
        response.put("total", devices.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Update device information
     */
    @PutMapping("/{id}")
    @Operation(summary = "Update device", description = "Update device information")
    public ResponseEntity<Map<String, Object>> updateDevice(
            @Parameter(description = "Device ID") @PathVariable Long id,
            @Valid @RequestBody Device deviceDetails) {
        
        try {
            Device updatedDevice = deviceService.updateDevice(id, deviceDetails);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Device updated successfully");
            response.put("device", updatedDevice);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error updating device with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Assign device to patient
     */
    @PostMapping("/{deviceId}/assign/{patientId}")
    @Operation(summary = "Assign device to patient", description = "Assign a device to a patient")
    public ResponseEntity<Map<String, Object>> assignDeviceToPatient(
            @Parameter(description = "Device ID") @PathVariable Long deviceId,
            @Parameter(description = "Patient ID") @PathVariable Long patientId) {
        
        try {
            Device assignedDevice = deviceService.assignDeviceToPatient(deviceId, patientId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Device assigned to patient successfully");
            response.put("device", assignedDevice);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error assigning device " + deviceId + " to patient " + patientId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Unassign device from patient
     */
    @PostMapping("/{deviceId}/unassign")
    @Operation(summary = "Unassign device from patient", description = "Remove device assignment from patient")
    public ResponseEntity<Map<String, Object>> unassignDeviceFromPatient(
            @Parameter(description = "Device ID") @PathVariable Long deviceId) {
        
        try {
            Device unassignedDevice = deviceService.unassignDeviceFromPatient(deviceId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Device unassigned from patient successfully");
            response.put("device", unassignedDevice);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error unassigning device " + deviceId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Update device battery level
     */
    @PatchMapping("/{deviceId}/battery")
    @Operation(summary = "Update battery level", description = "Update device battery level")
    public ResponseEntity<Map<String, Object>> updateBatteryLevel(
            @Parameter(description = "Device ID") @PathVariable Long deviceId,
            @RequestParam int batteryLevel) {
        
        try {
            Device updatedDevice = deviceService.updateBatteryLevel(deviceId, batteryLevel);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Battery level updated successfully");
            response.put("device", updatedDevice);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error updating battery level for device " + deviceId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Calibrate device
     */
    @PostMapping("/{deviceId}/calibrate")
    @Operation(summary = "Calibrate device", description = "Perform device calibration")
    public ResponseEntity<Map<String, Object>> calibrateDevice(
            @Parameter(description = "Device ID") @PathVariable Long deviceId) {
        
        try {
            Device calibratedDevice = deviceService.calibrateDevice(deviceId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Device calibrated successfully");
            response.put("device", calibratedDevice);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error calibrating device " + deviceId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Update device sync status
     */
    @PostMapping("/{deviceId}/sync")
    @Operation(summary = "Update sync status", description = "Update device last sync timestamp")
    public ResponseEntity<Map<String, Object>> updateLastSync(
            @Parameter(description = "Device ID") @PathVariable Long deviceId) {
        
        try {
            Device updatedDevice = deviceService.updateLastSync(deviceId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Sync status updated successfully");
            response.put("device", updatedDevice);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error updating sync status for device " + deviceId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Deactivate device
     */
    @DeleteMapping("/{id}")
    @Operation(summary = "Deactivate device", description = "Deactivate a device (soft delete)")
    public ResponseEntity<Map<String, Object>> deactivateDevice(
            @Parameter(description = "Device ID") @PathVariable Long id) {
        
        try {
            deviceService.deactivateDevice(id);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Device deactivated successfully");
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error deactivating device with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get device statistics
     */
    @GetMapping("/statistics")
    @Operation(summary = "Get device statistics", description = "Retrieve device statistics and analytics")
    public ResponseEntity<Map<String, Object>> getDeviceStatistics() {
        
        DeviceService.DeviceStatistics stats = deviceService.getDeviceStatistics();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("statistics", stats);
        
        return ResponseEntity.ok(response);
    }
}