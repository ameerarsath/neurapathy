package com.smartshoe.api.controller;

import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.service.MedicalReadingService;
import com.smartshoe.api.service.ExportService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.data.domain.Page;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for Medical Reading operations
 */
@RestController
@RequestMapping("/api/medical-readings")
@Tag(name = "Medical Reading Management", description = "API for managing medical sensor readings")
public class MedicalReadingController {
    
    private final MedicalReadingService medicalReadingService;
    private final ExportService exportService;
    
    public MedicalReadingController(MedicalReadingService medicalReadingService, ExportService exportService) {
        this.medicalReadingService = medicalReadingService;
        this.exportService = exportService;
    }
    
    /**
     * Get all medical readings
     */
    @GetMapping
    @Operation(summary = "Get all medical readings", description = "Retrieve all medical readings with pagination")
    public ResponseEntity<Map<String, Object>> getAllReadings(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getAllReadings(page, size);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("data", readings.getContent());
            response.put("total", readings.getTotalElements());
            response.put("pages", readings.getTotalPages());
            response.put("currentPage", page);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting all medical readings: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            response.put("data", new java.util.ArrayList<>());
            
            return ResponseEntity.ok(response);
        }
    }
    
    /**
     * Record a new medical reading
     */
    @PostMapping
    @Operation(summary = "Record medical reading", description = "Record a new medical reading from sensor data")
    public ResponseEntity<Map<String, Object>> recordReading(@Valid @RequestBody MedicalReading reading) {
        System.out.println("Recording new medical reading for patient ID: " + reading.getPatient().getId());
        
        try {
            MedicalReading savedReading = medicalReadingService.recordReading(reading);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Medical reading recorded successfully");
            response.put("reading", savedReading);
            
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error recording medical reading: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Record sensor data
     */
    @PostMapping("/sensor-data")
    @Operation(summary = "Record sensor data", description = "Record sensor data with simplified input")
    public ResponseEntity<Map<String, Object>> recordSensorData(
            @RequestParam Long patientId,
            @RequestParam Long deviceId,
            @RequestParam MedicalReading.ReadingType readingType,
            @RequestParam double value,
            @RequestParam String unit) {
        
        System.out.println("Recording sensor data for patient " + patientId + " from device " + deviceId);
        
        try {
            MedicalReading savedReading = medicalReadingService.recordSensorData(patientId, deviceId, readingType, value, unit);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Sensor data recorded successfully");
            response.put("reading", savedReading);
            
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error recording sensor data: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get reading by ID
     */
    @GetMapping("/{id}")
    @Operation(summary = "Get medical reading by ID", description = "Retrieve medical reading by reading ID")
    public ResponseEntity<Map<String, Object>> getReadingById(
            @Parameter(description = "Reading ID") @PathVariable Long id) {
        
        try {
            MedicalReading reading = medicalReadingService.getReadingById(id);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("reading", reading);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error finding reading with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.notFound().build();
        }
    }
    
    /**
     * Get readings by patient
     */
    @GetMapping("/patient/{patientId}")
    @Operation(summary = "Get readings by patient", description = "Retrieve medical readings for a specific patient")
    public ResponseEntity<Map<String, Object>> getReadingsByPatient(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getReadingsByPatient(patientId, page, size);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("readings", readings.getContent());
            response.put("total", readings.getTotalElements());
            response.put("pages", readings.getTotalPages());
            response.put("currentPage", page);
            response.put("patientId", patientId);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting readings for patient " + patientId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get readings by patient and type
     */
    @GetMapping("/patient/{patientId}/type/{readingType}")
    @Operation(summary = "Get readings by patient and type", description = "Retrieve medical readings for a patient filtered by reading type")
    public ResponseEntity<Map<String, Object>> getReadingsByPatientAndType(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Reading type") @PathVariable MedicalReading.ReadingType readingType) {
        
        try {
            List<MedicalReading> readings = medicalReadingService.getReadingsByPatientAndType(patientId, readingType);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("readings", readings);
            response.put("total", readings.size());
            response.put("patientId", patientId);
            response.put("readingType", readingType);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting readings for patient " + patientId + " and type " + readingType + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get readings by device
     */
    @GetMapping("/device/{deviceId}")
    @Operation(summary = "Get readings by device", description = "Retrieve medical readings from a specific device")
    public ResponseEntity<Map<String, Object>> getReadingsByDevice(
            @Parameter(description = "Device ID") @PathVariable Long deviceId) {
        
        try {
            List<MedicalReading> readings = medicalReadingService.getReadingsByDevice(deviceId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("readings", readings);
            response.put("total", readings.size());
            response.put("deviceId", deviceId);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting readings for device " + deviceId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get readings within date range
     */
    @GetMapping("/date-range")
    @Operation(summary = "Get readings in date range", description = "Retrieve medical readings within a specific date range")
    public ResponseEntity<Map<String, Object>> getReadingsInDateRange(
            @Parameter(description = "Start date") @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        List<MedicalReading> readings = medicalReadingService.getReadingsInDateRange(startDate, endDate);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("readings", readings);
        response.put("total", readings.size());
        response.put("startDate", startDate);
        response.put("endDate", endDate);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get patient readings within date range
     */
    @GetMapping("/patient/{patientId}/date-range")
    @Operation(summary = "Get patient readings in date range", description = "Retrieve patient readings within a specific date range")
    public ResponseEntity<Map<String, Object>> getPatientReadingsInDateRange(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Start date") @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        try {
            List<MedicalReading> readings = medicalReadingService.getPatientReadingsInDateRange(patientId, startDate, endDate);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("readings", readings);
            response.put("total", readings.size());
            response.put("patientId", patientId);
            response.put("startDate", startDate);
            response.put("endDate", endDate);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting patient readings in date range: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get abnormal readings
     */
    @GetMapping("/abnormal")
    @Operation(summary = "Get abnormal readings", description = "Retrieve all abnormal medical readings")
    public ResponseEntity<Map<String, Object>> getAbnormalReadings() {
        
        List<MedicalReading> readings = medicalReadingService.getAbnormalReadings();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("readings", readings);
        response.put("total", readings.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get critical readings
     */
    @GetMapping("/critical")
    @Operation(summary = "Get critical readings", description = "Retrieve critical readings requiring immediate attention")
    public ResponseEntity<Map<String, Object>> getCriticalReadings() {
        
        List<MedicalReading> readings = medicalReadingService.getCriticalReadings();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("readings", readings);
        response.put("total", readings.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get baseline readings for a patient
     */
    @GetMapping("/patient/{patientId}/baseline")
    @Operation(summary = "Get baseline readings", description = "Retrieve baseline readings for a patient")
    public ResponseEntity<Map<String, Object>> getBaselineReadings(
            @Parameter(description = "Patient ID") @PathVariable Long patientId) {
        
        try {
            List<MedicalReading> readings = medicalReadingService.getBaselineReadings(patientId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("readings", readings);
            response.put("total", readings.size());
            response.put("patientId", patientId);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting baseline readings for patient " + patientId + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get high quality readings
     */
    @GetMapping("/high-quality")
    @Operation(summary = "Get high quality readings", description = "Retrieve high quality readings above threshold")
    public ResponseEntity<Map<String, Object>> getHighQualityReadings(
            @RequestParam(defaultValue = "80.0") double qualityThreshold) {
        
        List<MedicalReading> readings = medicalReadingService.getHighQualityReadings(qualityThreshold);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("readings", readings);
        response.put("total", readings.size());
        response.put("qualityThreshold", qualityThreshold);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get latest reading by patient and type
     */
    @GetMapping("/patient/{patientId}/latest/{readingType}")
    @Operation(summary = "Get latest reading", description = "Get the latest reading for a patient by type")
    public ResponseEntity<Map<String, Object>> getLatestReadingByPatientAndType(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @Parameter(description = "Reading type") @PathVariable MedicalReading.ReadingType readingType) {
        
        try {
            MedicalReading reading = medicalReadingService.getLatestReadingByPatientAndType(patientId, readingType);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("reading", reading);
            response.put("patientId", patientId);
            response.put("readingType", readingType);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting latest reading for patient " + patientId + " and type " + readingType + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Mark reading as baseline
     */
    @PostMapping("/{readingId}/baseline")
    @Operation(summary = "Mark as baseline", description = "Mark a reading as baseline for future comparisons")
    public ResponseEntity<Map<String, Object>> markAsBaseline(
            @Parameter(description = "Reading ID") @PathVariable Long readingId) {
        
        try {
            MedicalReading updatedReading = medicalReadingService.markAsBaseline(readingId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Reading marked as baseline successfully");
            response.put("reading", updatedReading);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error marking reading as baseline: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Update reading severity
     */
    @PatchMapping("/{readingId}/severity")
    @Operation(summary = "Update severity", description = "Update the severity level of a reading")
    public ResponseEntity<Map<String, Object>> updateSeverity(
            @Parameter(description = "Reading ID") @PathVariable Long readingId,
            @RequestParam MedicalReading.SeverityLevel severity) {
        
        try {
            MedicalReading updatedReading = medicalReadingService.updateSeverity(readingId, severity);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Severity updated successfully");
            response.put("reading", updatedReading);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error updating reading severity: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Add provider notes
     */
    @PatchMapping("/{readingId}/notes")
    @Operation(summary = "Add provider notes", description = "Add clinical notes from healthcare provider")
    public ResponseEntity<Map<String, Object>> addProviderNotes(
            @Parameter(description = "Reading ID") @PathVariable Long readingId,
            @RequestParam String notes) {
        
        try {
            MedicalReading updatedReading = medicalReadingService.addProviderNotes(readingId, notes);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Provider notes added successfully");
            response.put("reading", updatedReading);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error adding provider notes: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get reading statistics
     */
    @GetMapping("/statistics")
    @Operation(summary = "Get reading statistics", description = "Retrieve overall medical reading statistics")
    public ResponseEntity<Map<String, Object>> getReadingStatistics() {
        
        MedicalReadingService.ReadingStatistics stats = medicalReadingService.getReadingStatistics();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("statistics", stats);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get patient reading statistics
     */
    @GetMapping("/patient/{patientId}/statistics")
    @Operation(summary = "Get patient reading statistics", description = "Retrieve reading statistics for a specific patient")
    public ResponseEntity<Map<String, Object>> getPatientReadingStatistics(
            @Parameter(description = "Patient ID") @PathVariable Long patientId) {
        
        try {
            MedicalReadingService.PatientReadingStatistics stats = medicalReadingService.getPatientReadingStatistics(patientId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("statistics", stats);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error getting patient reading statistics: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }

    /**
     * Export all medical readings to CSV
     */
    @GetMapping("/export/csv")
    @Operation(summary = "Export readings to CSV", description = "Export medical readings to CSV format")
    public ResponseEntity<String> exportToCSV(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "1000") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getAllReadings(page, size);
            String csvData = exportService.exportToCSV(readings.getContent());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.TEXT_PLAIN);
            headers.setContentDispositionFormData("attachment", 
                "medical_readings_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")) + ".csv");
            
            return ResponseEntity.ok()
                .headers(headers)
                .body(csvData);
                
        } catch (Exception e) {
            System.err.println("Error exporting to CSV: " + e.getMessage());
            return ResponseEntity.internalServerError().body("Error generating CSV export");
        }
    }

    /**
     * Export all medical readings to Excel
     */
    @GetMapping("/export/excel")
    @Operation(summary = "Export readings to Excel", description = "Export medical readings to Excel format")
    public ResponseEntity<byte[]> exportToExcel(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "1000") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getAllReadings(page, size);
            byte[] excelData = exportService.exportToExcel(readings.getContent());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_OCTET_STREAM);
            headers.setContentDispositionFormData("attachment", 
                "medical_readings_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")) + ".xlsx");
            
            return ResponseEntity.ok()
                .headers(headers)
                .body(excelData);
                
        } catch (Exception e) {
            System.err.println("Error exporting to Excel: " + e.getMessage());
            return ResponseEntity.internalServerError().body(new byte[0]);
        }
    }

    /**
     * Export all medical readings to PDF
     */
    @GetMapping("/export/pdf")
    @Operation(summary = "Export readings to PDF", description = "Export medical readings to PDF format")
    public ResponseEntity<byte[]> exportToPDF(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "1000") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getAllReadings(page, size);
            byte[] pdfData = exportService.exportToPDF(readings.getContent());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_PDF);
            headers.setContentDispositionFormData("attachment", 
                "medical_readings_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")) + ".pdf");
            
            return ResponseEntity.ok()
                .headers(headers)
                .body(pdfData);
                
        } catch (Exception e) {
            System.err.println("Error exporting to PDF: " + e.getMessage());
            return ResponseEntity.internalServerError().body(new byte[0]);
        }
    }

    /**
     * Export patient readings to CSV
     */
    @GetMapping("/patient/{patientId}/export/csv")
    @Operation(summary = "Export patient readings to CSV", description = "Export patient medical readings to CSV format")
    public ResponseEntity<String> exportPatientToCSV(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "1000") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getReadingsByPatient(patientId, page, size);
            String csvData = exportService.exportToCSV(readings.getContent());
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.TEXT_PLAIN);
            headers.setContentDispositionFormData("attachment", 
                "patient_" + patientId + "_readings_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")) + ".csv");
            
            return ResponseEntity.ok()
                .headers(headers)
                .body(csvData);
                
        } catch (Exception e) {
            System.err.println("Error exporting patient readings to CSV: " + e.getMessage());
            return ResponseEntity.internalServerError().body("Error generating CSV export");
        }
    }

    /**
     * Export patient readings to PDF
     */
    @GetMapping("/patient/{patientId}/export/pdf")
    @Operation(summary = "Export patient readings to PDF", description = "Export patient medical readings to PDF format")
    public ResponseEntity<byte[]> exportPatientToPDF(
            @Parameter(description = "Patient ID") @PathVariable Long patientId,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "1000") int size) {
        
        try {
            Page<MedicalReading> readings = medicalReadingService.getReadingsByPatient(patientId, page, size);
            String patientName = readings.getContent().isEmpty() ? "Unknown" : 
                readings.getContent().get(0).getPatient().getFullName();
            
            byte[] pdfData = exportService.exportPatientSummaryToPDF(readings.getContent(), patientName);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_PDF);
            headers.setContentDispositionFormData("attachment", 
                "patient_" + patientId + "_summary_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss")) + ".pdf");
            
            return ResponseEntity.ok()
                .headers(headers)
                .body(pdfData);
                
        } catch (Exception e) {
            System.err.println("Error exporting patient readings to PDF: " + e.getMessage());
            return ResponseEntity.internalServerError().body(new byte[0]);
        }
    }

    /**
     * Export readings within date range to CSV
     */
    @GetMapping("/export/csv/date-range")
    @Operation(summary = "Export readings to CSV by date range", description = "Export medical readings within date range to CSV format")
    public ResponseEntity<String> exportDateRangeToCSV(
            @Parameter(description = "Start date") @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @Parameter(description = "End date") @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {
        
        try {
            List<MedicalReading> readings = medicalReadingService.getReadingsInDateRange(startDate, endDate);
            String csvData = exportService.exportToCSV(readings);
            
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.TEXT_PLAIN);
            headers.setContentDispositionFormData("attachment", 
                "readings_" + startDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")) + "_to_" + 
                endDate.format(DateTimeFormatter.ofPattern("yyyy-MM-dd")) + ".csv");
            
            return ResponseEntity.ok()
                .headers(headers)
                .body(csvData);
                
        } catch (Exception e) {
            System.err.println("Error exporting date range to CSV: " + e.getMessage());
            return ResponseEntity.internalServerError().body("Error generating CSV export");
        }
    }
}