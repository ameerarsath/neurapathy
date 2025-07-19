package com.smartshoe.api.controller;

import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.service.PatientService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * REST Controller for Patient management operations
 */
@RestController
@RequestMapping("/api/patients")
@Tag(name = "Patient Management", description = "API for managing diabetic patients")
public class PatientController {
    
    private final PatientService patientService;
    
    public PatientController(PatientService patientService) {
        this.patientService = patientService;
    }
    
    /**
     * Create a new patient
     */
    @PostMapping
    @Operation(summary = "Create a new patient", description = "Register a new diabetic patient in the system")
    public ResponseEntity<Map<String, Object>> createPatient(@Valid @RequestBody Patient patient) {
        System.out.println("Creating new patient with email: " + patient.getEmail());
        
        try {
            Patient savedPatient = patientService.createPatient(patient);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Patient created successfully");
            response.put("patient", savedPatient);
            
            return ResponseEntity.status(HttpStatus.CREATED).body(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error creating patient: " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get patient by ID
     */
    @GetMapping("/{id}")
    @Operation(summary = "Get patient by ID", description = "Retrieve patient information by patient ID")
    public ResponseEntity<Map<String, Object>> getPatientById(
            @Parameter(description = "Patient ID") @PathVariable Long id) {
        
        try {
            Patient patient = patientService.getPatientById(id);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("patient", patient);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error finding patient with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.notFound().build();
        }
    }
    
    /**
     * Get all active patients
     */
    @GetMapping
    @Operation(summary = "Get all active patients", description = "Retrieve all active patients in the system")
    public ResponseEntity<Map<String, Object>> getAllActivePatients(
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        List<Patient> patients = patientService.getAllActivePatients();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("patients", patients);
        response.put("total", patients.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Search patients by name
     */
    @GetMapping("/search")
    @Operation(summary = "Search patients by name", description = "Search for patients by first or last name")
    public ResponseEntity<Map<String, Object>> searchPatients(
            @RequestParam String name,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "10") int size) {
        
        Pageable pageable = PageRequest.of(page, size);
        Page<Patient> patients = patientService.searchPatientsByName(name, pageable);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("patients", patients.getContent());
        response.put("total", patients.getTotalElements());
        response.put("pages", patients.getTotalPages());
        response.put("currentPage", page);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Get patients by diabetes type
     */
    @GetMapping("/diabetes-type/{type}")
    @Operation(summary = "Get patients by diabetes type", description = "Retrieve patients filtered by diabetes type")
    public ResponseEntity<Map<String, Object>> getPatientsByDiabetesType(
            @Parameter(description = "Diabetes type") @PathVariable Patient.DiabetesType type) {
        
        List<Patient> patients = patientService.getPatientsByDiabetesType(type);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("patients", patients);
        response.put("diabetesType", type);
        response.put("total", patients.size());
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Update patient information
     */
    @PutMapping("/{id}")
    @Operation(summary = "Update patient", description = "Update patient information")
    public ResponseEntity<Map<String, Object>> updatePatient(
            @Parameter(description = "Patient ID") @PathVariable Long id,
            @Valid @RequestBody Patient patientDetails) {
        
        try {
            Patient updatedPatient = patientService.updatePatient(id, patientDetails);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Patient updated successfully");
            response.put("patient", updatedPatient);
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error updating patient with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Deactivate patient
     */
    @DeleteMapping("/{id}")
    @Operation(summary = "Deactivate patient", description = "Deactivate a patient (soft delete)")
    public ResponseEntity<Map<String, Object>> deactivatePatient(
            @Parameter(description = "Patient ID") @PathVariable Long id) {
        
        try {
            patientService.deactivatePatient(id);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "Patient deactivated successfully");
            
            return ResponseEntity.ok(response);
            
        } catch (RuntimeException e) {
            System.err.println("Error deactivating patient with ID " + id + ": " + e.getMessage());
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", false);
            response.put("message", e.getMessage());
            
            return ResponseEntity.badRequest().body(response);
        }
    }
    
    /**
     * Get patient statistics
     */
    @GetMapping("/statistics")
    @Operation(summary = "Get patient statistics", description = "Retrieve patient statistics and demographics")
    public ResponseEntity<Map<String, Object>> getPatientStatistics() {
        
        PatientService.PatientStatistics stats = patientService.getPatientStatistics();
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("statistics", stats);
        
        return ResponseEntity.ok(response);
    }
    
    /**
     * Check if email exists
     */
    @GetMapping("/check-email")
    @Operation(summary = "Check email availability", description = "Check if an email address is already registered")
    public ResponseEntity<Map<String, Object>> checkEmailExists(@RequestParam String email) {
        
        boolean exists = patientService.emailExists(email);
        
        Map<String, Object> response = new HashMap<>();
        response.put("success", true);
        response.put("emailExists", exists);
        response.put("available", !exists);
        
        return ResponseEntity.ok(response);
    }
}