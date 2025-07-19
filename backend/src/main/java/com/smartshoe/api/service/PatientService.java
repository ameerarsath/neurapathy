package com.smartshoe.api.service;

import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.repository.PatientRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

/**
 * Service layer for Patient entity operations
 */
@Service
@Transactional(readOnly = true)
public class PatientService {
    
    private final PatientRepository patientRepository;
    
    public PatientService(PatientRepository patientRepository) {
        this.patientRepository = patientRepository;
    }
    
    /**
     * Create a new patient
     */
    @Transactional
    public Patient createPatient(Patient patient) {
        System.out.println("Creating new patient with email: " + patient.getEmail());
        
        if (patientRepository.existsByEmail(patient.getEmail())) {
            throw new RuntimeException("Patient with email " + patient.getEmail() + " already exists");
        }
        
        patient.setIsActive(true);
        Patient savedPatient = patientRepository.save(patient);
        
        System.out.println("Created patient with ID: " + savedPatient.getId());
        return savedPatient;
    }
    
    /**
     * Update an existing patient
     */
    @Transactional
    public Patient updatePatient(Long id, Patient patientDetails) {
        System.out.println("Updating patient with ID: " + id);
        
        Patient patient = getPatientById(id);
        
        // Update fields
        patient.setFirstName(patientDetails.getFirstName());
        patient.setLastName(patientDetails.getLastName());
        patient.setPhoneNumber(patientDetails.getPhoneNumber());
        patient.setDiabetesType(patientDetails.getDiabetesType());
        patient.setDiagnosisDate(patientDetails.getDiagnosisDate());
        
        Patient updatedPatient = patientRepository.save(patient);
        System.out.println("Updated patient with ID: " + updatedPatient.getId());
        
        return updatedPatient;
    }
    
    /**
     * Get patient by ID
     */
    public Patient getPatientById(Long id) {
        return patientRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Patient not found with ID: " + id));
    }
    
    /**
     * Get patient by email
     */
    public Optional<Patient> getPatientByEmail(String email) {
        return patientRepository.findByEmail(email);
    }
    
    /**
     * Get all active patients
     */
    public List<Patient> getAllActivePatients() {
        return patientRepository.findByIsActiveTrue();
    }
    
    /**
     * Search patients by name
     */
    public Page<Patient> searchPatientsByName(String name, Pageable pageable) {
        return patientRepository.findByNameContainingIgnoreCase(name, pageable);
    }
    
    /**
     * Get patients by diabetes type
     */
    public List<Patient> getPatientsByDiabetesType(Patient.DiabetesType diabetesType) {
        return patientRepository.findByDiabetesTypeAndIsActiveTrue(diabetesType);
    }
    
    /**
     * Get patients by age range
     */
    public List<Patient> getPatientsByAgeRange(int minAge, int maxAge) {
        return patientRepository.findByAgeRange(minAge, maxAge);
    }
    
    /**
     * Get patients diagnosed within date range
     */
    public List<Patient> getPatientsByDiagnosisDateRange(LocalDate startDate, LocalDate endDate) {
        return patientRepository.findByDiagnosisDateBetweenAndIsActiveTrue(startDate, endDate);
    }
    
    /**
     * Deactivate patient (soft delete)
     */
    @Transactional
    public void deactivatePatient(Long id) {
        System.out.println("Deactivating patient with ID: " + id);
        
        Patient patient = getPatientById(id);
        patient.setIsActive(false);
        patientRepository.save(patient);
        
        System.out.println("Deactivated patient with ID: " + id);
    }
    
    /**
     * Reactivate patient
     */
    @Transactional
    public void reactivatePatient(Long id) {
        System.out.println("Reactivating patient with ID: " + id);
        
        Patient patient = patientRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Patient not found with ID: " + id));
        
        patient.setIsActive(true);
        patientRepository.save(patient);
        
        System.out.println("Reactivated patient with ID: " + id);
    }
    
    /**
     * Check if email exists
     */
    public boolean emailExists(String email) {
        return patientRepository.existsByEmail(email);
    }
    
    /**
     * Get patient statistics
     */
    public PatientStatistics getPatientStatistics() {
        long totalActive = patientRepository.countByIsActiveTrue();
        long type1Count = patientRepository.countByDiabetesTypeAndIsActiveTrue(Patient.DiabetesType.TYPE_1);
        long type2Count = patientRepository.countByDiabetesTypeAndIsActiveTrue(Patient.DiabetesType.TYPE_2);
        
        return PatientStatistics.builder()
                .totalActivePatients(totalActive)
                .type1DiabetesCount(type1Count)
                .type2DiabetesCount(type2Count)
                .build();
    }
    
    /**
     * Statistics DTO
     */
    public static class PatientStatistics {
        private long totalActivePatients;
        private long type1DiabetesCount;
        private long type2DiabetesCount;
        
        public PatientStatistics() {}
        
        public static PatientStatistics builder() {
            return new PatientStatistics();
        }
        
        public PatientStatistics totalActivePatients(long totalActivePatients) {
            this.totalActivePatients = totalActivePatients;
            return this;
        }
        
        public PatientStatistics type1DiabetesCount(long type1DiabetesCount) {
            this.type1DiabetesCount = type1DiabetesCount;
            return this;
        }
        
        public PatientStatistics type2DiabetesCount(long type2DiabetesCount) {
            this.type2DiabetesCount = type2DiabetesCount;
            return this;
        }
        
        public PatientStatistics build() {
            return this;
        }
        
        public long getTotalActivePatients() { return totalActivePatients; }
        public void setTotalActivePatients(long totalActivePatients) { this.totalActivePatients = totalActivePatients; }
        public long getType1DiabetesCount() { return type1DiabetesCount; }
        public void setType1DiabetesCount(long type1DiabetesCount) { this.type1DiabetesCount = type1DiabetesCount; }
        public long getType2DiabetesCount() { return type2DiabetesCount; }
        public void setType2DiabetesCount(long type2DiabetesCount) { this.type2DiabetesCount = type2DiabetesCount; }
    }
}