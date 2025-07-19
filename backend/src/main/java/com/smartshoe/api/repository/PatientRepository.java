package com.smartshoe.api.repository;

import com.smartshoe.api.entity.Patient;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for Patient entity operations
 */
@Repository
public interface PatientRepository extends JpaRepository<Patient, Long> {
    
    /**
     * Find patient by email address
     */
    Optional<Patient> findByEmail(String email);
    
    /**
     * Find active patients
     */
    List<Patient> findByIsActiveTrue();
    
    /**
     * Find patients by diabetes type
     */
    List<Patient> findByDiabetesTypeAndIsActiveTrue(Patient.DiabetesType diabetesType);
    
    /**
     * Search patients by name (case insensitive)
     */
    @Query("SELECT p FROM Patient p WHERE " +
           "LOWER(CONCAT(p.firstName, ' ', p.lastName)) LIKE LOWER(CONCAT('%', :name, '%')) " +
           "AND p.isActive = true")
    Page<Patient> findByNameContainingIgnoreCase(@Param("name") String name, Pageable pageable);
    
    /**
     * Find patients diagnosed within date range
     */
    List<Patient> findByDiagnosisDateBetweenAndIsActiveTrue(LocalDate startDate, LocalDate endDate);
    
    /**
     * Find patients by age range
     */
    @Query("SELECT p FROM Patient p WHERE " +
           "YEAR(CURRENT_DATE) - YEAR(p.dateOfBirth) BETWEEN :minAge AND :maxAge " +
           "AND p.isActive = true")
    List<Patient> findByAgeRange(@Param("minAge") int minAge, @Param("maxAge") int maxAge);
    
    /**
     * Check if email exists
     */
    boolean existsByEmail(String email);
    
    /**
     * Count active patients
     */
    long countByIsActiveTrue();
    
    /**
     * Count patients by diabetes type
     */
    long countByDiabetesTypeAndIsActiveTrue(Patient.DiabetesType diabetesType);
}