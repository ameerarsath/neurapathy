package com.smartshoe.api.repository;

import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.entity.Device;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Repository interface for MedicalReading entity operations
 */
@Repository
public interface MedicalReadingRepository extends JpaRepository<MedicalReading, Long> {
    
    /**
     * Find readings by patient
     */
    Page<MedicalReading> findByPatientOrderByRecordedAtDesc(Patient patient, Pageable pageable);
    
    /**
     * Find readings by patient ID
     */
    Page<MedicalReading> findByPatientIdOrderByRecordedAtDesc(Long patientId, Pageable pageable);
    
    /**
     * Find readings by device
     */
    List<MedicalReading> findByDeviceOrderByRecordedAtDesc(Device device);
    
    /**
     * Find readings by reading type
     */
    List<MedicalReading> findByReadingTypeOrderByRecordedAtDesc(MedicalReading.ReadingType readingType);
    
    /**
     * Find readings by patient and reading type
     */
    List<MedicalReading> findByPatientAndReadingTypeOrderByRecordedAtDesc(
            Patient patient, MedicalReading.ReadingType readingType);
    
    /**
     * Find readings within date range
     */
    List<MedicalReading> findByRecordedAtBetweenOrderByRecordedAtDesc(
            LocalDateTime startDate, LocalDateTime endDate);
    
    /**
     * Find readings by patient within date range
     */
    List<MedicalReading> findByPatientAndRecordedAtBetweenOrderByRecordedAtDesc(
            Patient patient, LocalDateTime startDate, LocalDateTime endDate);
    
    /**
     * Find abnormal readings (not normal severity)
     */
    @Query("SELECT mr FROM MedicalReading mr WHERE " +
           "mr.severityLevel != com.smartshoe.api.entity.MedicalReading$SeverityLevel.NORMAL " +
           "ORDER BY mr.recordedAt DESC")
    List<MedicalReading> findAbnormalReadings();
    
    /**
     * Find critical readings requiring attention
     */
    @Query("SELECT mr FROM MedicalReading mr WHERE " +
           "mr.severityLevel IN ('SEVERE', 'CRITICAL') " +
           "ORDER BY mr.recordedAt DESC")
    List<MedicalReading> findCriticalReadings();
    
    /**
     * Find baseline readings for a patient
     */
    List<MedicalReading> findByPatientAndIsBaselineTrueOrderByRecordedAtDesc(Patient patient);
    
    /**
     * Find high quality readings
     */
    @Query("SELECT mr FROM MedicalReading mr WHERE " +
           "mr.qualityScore >= :threshold " +
           "ORDER BY mr.recordedAt DESC")
    List<MedicalReading> findHighQualityReadings(@Param("threshold") double threshold);
    
    /**
     * Find latest reading by patient and type
     */
    @Query("SELECT mr FROM MedicalReading mr WHERE " +
           "mr.patient = :patient AND mr.readingType = :readingType " +
           "ORDER BY mr.recordedAt DESC")
    List<MedicalReading> findLatestByPatientAndType(
            @Param("patient") Patient patient, 
            @Param("readingType") MedicalReading.ReadingType readingType,
            Pageable pageable);
    
    /**
     * Count readings by patient
     */
    long countByPatient(Patient patient);
    
    /**
     * Count readings by reading type
     */
    long countByReadingType(MedicalReading.ReadingType readingType);
    
    /**
     * Count abnormal readings for a patient
     */
    @Query("SELECT COUNT(mr) FROM MedicalReading mr WHERE " +
           "mr.patient = :patient AND " +
           "mr.severityLevel != com.smartshoe.api.entity.MedicalReading$SeverityLevel.NORMAL")
    long countAbnormalReadingsByPatient(@Param("patient") Patient patient);
}