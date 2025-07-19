package com.smartshoe.api.repository;

import com.smartshoe.api.entity.NeuropathyTest;
import com.smartshoe.api.entity.Patient;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface NeuropathyTestRepository extends JpaRepository<NeuropathyTest, Long> {
    
    List<NeuropathyTest> findByPatientOrderByStartedAtDesc(Patient patient);
    
    List<NeuropathyTest> findByPatientIdOrderByStartedAtDesc(Long patientId);
    
    List<NeuropathyTest> findByTestStatusOrderByStartedAtDesc(NeuropathyTest.TestStatus status);
    
    @Query("SELECT nt FROM NeuropathyTest nt WHERE nt.patient.id = :patientId AND nt.testStatus = :status")
    List<NeuropathyTest> findByPatientIdAndStatus(@Param("patientId") Long patientId, 
                                                  @Param("status") NeuropathyTest.TestStatus status);
    
    @Query("SELECT nt FROM NeuropathyTest nt WHERE nt.startedAt BETWEEN :startDate AND :endDate ORDER BY nt.startedAt DESC")
    List<NeuropathyTest> findTestsBetweenDates(@Param("startDate") LocalDateTime startDate, 
                                               @Param("endDate") LocalDateTime endDate);
    
    @Query("SELECT nt FROM NeuropathyTest nt WHERE nt.patient.id = :patientId AND nt.baselineTest = true ORDER BY nt.startedAt DESC")
    Optional<NeuropathyTest> findLatestBaselineTest(@Param("patientId") Long patientId);
    
    @Query("SELECT COUNT(nt) FROM NeuropathyTest nt WHERE nt.patient.id = :patientId AND nt.testStatus = 'COMPLETED'")
    Long countCompletedTestsByPatient(@Param("patientId") Long patientId);
}