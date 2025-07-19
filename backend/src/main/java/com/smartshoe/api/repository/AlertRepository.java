package com.smartshoe.api.repository;

import com.smartshoe.api.entity.Alert;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * Repository for Alert entities
 */
@Repository
public interface AlertRepository extends JpaRepository<Alert, Long> {
    
    List<Alert> findByPatientIdOrderByCreatedAtDesc(Long patientId);
    
    List<Alert> findByIsAcknowledgedFalseOrderByCreatedAtDesc();
    
    List<Alert> findBySeverityOrderByCreatedAtDesc(Alert.AlertSeverity severity);
    
    List<Alert> findByPatientIdAndIsAcknowledgedFalse(Long patientId);
}