package com.smartshoe.api.service;

import com.smartshoe.api.entity.Alert;
import com.smartshoe.api.repository.AlertRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Service for managing patient alerts
 */
@Service
@Transactional
public class AlertService {
    
    private static final Logger log = LoggerFactory.getLogger(AlertService.class);
    
    private final AlertRepository alertRepository;
    
    public AlertService(AlertRepository alertRepository) {
        this.alertRepository = alertRepository;
    }
    
    public Alert createAlert(Alert alert) {
        log.info("Creating new alert for patient: {}", alert.getPatientId());
        return alertRepository.save(alert);
    }
    
    public List<Alert> getAlertsByPatientId(Long patientId) {
        return alertRepository.findByPatientIdOrderByCreatedAtDesc(patientId);
    }
    
    public List<Alert> getUnacknowledgedAlerts() {
        return alertRepository.findByIsAcknowledgedFalseOrderByCreatedAtDesc();
    }
    
    public Alert acknowledgeAlert(Long alertId, String acknowledgedBy) {
        Alert alert = alertRepository.findById(alertId)
            .orElseThrow(() -> new RuntimeException("Alert not found"));
        
        alert.setIsAcknowledged(true);
        alert.setAcknowledgedBy(acknowledgedBy);
        alert.setAcknowledgedAt(LocalDateTime.now());
        
        return alertRepository.save(alert);
    }
    
    public List<Alert> getCriticalAlerts() {
        return alertRepository.findBySeverityOrderByCreatedAtDesc(Alert.AlertSeverity.CRITICAL);
    }
}