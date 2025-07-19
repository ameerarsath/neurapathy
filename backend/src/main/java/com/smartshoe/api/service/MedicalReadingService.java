package com.smartshoe.api.service;

import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.entity.Device;
import com.smartshoe.api.repository.MedicalReadingRepository;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Service layer for MedicalReading entity operations
 */
@Service
@Transactional(readOnly = true)
public class MedicalReadingService {
    
    private final MedicalReadingRepository medicalReadingRepository;
    private final PatientService patientService;
    private final DeviceService deviceService;
    private final MLAnalysisService mlAnalysisService;
    
    public MedicalReadingService(MedicalReadingRepository medicalReadingRepository, 
                               PatientService patientService, 
                               DeviceService deviceService,
                               MLAnalysisService mlAnalysisService) {
        this.medicalReadingRepository = medicalReadingRepository;
        this.patientService = patientService;
        this.deviceService = deviceService;
        this.mlAnalysisService = mlAnalysisService;
    }
    
    /**
     * Record a new medical reading
     */
    @Transactional
    public MedicalReading recordReading(MedicalReading reading) {
        System.out.println("Recording new medical reading for patient ID: " + reading.getPatient().getId());
        
        // Auto-assess clinical significance
        assessClinicalSignificance(reading);
        
        // Calculate quality score
        calculateQualityScore(reading);
        
        MedicalReading savedReading = medicalReadingRepository.save(reading);
        
        // Trigger ML analysis asynchronously
        try {
            mlAnalysisService.analyzeNewReading(reading.getPatient(), savedReading);
        } catch (Exception e) {
            System.err.println("Error triggering ML analysis: " + e.getMessage());
        }
        
        System.out.println("Recorded medical reading with ID: " + savedReading.getId());
        return savedReading;
    }
    
    /**
     * Record sensor data reading
     */
    @Transactional
    public MedicalReading recordSensorData(Long patientId, Long deviceId, MedicalReading.ReadingType readingType, 
                                         double value, String unit) {
        System.out.println("Recording sensor data for patient " + patientId + " from device " + deviceId);
        
        Patient patient = patientService.getPatientById(patientId);
        Device device = deviceService.getDeviceById(deviceId);
        
        MedicalReading reading = MedicalReading.builder()
                .patient(patient)
                .device(device)
                .readingType(readingType)
                .value(value)
                .unit(unit)
                .recordedAt(LocalDateTime.now())
                .build();
        
        return recordReading(reading);
    }
    
    /**
     * Get all readings with pagination
     */
    public Page<MedicalReading> getAllReadings(int page, int size) {
        Pageable pageable = PageRequest.of(page, size);
        return medicalReadingRepository.findAll(pageable);
    }
    
    /**
     * Get reading by ID
     */
    public MedicalReading getReadingById(Long id) {
        return medicalReadingRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Medical reading not found with ID: " + id));
    }
    
    /**
     * Get readings by patient with pagination
     */
    public Page<MedicalReading> getReadingsByPatient(Long patientId, int page, int size) {
        Pageable pageable = PageRequest.of(page, size);
        return medicalReadingRepository.findByPatientIdOrderByRecordedAtDesc(patientId, pageable);
    }
    
    /**
     * Get readings by patient and type
     */
    public List<MedicalReading> getReadingsByPatientAndType(Long patientId, MedicalReading.ReadingType readingType) {
        Patient patient = patientService.getPatientById(patientId);
        return medicalReadingRepository.findByPatientAndReadingTypeOrderByRecordedAtDesc(patient, readingType);
    }
    
    /**
     * Get readings by device
     */
    public List<MedicalReading> getReadingsByDevice(Long deviceId) {
        Device device = deviceService.getDeviceById(deviceId);
        return medicalReadingRepository.findByDeviceOrderByRecordedAtDesc(device);
    }
    
    /**
     * Get readings within date range
     */
    public List<MedicalReading> getReadingsInDateRange(LocalDateTime startDate, LocalDateTime endDate) {
        return medicalReadingRepository.findByRecordedAtBetweenOrderByRecordedAtDesc(startDate, endDate);
    }
    
    /**
     * Get readings by patient within date range
     */
    public List<MedicalReading> getPatientReadingsInDateRange(Long patientId, LocalDateTime startDate, LocalDateTime endDate) {
        Patient patient = patientService.getPatientById(patientId);
        return medicalReadingRepository.findByPatientAndRecordedAtBetweenOrderByRecordedAtDesc(patient, startDate, endDate);
    }
    
    /**
     * Get abnormal readings
     */
    public List<MedicalReading> getAbnormalReadings() {
        return medicalReadingRepository.findAbnormalReadings();
    }
    
    /**
     * Get critical readings requiring attention
     */
    public List<MedicalReading> getCriticalReadings() {
        return medicalReadingRepository.findCriticalReadings();
    }
    
    /**
     * Get baseline readings for a patient
     */
    public List<MedicalReading> getBaselineReadings(Long patientId) {
        Patient patient = patientService.getPatientById(patientId);
        return medicalReadingRepository.findByPatientAndIsBaselineTrueOrderByRecordedAtDesc(patient);
    }
    
    /**
     * Get high quality readings
     */
    public List<MedicalReading> getHighQualityReadings(double qualityThreshold) {
        return medicalReadingRepository.findHighQualityReadings(qualityThreshold);
    }
    
    /**
     * Get latest reading by patient and type
     */
    public MedicalReading getLatestReadingByPatientAndType(Long patientId, MedicalReading.ReadingType readingType) {
        Patient patient = patientService.getPatientById(patientId);
        Pageable pageable = PageRequest.of(0, 1);
        List<MedicalReading> readings = medicalReadingRepository.findLatestByPatientAndType(patient, readingType, pageable);
        
        return readings.isEmpty() ? null : readings.get(0);
    }
    
    /**
     * Mark reading as baseline
     */
    @Transactional
    public MedicalReading markAsBaseline(Long readingId) {
        System.out.println("Marking reading " + readingId + " as baseline");
        
        MedicalReading reading = getReadingById(readingId);
        reading.setIsBaseline(true);
        
        MedicalReading updatedReading = medicalReadingRepository.save(reading);
        System.out.println("Marked reading " + readingId + " as baseline");
        
        return updatedReading;
    }
    
    /**
     * Update reading severity
     */
    @Transactional
    public MedicalReading updateSeverity(Long readingId, MedicalReading.SeverityLevel severity) {
        System.out.println("Updating reading " + readingId + " severity to " + severity);
        
        MedicalReading reading = getReadingById(readingId);
        reading.setSeverityLevel(severity);
        
        return medicalReadingRepository.save(reading);
    }
    
    /**
     * Add provider notes to reading
     */
    @Transactional
    public MedicalReading addProviderNotes(Long readingId, String notes) {
        System.out.println("Adding provider notes to reading " + readingId);
        
        MedicalReading reading = getReadingById(readingId);
        reading.setProviderNotes(notes);
        
        return medicalReadingRepository.save(reading);
    }
    
    /**
     * Get reading statistics
     */
    public ReadingStatistics getReadingStatistics() {
        long totalReadings = medicalReadingRepository.count();
        long normalReadings = medicalReadingRepository.countByReadingType(MedicalReading.ReadingType.VIBRATION);
        long abnormalReadings = medicalReadingRepository.findAbnormalReadings().size();
        long criticalReadings = medicalReadingRepository.findCriticalReadings().size();
        
        return ReadingStatistics.builder()
                .totalReadings(totalReadings)
                .normalReadings(normalReadings)
                .abnormalReadings(abnormalReadings)
                .criticalReadings(criticalReadings)
                .build();
    }
    
    /**
     * Get patient reading statistics
     */
    public PatientReadingStatistics getPatientReadingStatistics(Long patientId) {
        Patient patient = patientService.getPatientById(patientId);
        
        long totalReadings = medicalReadingRepository.countByPatient(patient);
        long abnormalReadings = medicalReadingRepository.countAbnormalReadingsByPatient(patient);
        
        return PatientReadingStatistics.builder()
                .patientId(patientId)
                .totalReadings(totalReadings)
                .normalReadings(totalReadings - abnormalReadings)
                .abnormalReadings(abnormalReadings)
                .build();
    }
    
    /**
     * Auto-assess clinical significance based on values
     */
    private void assessClinicalSignificance(MedicalReading reading) {
        if (reading.getSeverityLevel() != null) {
            return; // Already assessed
        }
        
        switch (reading.getReadingType()) {
            // Smart shoes do not provide pressure sensing - removed PRESSURE case
            case VIBRATION:
                if (reading.getValue() < 10.0) {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.SEVERE);
                } else if (reading.getValue() < 20.0) {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.MODERATE);
                } else if (reading.getValue() < 30.0) {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.MILD);
                } else {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.NORMAL);
                }
                break;
            case TEMPERATURE:
                if (reading.getValue() < 25.0 || reading.getValue() > 40.0) {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.SEVERE);
                } else if (reading.getValue() < 28.0 || reading.getValue() > 37.0) {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.MODERATE);
                } else {
                    reading.setSeverityLevel(MedicalReading.SeverityLevel.NORMAL);
                }
                break;
            default:
                reading.setSeverityLevel(MedicalReading.SeverityLevel.NORMAL);
        }
    }
    
    /**
     * Calculate quality score based on various factors
     */
    private void calculateQualityScore(MedicalReading reading) {
        if (reading.getQualityScore() != null) {
            return; // Already calculated
        }
        
        double score = 100.0; // Start with perfect score
        
        // Adjust score based on signal strength
        if (reading.getSignalStrength() != null && reading.getSignalStrength() < 50) {
            score -= 20.0;
        }
        
        // Adjust score based on motion artifacts
        if (reading.getHasMotionArtifacts() != null && reading.getHasMotionArtifacts()) {
            score -= 15.0;
        }
        
        // Ensure score is within bounds
        score = Math.max(0.0, Math.min(100.0, score));
        reading.setQualityScore(score);
    }
    
    /**
     * Statistics DTOs
     */
    public static class ReadingStatistics {
        private long totalReadings;
        private long normalReadings;
        private long abnormalReadings;
        private long criticalReadings;
        
        public ReadingStatistics() {}
        
        public static ReadingStatistics builder() {
            return new ReadingStatistics();
        }
        
        public ReadingStatistics totalReadings(long totalReadings) {
            this.totalReadings = totalReadings;
            return this;
        }
        
        public ReadingStatistics normalReadings(long normalReadings) {
            this.normalReadings = normalReadings;
            return this;
        }
        
        public ReadingStatistics abnormalReadings(long abnormalReadings) {
            this.abnormalReadings = abnormalReadings;
            return this;
        }
        
        public ReadingStatistics criticalReadings(long criticalReadings) {
            this.criticalReadings = criticalReadings;
            return this;
        }
        
        public ReadingStatistics build() {
            return this;
        }
        
        public long getTotalReadings() { return totalReadings; }
        public void setTotalReadings(long totalReadings) { this.totalReadings = totalReadings; }
        public long getNormalReadings() { return normalReadings; }
        public void setNormalReadings(long normalReadings) { this.normalReadings = normalReadings; }
        public long getAbnormalReadings() { return abnormalReadings; }
        public void setAbnormalReadings(long abnormalReadings) { this.abnormalReadings = abnormalReadings; }
        public long getCriticalReadings() { return criticalReadings; }
        public void setCriticalReadings(long criticalReadings) { this.criticalReadings = criticalReadings; }
    }
    
    public static class PatientReadingStatistics {
        private Long patientId;
        private long totalReadings;
        private long normalReadings;
        private long abnormalReadings;
        
        public PatientReadingStatistics() {}
        
        public static PatientReadingStatistics builder() {
            return new PatientReadingStatistics();
        }
        
        public PatientReadingStatistics patientId(Long patientId) {
            this.patientId = patientId;
            return this;
        }
        
        public PatientReadingStatistics totalReadings(long totalReadings) {
            this.totalReadings = totalReadings;
            return this;
        }
        
        public PatientReadingStatistics normalReadings(long normalReadings) {
            this.normalReadings = normalReadings;
            return this;
        }
        
        public PatientReadingStatistics abnormalReadings(long abnormalReadings) {
            this.abnormalReadings = abnormalReadings;
            return this;
        }
        
        public PatientReadingStatistics build() {
            return this;
        }
        
        public Long getPatientId() { return patientId; }
        public void setPatientId(Long patientId) { this.patientId = patientId; }
        public long getTotalReadings() { return totalReadings; }
        public void setTotalReadings(long totalReadings) { this.totalReadings = totalReadings; }
        public long getNormalReadings() { return normalReadings; }
        public void setNormalReadings(long normalReadings) { this.normalReadings = normalReadings; }
        public long getAbnormalReadings() { return abnormalReadings; }
        public void setAbnormalReadings(long abnormalReadings) { this.abnormalReadings = abnormalReadings; }
    }
}