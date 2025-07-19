package com.smartshoe.api.service;

import com.smartshoe.api.entity.NeuropathyTest;
import com.smartshoe.api.entity.TestStimulus;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.repository.TestStimulusRepository;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.stream.Collectors;

/**
 * ML Model Service for Neuropathy Analysis
 * Provides machine learning-based predictions for diabetic neuropathy severity assessment
 */
@Service
public class MLModelService {
    
    @Autowired
    private NeuropathyTestService neuropathyTestService;
    
    @Autowired
    private TestStimulusRepository testStimulusRepository;
    
    /**
     * Analyzes neuropathy test results and provides ML-based severity assessment
     */
    public NeuropathyAnalysisResult analyzeNeuropathyTest(Long testId) {
        NeuropathyTest test = neuropathyTestService.getTestById(testId);
        List<TestStimulus> stimuli = testStimulusRepository.findByNeuropathyTestIdOrderByStimulusSequenceAsc(testId);
        
        if (stimuli.isEmpty()) {
            throw new IllegalArgumentException("Test has no stimulus data for analysis");
        }
        
        // Extract features for ML analysis
        MLFeatures features = extractFeatures(stimuli, test.getPatient());
        
        // Run ML prediction models
        NeuropathySeverity severity = predictNeuropathySeverity(features);
        double riskScore = calculateRiskScore(features);
        Map<String, Double> detailedAnalysis = performDetailedAnalysis(features);
        
        return new NeuropathyAnalysisResult(
            severity,
            riskScore,
            detailedAnalysis,
            generateRecommendations(severity, riskScore),
            features.getConfidenceScore()
        );
    }
    
    /**
     * Extract ML features from test stimulus data
     */
    private MLFeatures extractFeatures(List<TestStimulus> stimuli, Patient patient) {
        MLFeatures features = new MLFeatures();
        
        // Sensation Detection Metrics
        long totalStimuli = stimuli.stream().filter(s -> !s.getNoStimulusTrial()).count();
        long detectedStimuli = stimuli.stream()
            .filter(s -> !s.getNoStimulusTrial() && s.getPatientFeltSensation() != null && s.getPatientFeltSensation())
            .count();
        
        features.setSensationDetectionRate(totalStimuli > 0 ? (double) detectedStimuli / totalStimuli : 0.0);
        
        // False Positive Rate (control trials where patient felt something)
        long controlTrials = stimuli.stream().filter(TestStimulus::getNoStimulusTrial).count();
        long falsePositives = stimuli.stream()
            .filter(s -> s.getNoStimulusTrial() && s.getPatientFeltSensation() != null && s.getPatientFeltSensation())
            .count();
        
        features.setFalsePositiveRate(controlTrials > 0 ? (double) falsePositives / controlTrials : 0.0);
        
        // Intensity Perception Accuracy
        double intensityAccuracy = stimuli.stream()
            .filter(s -> !s.getNoStimulusTrial() && s.getPatientFeltSensation() != null && s.getPatientFeltSensation())
            .mapToDouble(TestStimulus::getIntensityAccuracy)
            .average()
            .orElse(0.0);
        features.setIntensityAccuracy(intensityAccuracy);
        
        // Type Recognition Accuracy
        long correctTypeRecognitions = stimuli.stream()
            .filter(s -> !s.getNoStimulusTrial() && s.isTypeMatchCorrect())
            .count();
        features.setTypeRecognitionRate(detectedStimuli > 0 ? (double) correctTypeRecognitions / detectedStimuli : 0.0);
        
        // Response Consistency (confidence metrics)
        double avgConfidence = stimuli.stream()
            .filter(s -> s.getResponseConfidence() != null)
            .mapToInt(TestStimulus::getResponseConfidence)
            .average()
            .orElse(3.0);
        features.setAverageConfidence(avgConfidence / 5.0); // Normalize to 0-1
        
        // Vibration-specific analysis
        analyzeByModalityType(stimuli, features);
        
        // Patient demographic factors
        features.setPatientAge(calculatePatientAge(patient));
        features.setDiabetesDuration(calculateDiabetesDuration(patient));
        features.setDiabetesType(patient.getDiabetesType().toString());
        
        // Response time analysis
        analyzeResponseTimes(stimuli, features);
        
        return features;
    }
    
    /**
     * Analyze performance by stimulus modality (vibration, temperature, etc.)
     */
    private void analyzeByModalityType(List<TestStimulus> stimuli, MLFeatures features) {
        Map<TestStimulus.StimulusType, Double> modalityPerformance = new HashMap<>();
        
        for (TestStimulus.StimulusType type : TestStimulus.StimulusType.values()) {
            if (type == TestStimulus.StimulusType.NONE) continue;
            
            List<TestStimulus> typeStimuli = stimuli.stream()
                .filter(s -> s.getStimulusType() == type && !s.getNoStimulusTrial())
                .collect(Collectors.toList());
            
            if (!typeStimuli.isEmpty()) {
                double detectionRate = typeStimuli.stream()
                    .mapToDouble(s -> s.getPatientFeltSensation() != null && s.getPatientFeltSensation() ? 1.0 : 0.0)
                    .average()
                    .orElse(0.0);
                modalityPerformance.put(type, detectionRate);
            }
        }
        
        features.setModalityPerformance(modalityPerformance);
        
        // Calculate modality-specific thresholds (smart shoes provide vibration, temperature, and pinprick only)
        features.setVibrationThreshold(calculateModalityThreshold(stimuli, TestStimulus.StimulusType.VIBRATION));
        features.setTemperatureThreshold(calculateModalityThreshold(stimuli, TestStimulus.StimulusType.TEMPERATURE_HOT));
    }
    
    /**
     * Calculate detection threshold for specific modality
     */
    private double calculateModalityThreshold(List<TestStimulus> stimuli, TestStimulus.StimulusType type) {
        return stimuli.stream()
            .filter(s -> s.getStimulusType() == type && !s.getNoStimulusTrial())
            .filter(s -> s.getPatientFeltSensation() != null && s.getPatientFeltSensation())
            .mapToDouble(s -> s.getStimulusIntensity() != null ? s.getStimulusIntensity() : 1.0)
            .min()
            .orElse(1.0);
    }
    
    /**
     * Analyze response time patterns
     */
    private void analyzeResponseTimes(List<TestStimulus> stimuli, MLFeatures features) {
        List<Long> responseTimes = stimuli.stream()
            .filter(s -> s.getResponseTimeMs() != null)
            .map(TestStimulus::getResponseTimeMs)
            .collect(Collectors.toList());
        
        if (!responseTimes.isEmpty()) {
            double avgResponseTime = responseTimes.stream().mapToLong(Long::longValue).average().orElse(0.0);
            features.setAverageResponseTime(avgResponseTime);
            
            // Calculate response time variability
            double variance = responseTimes.stream()
                .mapToDouble(t -> Math.pow(t - avgResponseTime, 2))
                .average()
                .orElse(0.0);
            features.setResponseTimeVariability(Math.sqrt(variance));
        }
    }
    
    /**
     * ML-based neuropathy severity prediction
     */
    private NeuropathySeverity predictNeuropathySeverity(MLFeatures features) {
        // ML Model Logic - This would typically call an external ML service or trained model
        double severityScore = calculateSeverityScore(features);
        
        if (severityScore >= 0.8) return NeuropathySeverity.SEVERE;
        if (severityScore >= 0.6) return NeuropathySeverity.MODERATE;
        if (severityScore >= 0.3) return NeuropathySeverity.MILD;
        return NeuropathySeverity.NORMAL;
    }
    
    /**
     * Calculate composite severity score using ML features
     */
    private double calculateSeverityScore(MLFeatures features) {
        double score = 0.0;
        
        // Weighted feature importance based on clinical research
        score += (1.0 - features.getSensationDetectionRate()) * 0.35; // Loss of sensation
        score += features.getFalsePositiveRate() * 0.15; // False sensations
        score += (1.0 - features.getIntensityAccuracy()) * 0.20; // Intensity perception
        score += (1.0 - features.getTypeRecognitionRate()) * 0.15; // Type discrimination
        score += (1.0 - features.getAverageConfidence()) * 0.10; // Response confidence
        score += Math.min(1.0, features.getAverageResponseTime() / 5000.0) * 0.05; // Response delay
        
        return Math.min(1.0, score);
    }
    
    /**
     * Calculate overall risk score for diabetic complications
     */
    private double calculateRiskScore(MLFeatures features) {
        double riskScore = calculateSeverityScore(features);
        
        // Adjust for patient demographics
        if (features.getPatientAge() > 65) riskScore += 0.1;
        if (features.getDiabetesDuration() > 10) riskScore += 0.15;
        if ("TYPE_1".equals(features.getDiabetesType())) riskScore += 0.05;
        
        // Modality-specific risk factors
        if (features.getVibrationThreshold() > 0.7) riskScore += 0.1; // Poor vibration sensitivity
        if (features.getTemperatureThreshold() > 0.8) riskScore += 0.1; // Poor temperature sensitivity
        
        return Math.min(1.0, riskScore);
    }
    
    /**
     * Perform detailed analysis by test components
     */
    private Map<String, Double> performDetailedAnalysis(MLFeatures features) {
        Map<String, Double> analysis = new HashMap<>();
        
        analysis.put("sensation_detection_score", features.getSensationDetectionRate());
        analysis.put("false_positive_score", 1.0 - features.getFalsePositiveRate());
        analysis.put("intensity_perception_score", features.getIntensityAccuracy());
        analysis.put("type_recognition_score", features.getTypeRecognitionRate());
        analysis.put("response_consistency_score", features.getAverageConfidence());
        analysis.put("vibration_sensitivity_score", 1.0 - features.getVibrationThreshold());
        analysis.put("temperature_sensitivity_score", 1.0 - features.getTemperatureThreshold());
        
        return analysis;
    }
    
    /**
     * Generate clinical recommendations based on analysis
     */
    private List<String> generateRecommendations(NeuropathySeverity severity, double riskScore) {
        List<String> recommendations = new java.util.ArrayList<>();
        
        switch (severity) {
            case SEVERE:
                recommendations.add("Immediate referral to endocrinologist and neurologist recommended");
                recommendations.add("Implement comprehensive foot care program");
                recommendations.add("Consider advanced pain management consultation");
                recommendations.add("Monthly follow-up testing recommended");
                break;
            case MODERATE:
                recommendations.add("Endocrinologist consultation recommended within 2 weeks");
                recommendations.add("Enhance diabetic foot care routine");
                recommendations.add("Bi-monthly neuropathy assessment recommended");
                recommendations.add("Consider medication review for neuropathic symptoms");
                break;
            case MILD:
                recommendations.add("Continue current diabetes management plan");
                recommendations.add("Quarterly neuropathy screening recommended");
                recommendations.add("Implement preventive foot care measures");
                recommendations.add("Monitor for symptom progression");
                break;
            case NORMAL:
                recommendations.add("Maintain excellent diabetes control");
                recommendations.add("Annual neuropathy screening recommended");
                recommendations.add("Continue preventive care measures");
                break;
        }
        
        if (riskScore > 0.7) {
            recommendations.add("High risk for complications - consider more frequent monitoring");
        }
        
        return recommendations;
    }
    
    // Helper methods
    private int calculatePatientAge(Patient patient) {
        return java.time.Period.between(patient.getDateOfBirth(), java.time.LocalDate.now()).getYears();
    }
    
    private int calculateDiabetesDuration(Patient patient) {
        if (patient.getDiagnosisDate() == null) return 0;
        return java.time.Period.between(patient.getDiagnosisDate(), java.time.LocalDate.now()).getYears();
    }
    
    // Enums and inner classes
    public enum NeuropathySeverity {
        NORMAL, MILD, MODERATE, SEVERE
    }
    
    /**
     * ML Features extracted from neuropathy test data
     */
    public static class MLFeatures {
        private double sensationDetectionRate;
        private double falsePositiveRate;
        private double intensityAccuracy;
        private double typeRecognitionRate;
        private double averageConfidence;
        private double averageResponseTime;
        private double responseTimeVariability;
        private Map<TestStimulus.StimulusType, Double> modalityPerformance;
        private double vibrationThreshold;
        private double temperatureThreshold;
        private int patientAge;
        private int diabetesDuration;
        private String diabetesType;
        
        // Getters and setters
        public double getSensationDetectionRate() { return sensationDetectionRate; }
        public void setSensationDetectionRate(double sensationDetectionRate) { this.sensationDetectionRate = sensationDetectionRate; }
        
        public double getFalsePositiveRate() { return falsePositiveRate; }
        public void setFalsePositiveRate(double falsePositiveRate) { this.falsePositiveRate = falsePositiveRate; }
        
        public double getIntensityAccuracy() { return intensityAccuracy; }
        public void setIntensityAccuracy(double intensityAccuracy) { this.intensityAccuracy = intensityAccuracy; }
        
        public double getTypeRecognitionRate() { return typeRecognitionRate; }
        public void setTypeRecognitionRate(double typeRecognitionRate) { this.typeRecognitionRate = typeRecognitionRate; }
        
        public double getAverageConfidence() { return averageConfidence; }
        public void setAverageConfidence(double averageConfidence) { this.averageConfidence = averageConfidence; }
        
        public double getAverageResponseTime() { return averageResponseTime; }
        public void setAverageResponseTime(double averageResponseTime) { this.averageResponseTime = averageResponseTime; }
        
        public double getResponseTimeVariability() { return responseTimeVariability; }
        public void setResponseTimeVariability(double responseTimeVariability) { this.responseTimeVariability = responseTimeVariability; }
        
        public Map<TestStimulus.StimulusType, Double> getModalityPerformance() { return modalityPerformance; }
        public void setModalityPerformance(Map<TestStimulus.StimulusType, Double> modalityPerformance) { this.modalityPerformance = modalityPerformance; }
        
        public double getVibrationThreshold() { return vibrationThreshold; }
        public void setVibrationThreshold(double vibrationThreshold) { this.vibrationThreshold = vibrationThreshold; }
        
        public double getTemperatureThreshold() { return temperatureThreshold; }
        public void setTemperatureThreshold(double temperatureThreshold) { this.temperatureThreshold = temperatureThreshold; }
        
        
        public int getPatientAge() { return patientAge; }
        public void setPatientAge(int patientAge) { this.patientAge = patientAge; }
        
        public int getDiabetesDuration() { return diabetesDuration; }
        public void setDiabetesDuration(int diabetesDuration) { this.diabetesDuration = diabetesDuration; }
        
        public String getDiabetesType() { return diabetesType; }
        public void setDiabetesType(String diabetesType) { this.diabetesType = diabetesType; }
        
        public double getConfidenceScore() {
            // Calculate overall confidence in the ML prediction
            return Math.min(1.0, averageConfidence * 0.6 + (1.0 - responseTimeVariability / 1000.0) * 0.4);
        }
    }
    
    /**
     * ML Analysis Result containing predictions and recommendations
     */
    public static class NeuropathyAnalysisResult {
        private final NeuropathySeverity severity;
        private final double riskScore;
        private final Map<String, Double> detailedAnalysis;
        private final List<String> recommendations;
        private final double confidenceScore;
        
        public NeuropathyAnalysisResult(NeuropathySeverity severity, double riskScore, 
                                      Map<String, Double> detailedAnalysis, 
                                      List<String> recommendations, double confidenceScore) {
            this.severity = severity;
            this.riskScore = riskScore;
            this.detailedAnalysis = detailedAnalysis;
            this.recommendations = recommendations;
            this.confidenceScore = confidenceScore;
        }
        
        // Getters
        public NeuropathySeverity getSeverity() { return severity; }
        public double getRiskScore() { return riskScore; }
        public Map<String, Double> getDetailedAnalysis() { return detailedAnalysis; }
        public List<String> getRecommendations() { return recommendations; }
        public double getConfidenceScore() { return confidenceScore; }
    }
}