package com.smartshoe.api.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.smartshoe.api.dto.ml.MLPredictionRequest;
import com.smartshoe.api.dto.ml.MLPredictionResponse;
import com.smartshoe.api.dto.ml.BatchPredictionRequest;
import com.smartshoe.api.dto.ml.BatchPredictionResponse;
import com.smartshoe.api.dto.ml.ModelMetrics;
import com.smartshoe.api.entity.Patient;
import com.smartshoe.api.entity.MedicalReading;
import com.smartshoe.api.entity.ml.MLPrediction;
import com.smartshoe.api.repository.ml.MLPredictionRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.HttpServerErrorException;

import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

@Service
public class MLPredictionService {

    private final RestTemplate restTemplate;
    private final MLPredictionRepository mlPredictionRepository;
    private final ObjectMapper objectMapper;
    
    @Value("${ml.api.base-url:http://localhost:8000}")
    private String mlApiBaseUrl;
    
    @Value("${ml.api.token:ml_api_dev_token}")
    private String mlApiToken;
    
    @Value("${ml.api.timeout:30000}")
    private int mlApiTimeout;
    
    public MLPredictionService(MLPredictionRepository mlPredictionRepository, ObjectMapper objectMapper) {
        this.restTemplate = new RestTemplate();
        this.mlPredictionRepository = mlPredictionRepository;
        this.objectMapper = objectMapper;
    }
    
    /**
     * Predict neuropathy progression for a patient
     */
    public MLPredictionResponse predictNeuropathyProgression(Patient patient, MedicalReading reading) {
        try {
            // Prepare prediction request
            MLPredictionRequest request = buildPredictionRequest(patient, reading, "neuropathy_progression");
            
            // Make API call
            String url = mlApiBaseUrl + "/predict/neuropathy-progression";
            HttpHeaders headers = createHeaders();
            HttpEntity<MLPredictionRequest> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<MLPredictionResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, MLPredictionResponse.class
            );
            
            MLPredictionResponse result = response.getBody();
            
            // Save prediction to database
            if (result != null) {
                savePrediction(patient, result, "neuropathy_progression");
            }
            
            return result;
            
        } catch (HttpClientErrorException | HttpServerErrorException e) {
            System.err.println("ML API Error: " + e.getStatusCode() + " - " + e.getResponseBodyAsString());
            throw new RuntimeException("ML prediction failed: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("ML Service Error: " + e.getMessage());
            throw new RuntimeException("ML service error: " + e.getMessage());
        }
    }
    
    /**
     * Predict glucose complications
     */
    public MLPredictionResponse predictGlucoseComplications(Patient patient, MedicalReading reading) {
        try {
            MLPredictionRequest request = buildPredictionRequest(patient, reading, "glucose_complications");
            
            String url = mlApiBaseUrl + "/predict/glucose-complications";
            HttpHeaders headers = createHeaders();
            HttpEntity<MLPredictionRequest> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<MLPredictionResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, MLPredictionResponse.class
            );
            
            MLPredictionResponse result = response.getBody();
            
            if (result != null) {
                savePrediction(patient, result, "glucose_complications");
            }
            
            return result;
            
        } catch (Exception e) {
            System.err.println("Glucose complications prediction error: " + e.getMessage());
            throw new RuntimeException("Glucose complications prediction failed: " + e.getMessage());
        }
    }
    
    /**
     * Detect anomalies in sensor data
     */
    public MLPredictionResponse detectAnomalies(Patient patient, MedicalReading reading) {
        try {
            MLPredictionRequest request = buildPredictionRequest(patient, reading, "anomaly_detection");
            
            String url = mlApiBaseUrl + "/predict/anomaly-detection";
            HttpHeaders headers = createHeaders();
            HttpEntity<MLPredictionRequest> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<MLPredictionResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, MLPredictionResponse.class
            );
            
            MLPredictionResponse result = response.getBody();
            
            if (result != null) {
                savePrediction(patient, result, "anomaly_detection");
            }
            
            return result;
            
        } catch (Exception e) {
            System.err.println("Anomaly detection error: " + e.getMessage());
            throw new RuntimeException("Anomaly detection failed: " + e.getMessage());
        }
    }
    
    /**
     * Calculate risk stratification
     */
    public MLPredictionResponse calculateRiskStratification(Patient patient, MedicalReading reading) {
        try {
            MLPredictionRequest request = buildPredictionRequest(patient, reading, "risk_stratification");
            
            String url = mlApiBaseUrl + "/predict/risk-stratification";
            HttpHeaders headers = createHeaders();
            HttpEntity<MLPredictionRequest> entity = new HttpEntity<>(request, headers);
            
            ResponseEntity<MLPredictionResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, MLPredictionResponse.class
            );
            
            MLPredictionResponse result = response.getBody();
            
            if (result != null) {
                savePrediction(patient, result, "risk_stratification");
            }
            
            return result;
            
        } catch (Exception e) {
            System.err.println("Risk stratification error: " + e.getMessage());
            throw new RuntimeException("Risk stratification failed: " + e.getMessage());
        }
    }
    
    /**
     * Batch prediction for multiple patients
     */
    public BatchPredictionResponse batchPredict(List<Patient> patients, List<MedicalReading> readings, String modelType) {
        try {
            List<MLPredictionRequest> requests = new ArrayList<>();
            
            for (int i = 0; i < patients.size() && i < readings.size(); i++) {
                MLPredictionRequest request = buildPredictionRequest(patients.get(i), readings.get(i), modelType);
                requests.add(request);
            }
            
            BatchPredictionRequest batchRequest = new BatchPredictionRequest();
            batchRequest.setRequests(requests);
            batchRequest.setBatchId("batch_" + System.currentTimeMillis());
            batchRequest.setTimestamp(LocalDateTime.now());
            
            String url = mlApiBaseUrl + "/predict/batch";
            HttpHeaders headers = createHeaders();
            HttpEntity<BatchPredictionRequest> entity = new HttpEntity<>(batchRequest, headers);
            
            ResponseEntity<BatchPredictionResponse> response = restTemplate.exchange(
                url, HttpMethod.POST, entity, BatchPredictionResponse.class
            );
            
            return response.getBody();
            
        } catch (Exception e) {
            System.err.println("Batch prediction error: " + e.getMessage());
            throw new RuntimeException("Batch prediction failed: " + e.getMessage());
        }
    }
    
    /**
     * Get model metrics
     */
    public ModelMetrics getModelMetrics(String modelType) {
        try {
            String url = mlApiBaseUrl + "/metrics/" + modelType;
            HttpHeaders headers = createHeaders();
            HttpEntity<String> entity = new HttpEntity<>(headers);
            
            ResponseEntity<ModelMetrics> response = restTemplate.exchange(
                url, HttpMethod.GET, entity, ModelMetrics.class
            );
            
            return response.getBody();
            
        } catch (Exception e) {
            System.err.println("Model metrics error: " + e.getMessage());
            throw new RuntimeException("Failed to get model metrics: " + e.getMessage());
        }
    }
    
    /**
     * Get all available models
     */
    public Map<String, Object> getAvailableModels() {
        try {
            String url = mlApiBaseUrl + "/models";
            HttpHeaders headers = createHeaders();
            HttpEntity<String> entity = new HttpEntity<>(headers);
            
            ResponseEntity<Map> response = restTemplate.exchange(
                url, HttpMethod.GET, entity, Map.class
            );
            
            return response.getBody();
            
        } catch (Exception e) {
            System.err.println("Available models error: " + e.getMessage());
            throw new RuntimeException("Failed to get available models: " + e.getMessage());
        }
    }
    
    /**
     * Get predictions for a patient
     */
    public List<MLPrediction> getPredictionsForPatient(Long patientId) {
        return mlPredictionRepository.findByPatientIdOrderByTimestampDesc(patientId);
    }
    
    /**
     * Get predictions by model type
     */
    public List<MLPrediction> getPredictionsByModelType(String modelType) {
        return mlPredictionRepository.findByModelTypeOrderByTimestampDesc(modelType);
    }
    
    /**
     * Get recent predictions
     */
    public List<MLPrediction> getRecentPredictions(int limit) {
        return mlPredictionRepository.findTopByOrderByTimestampDesc(limit);
    }
    
    /**
     * Build prediction request from patient and reading data
     */
    private MLPredictionRequest buildPredictionRequest(Patient patient, MedicalReading reading, String modelType) {
        MLPredictionRequest request = new MLPredictionRequest();
        request.setPatientId(patient.getId().toString());
        request.setModelType(modelType);
        request.setTimestamp(LocalDateTime.now());
        
        // Build features based on patient and reading data
        Map<String, Object> features = new HashMap<>();
        
        // Patient demographics
        features.put("age", patient.getAge());
        features.put("gender_encoded", patient.getGender() != null ? (patient.getGender().equals("MALE") ? 1 : 0) : 0);
        features.put("diabetes_type_encoded", patient.getDiabetesType() != null ? 
            (patient.getDiabetesType().toString().equals("TYPE_1") ? 1 : 2) : 0);
        
        // Calculate years with diabetes
        if (patient.getDiagnosisDate() != null) {
            features.put("years_diabetes", java.time.Period.between(patient.getDiagnosisDate(), java.time.LocalDate.now()).getYears());
        } else {
            features.put("years_diabetes", 0);
        }
        
        // Reading data
        if (reading != null) {
            features.put("reading_value", reading.getValue());
            features.put("signal_strength", reading.getSignalStrength());
            features.put("quality_score", reading.getQualityScore());
            features.put("has_motion_artifacts", reading.getHasMotionArtifacts() ? 1 : 0);
            features.put("severity_level", reading.getSeverityLevel() != null ? 
                reading.getSeverityLevel().ordinal() : 0);
        }
        
        // Default values for missing features
        features.put("bmi", 25.0); // Default BMI
        features.put("hba1c_avg", 7.0); // Default HbA1c
        features.put("pinprick_threshold_avg", 2.0);
        features.put("temp_hot_threshold_avg", 40.0);
        features.put("temp_cold_threshold_avg", 15.0);
        features.put("vibration_threshold_avg", 25.0);
        features.put("response_time_avg", 2.5);
        features.put("test_completion_rate", 0.95);
        features.put("symptom_score_total", 5.0);
        features.put("medication_adherence_avg", 0.85);
        features.put("blood_sugar_variability", 15.0);
        
        request.setFeatures(features);
        return request;
    }
    
    /**
     * Create HTTP headers with authentication
     */
    private HttpHeaders createHeaders() {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.setBearerAuth(mlApiToken);
        return headers;
    }
    
    /**
     * Save prediction to database
     */
    private void savePrediction(Patient patient, MLPredictionResponse response, String modelType) {
        try {
            MLPrediction prediction = new MLPrediction();
            prediction.setPatientId(patient.getId());
            prediction.setModelType(modelType);
            prediction.setPrediction(response.getPrediction());
            prediction.setConfidence(response.getConfidence());
            prediction.setModelVersion(response.getModelVersion());
            prediction.setTimestamp(LocalDateTime.now());
            
            // Store additional data as JSON
            if (response.getAdditionalData() != null) {
                prediction.setAdditionalData(objectMapper.writeValueAsString(response.getAdditionalData()));
            }
            
            if (response.getFeatureImportance() != null) {
                prediction.setFeatureImportance(objectMapper.writeValueAsString(response.getFeatureImportance()));
            }
            
            mlPredictionRepository.save(prediction);
            
        } catch (Exception e) {
            System.err.println("Error saving prediction: " + e.getMessage());
        }
    }
}