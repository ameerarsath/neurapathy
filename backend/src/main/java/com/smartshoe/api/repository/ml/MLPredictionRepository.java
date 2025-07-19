package com.smartshoe.api.repository.ml;

import com.smartshoe.api.entity.ml.MLPrediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface MLPredictionRepository extends JpaRepository<MLPrediction, Long> {
    
    /**
     * Find predictions for a specific patient
     */
    List<MLPrediction> findByPatientIdOrderByTimestampDesc(Long patientId);
    
    /**
     * Find predictions by model type
     */
    List<MLPrediction> findByModelTypeOrderByTimestampDesc(String modelType);
    
    /**
     * Find predictions for a patient by model type
     */
    List<MLPrediction> findByPatientIdAndModelTypeOrderByTimestampDesc(Long patientId, String modelType);
    
    /**
     * Find top N recent predictions
     */
    @Query("SELECT p FROM MLPrediction p ORDER BY p.timestamp DESC LIMIT :limit")
    List<MLPrediction> findTopByOrderByTimestampDesc(@Param("limit") int limit);
    
    /**
     * Find predictions within date range
     */
    List<MLPrediction> findByTimestampBetweenOrderByTimestampDesc(LocalDateTime startDate, LocalDateTime endDate);
    
    /**
     * Find predictions for a patient within date range
     */
    List<MLPrediction> findByPatientIdAndTimestampBetweenOrderByTimestampDesc(
        Long patientId, LocalDateTime startDate, LocalDateTime endDate);
    
    /**
     * Find latest prediction for a patient by model type
     */
    Optional<MLPrediction> findFirstByPatientIdAndModelTypeOrderByTimestampDesc(Long patientId, String modelType);
    
    /**
     * Find high-risk predictions
     */
    @Query("SELECT p FROM MLPrediction p WHERE p.prediction > :threshold ORDER BY p.timestamp DESC")
    List<MLPrediction> findHighRiskPredictions(@Param("threshold") Double threshold);
    
    /**
     * Count predictions for a patient
     */
    long countByPatientId(Long patientId);
    
    /**
     * Count predictions by model type
     */
    long countByModelType(String modelType);
    
    /**
     * Get average confidence for a model type
     */
    @Query("SELECT AVG(p.confidence) FROM MLPrediction p WHERE p.modelType = :modelType")
    Double getAverageConfidenceByModelType(@Param("modelType") String modelType);
    
    /**
     * Get average prediction value for a patient
     */
    @Query("SELECT AVG(p.prediction) FROM MLPrediction p WHERE p.patientId = :patientId AND p.modelType = :modelType")
    Double getAveragePredictionByPatientAndModel(@Param("patientId") Long patientId, @Param("modelType") String modelType);
    
    /**
     * Find predictions with low confidence
     */
    @Query("SELECT p FROM MLPrediction p WHERE p.confidence < :threshold ORDER BY p.timestamp DESC")
    List<MLPrediction> findLowConfidencePredictions(@Param("threshold") Double threshold);
    
    /**
     * Delete old predictions (data retention)
     */
    void deleteByTimestampBefore(LocalDateTime cutoffDate);
}