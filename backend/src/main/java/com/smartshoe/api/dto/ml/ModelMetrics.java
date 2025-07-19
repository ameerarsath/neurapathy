package com.smartshoe.api.dto.ml;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.time.LocalDateTime;

@Data
public class ModelMetrics {
    
    @JsonProperty("model_type")
    private String modelType;
    
    @JsonProperty("accuracy")
    private Double accuracy;
    
    @JsonProperty("precision")
    private Double precision;
    
    @JsonProperty("recall")
    private Double recall;
    
    @JsonProperty("f1_score")
    private Double f1Score;
    
    @JsonProperty("auc")
    private Double auc;
    
    @JsonProperty("last_updated")
    private LocalDateTime lastUpdated;
    
    @JsonProperty("prediction_count")
    private Integer predictionCount;
    
    @JsonProperty("average_latency_ms")
    private Double averageLatencyMs;
}