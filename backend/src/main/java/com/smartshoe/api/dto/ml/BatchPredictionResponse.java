package com.smartshoe.api.dto.ml;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

@Data
public class BatchPredictionResponse {
    
    @JsonProperty("batch_id")
    private String batchId;
    
    @JsonProperty("responses")
    private List<MLPredictionResponse> responses;
    
    @JsonProperty("total_requests")
    private Integer totalRequests;
    
    @JsonProperty("successful_predictions")
    private Integer successfulPredictions;
    
    @JsonProperty("failed_predictions")
    private Integer failedPredictions;
    
    @JsonProperty("timestamp")
    private LocalDateTime timestamp;
    
    @JsonProperty("total_processing_time_ms")
    private Double totalProcessingTimeMs;
}