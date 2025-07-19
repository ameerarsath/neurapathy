package com.smartshoe.api.dto.ml;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.time.LocalDateTime;
import java.util.List;

public class BatchPredictionRequest {
    
    @JsonProperty("requests")
    private List<MLPredictionRequest> requests;
    
    @JsonProperty("batch_id")
    private String batchId;
    
    @JsonProperty("timestamp")
    private LocalDateTime timestamp;
    
    // Getters and Setters
    public List<MLPredictionRequest> getRequests() { return requests; }
    public void setRequests(List<MLPredictionRequest> requests) { this.requests = requests; }
    public String getBatchId() { return batchId; }
    public void setBatchId(String batchId) { this.batchId = batchId; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }
}