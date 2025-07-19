package com.smartshoe.api.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestTemplate;
import org.springframework.boot.web.client.RestTemplateBuilder;

import java.time.Duration;
import java.util.Map;

@Configuration
@ConfigurationProperties(prefix = "ml")
public class MLConfig {
    
    private Api api = new Api();
    
    public Api getApi() {
        return api;
    }
    
    public void setApi(Api api) {
        this.api = api;
    }
    
    @Bean
    public RestTemplate mlRestTemplate(RestTemplateBuilder builder) {
        return builder
                .setConnectTimeout(Duration.ofMillis(api.getTimeout()))
                .setReadTimeout(Duration.ofMillis(api.getTimeout()))
                .build();
    }
    
    public static class Api {
        private String baseUrl = "http://localhost:8000";
        private String token = "ml_api_dev_token";
        private int timeout = 30000;
        private boolean enabled = true;
        private Map<String, ModelConfig> models;
        
        public String getBaseUrl() {
            return baseUrl;
        }
        
        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }
        
        public String getToken() {
            return token;
        }
        
        public void setToken(String token) {
            this.token = token;
        }
        
        public int getTimeout() {
            return timeout;
        }
        
        public void setTimeout(int timeout) {
            this.timeout = timeout;
        }
        
        public boolean isEnabled() {
            return enabled;
        }
        
        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }
        
        public Map<String, ModelConfig> getModels() {
            return models;
        }
        
        public void setModels(Map<String, ModelConfig> models) {
            this.models = models;
        }
    }
    
    public static class ModelConfig {
        private boolean enabled = true;
        private double accuracyThreshold = 0.8;
        
        public boolean isEnabled() {
            return enabled;
        }
        
        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }
        
        public double getAccuracyThreshold() {
            return accuracyThreshold;
        }
        
        public void setAccuracyThreshold(double accuracyThreshold) {
            this.accuracyThreshold = accuracyThreshold;
        }
    }
}