package com.smartshoe.api.controller;

import com.smartshoe.api.entity.*;
import com.smartshoe.api.repository.*;
import com.smartshoe.api.service.MLModelService;
import com.smartshoe.api.util.ValidationUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Neuropathy Test Controller - Handles neuropathy testing sessions and patient responses
 */
@RestController
@RequestMapping("/api/neuropathy")
@CrossOrigin(origins = "*")
public class NeuropathyTestController {

    @Autowired
    private NeuropathyTestRepository neuropathyTestRepository;
    
    @Autowired
    private TestStimulusRepository testStimulusRepository;
    
    @Autowired
    private PatientRepository patientRepository;
    
    @Autowired
    private DeviceRepository deviceRepository;
    
    @Autowired
    private MLModelService mlModelService;

    /**
     * Start a new neuropathy test session
     */
    @PostMapping("/test/start")
    public ResponseEntity<Map<String, Object>> startTest(@RequestBody Map<String, Object> request) {
        try {
            // Validate required parameters
            if (!ValidationUtils.hasRequiredParameters(request, "patientId", "deviceId", "footSide")) {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Missing required parameters: patientId, deviceId, footSide"
                ));
            }
            
            Long patientId;
            Long deviceId;
            String footSideStr;
            Boolean isBaseline;
            
            try {
                patientId = ValidationUtils.safeLongValue(request.get("patientId"));
                deviceId = ValidationUtils.safeLongValue(request.get("deviceId"));
                footSideStr = ValidationUtils.safeStringValue(request.get("footSide"));
                isBaseline = ValidationUtils.safeBooleanValue(request.getOrDefault("isBaseline", false));
            } catch (IllegalArgumentException e) {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Invalid parameter format: " + e.getMessage()
                ));
            }
            
            Optional<Patient> patient = patientRepository.findById(patientId);
            Optional<Device> device = deviceRepository.findById(deviceId);
            
            if (!patient.isPresent() || !device.isPresent()) {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Patient or Device not found"
                ));
            }
            
            NeuropathyTest.FootSide footSide;
            try {
                footSide = NeuropathyTest.FootSide.valueOf(footSideStr.toUpperCase());
            } catch (IllegalArgumentException e) {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Invalid footSide value: " + footSideStr
                ));
            }
            
            NeuropathyTest test = new NeuropathyTest(patient.get(), device.get(), footSide);
            test.setBaselineTest(isBaseline);
            test.setTotalStimuli(20); // Default 20 stimuli per test
            test.setTestDurationMinutes(15); // Estimated 15 minutes
            test.setTestStatus(NeuropathyTest.TestStatus.PENDING);
            
            NeuropathyTest savedTest = neuropathyTestRepository.save(test);
            
            // Generate test stimuli sequence
            generateTestStimuli(savedTest);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("testId", savedTest.getId());
            response.put("totalStimuli", savedTest.getTotalStimuli());
            response.put("estimatedDuration", savedTest.getTestDurationMinutes());
            response.put("instructions", getTestInstructions());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error starting test: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Mark test instructions as shown and start the actual test
     */
    @PostMapping("/test/{testId}/begin")
    public ResponseEntity<Map<String, Object>> beginTest(@PathVariable Long testId) {
        try {
            Optional<NeuropathyTest> testOpt = neuropathyTestRepository.findById(testId);
            if (!testOpt.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            NeuropathyTest test = testOpt.get();
            test.setInstructionsShown(true);
            test.setTestStatus(NeuropathyTest.TestStatus.IN_PROGRESS);
            neuropathyTestRepository.save(test);
            
            // Get first stimulus
            List<TestStimulus> stimuli = testStimulusRepository.findByNeuropathyTestIdOrderByStimulusSequenceAsc(testId);
            TestStimulus currentStimulus = stimuli.isEmpty() ? null : stimuli.get(0);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("testStarted", true);
            if (currentStimulus != null) {
                response.put("currentStimulus", createStimulusResponse(currentStimulus, false)); // Patient view
            }
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error beginning test: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Submit patient response to a stimulus
     */
    @PostMapping("/test/{testId}/respond")
    public ResponseEntity<Map<String, Object>> submitResponse(@PathVariable Long testId, 
                                                              @RequestBody Map<String, Object> response) {
        try {
            Long stimulusId = Long.valueOf(response.get("stimulusId").toString());
            Boolean feltSensation = Boolean.valueOf(response.getOrDefault("feltSensation", false).toString());
            
            Optional<TestStimulus> stimulusOpt = testStimulusRepository.findById(stimulusId);
            if (!stimulusOpt.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            TestStimulus stimulus = stimulusOpt.get();
            stimulus.setPatientFeltSensation(feltSensation);
            stimulus.setResponseTime(LocalDateTime.now());
            
            if (feltSensation) {
                if (response.containsKey("perceivedIntensity")) {
                    stimulus.setPerceivedIntensity(Integer.valueOf(response.get("perceivedIntensity").toString()));
                }
                if (response.containsKey("perceivedType")) {
                    stimulus.setPerceivedType(TestStimulus.StimulusType.valueOf(response.get("perceivedType").toString()));
                }
                if (response.containsKey("perceivedLocation")) {
                    stimulus.setPerceivedLocation(response.get("perceivedLocation").toString());
                }
                if (response.containsKey("responseConfidence")) {
                    stimulus.setResponseConfidence(Integer.valueOf(response.get("responseConfidence").toString()));
                }
            }
            
            testStimulusRepository.save(stimulus);
            
            // Update test progress
            NeuropathyTest test = stimulus.getNeuropathyTest();
            Long completedCount = testStimulusRepository.countCompletedResponses(testId);
            test.setCompletedStimuli(completedCount.intValue());
            
            // Check if test is complete
            boolean testComplete = completedCount >= test.getTotalStimuli();
            if (testComplete) {
                test.markCompleted();
            }
            
            neuropathyTestRepository.save(test);
            
            // Get next stimulus if test not complete
            TestStimulus nextStimulus = null;
            if (!testComplete) {
                List<TestStimulus> pendingStimuli = testStimulusRepository.findPendingResponses(testId);
                nextStimulus = pendingStimuli.isEmpty() ? null : pendingStimuli.get(0);
            }
            
            Map<String, Object> responseMap = new HashMap<>();
            responseMap.put("success", true);
            responseMap.put("testComplete", testComplete);
            responseMap.put("progress", test.getCompletionPercentage());
            responseMap.put("completedStimuli", test.getCompletedStimuli());
            responseMap.put("totalStimuli", test.getTotalStimuli());
            
            if (nextStimulus != null) {
                responseMap.put("nextStimulus", createStimulusResponse(nextStimulus, false)); // Patient view
            }
            
            return ResponseEntity.ok(responseMap);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error submitting response: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Get current test status and progress
     */
    @GetMapping("/test/{testId}/status")
    public ResponseEntity<Map<String, Object>> getTestStatus(@PathVariable Long testId) {
        try {
            Optional<NeuropathyTest> testOpt = neuropathyTestRepository.findById(testId);
            if (!testOpt.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            NeuropathyTest test = testOpt.get();
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("testId", test.getId());
            response.put("status", test.getTestStatus().toString());
            response.put("progress", test.getCompletionPercentage());
            response.put("completedStimuli", test.getCompletedStimuli());
            response.put("totalStimuli", test.getTotalStimuli());
            response.put("startedAt", test.getStartedAt());
            response.put("footSide", test.getFootSide().toString());
            response.put("isBaseline", test.getBaselineTest());
            
            if (test.getCompletedAt() != null) {
                response.put("completedAt", test.getCompletedAt());
            }
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error getting test status: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Get physician view of test results (includes stimulus data)
     */
    @GetMapping("/test/{testId}/physician-results")
    public ResponseEntity<Map<String, Object>> getPhysicianResults(@PathVariable Long testId) {
        try {
            Optional<NeuropathyTest> testOpt = neuropathyTestRepository.findById(testId);
            if (!testOpt.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            NeuropathyTest test = testOpt.get();
            List<TestStimulus> stimuli = testStimulusRepository.findByNeuropathyTestIdOrderByStimulusSequenceAsc(testId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("test", createTestSummary(test));
            response.put("stimuli", stimuli.stream().map(s -> createStimulusResponse(s, true)).toList()); // Physician view
            response.put("analytics", calculateTestAnalytics(stimuli));
            
            // Add ML-based analysis if test is completed
            if (test.getTestStatus() == NeuropathyTest.TestStatus.COMPLETED) {
                try {
                    MLModelService.NeuropathyAnalysisResult mlAnalysis = mlModelService.analyzeNeuropathyTest(testId);
                    response.put("mlAnalysis", createMLAnalysisResponse(mlAnalysis));
                } catch (Exception e) {
                    response.put("mlAnalysisError", "ML analysis failed: " + e.getMessage());
                }
            }
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error getting physician results: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Get patient's test history
     */
    @GetMapping("/patient/{patientId}/tests")
    public ResponseEntity<Map<String, Object>> getPatientTests(@PathVariable Long patientId) {
        try {
            List<NeuropathyTest> tests = neuropathyTestRepository.findByPatientIdOrderByStartedAtDesc(patientId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("total", tests.size());
            response.put("tests", tests.stream().map(this::createTestSummary).toList());
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error getting patient tests: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Get ML analysis for a completed test
     */
    @GetMapping("/test/{testId}/ml-analysis")
    public ResponseEntity<Map<String, Object>> getMLAnalysis(@PathVariable Long testId) {
        try {
            Optional<NeuropathyTest> testOpt = neuropathyTestRepository.findById(testId);
            if (!testOpt.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            NeuropathyTest test = testOpt.get();
            if (test.getTestStatus() != NeuropathyTest.TestStatus.COMPLETED) {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Test must be completed for ML analysis"
                ));
            }
            
            MLModelService.NeuropathyAnalysisResult mlAnalysis = mlModelService.analyzeNeuropathyTest(testId);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("mlAnalysis", createMLAnalysisResponse(mlAnalysis));
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error performing ML analysis: " + e.getMessage()
            ));
        }
    }
    
    /**
     * Get real-time risk assessment during test (for partial data)
     */
    @PostMapping("/test/{testId}/risk-assessment")
    public ResponseEntity<Map<String, Object>> getRealTimeRiskAssessment(@PathVariable Long testId) {
        try {
            Optional<NeuropathyTest> testOpt = neuropathyTestRepository.findById(testId);
            if (!testOpt.isPresent()) {
                return ResponseEntity.notFound().build();
            }
            
            // Get completed stimuli for partial analysis
            List<TestStimulus> completedStimuli = testStimulusRepository.findCompletedResponses(testId);
            
            if (completedStimuli.size() < 5) {
                return ResponseEntity.badRequest().body(Map.of(
                    "success", false,
                    "message", "Need at least 5 completed responses for risk assessment"
                ));
            }
            
            // Perform preliminary risk assessment
            Map<String, Object> riskAssessment = performPartialRiskAssessment(completedStimuli);
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("riskAssessment", riskAssessment);
            response.put("completedResponses", completedStimuli.size());
            response.put("note", "Preliminary assessment based on partial data");
            
            return ResponseEntity.ok(response);
            
        } catch (Exception e) {
            return ResponseEntity.badRequest().body(Map.of(
                "success", false,
                "message", "Error performing risk assessment: " + e.getMessage()
            ));
        }
    }
    
    // Helper methods
    private void generateTestStimuli(NeuropathyTest test) {
        // Generate a randomized sequence of stimuli including control trials
        TestStimulus.StimulusType[] types = {
            TestStimulus.StimulusType.VIBRATION,
            TestStimulus.StimulusType.TEMPERATURE_HOT,
            TestStimulus.StimulusType.TEMPERATURE_COLD,
            TestStimulus.StimulusType.PINPRICK
        };
        
        String[] locations = {
            "{\"x\": 25, \"y\": 80, \"region\": \"heel\"}",
            "{\"x\": 50, \"y\": 60, \"region\": \"arch\"}",
            "{\"x\": 30, \"y\": 20, \"region\": \"toe\"}",
            "{\"x\": 70, \"y\": 40, \"region\": \"ball\"}",
            "{\"x\": 45, \"y\": 50, \"region\": \"center\"}"
        };
        
        for (int i = 0; i < test.getTotalStimuli(); i++) {
            TestStimulus stimulus = new TestStimulus();
            stimulus.setNeuropathyTest(test);
            stimulus.setStimulusSequence(i + 1);
            
            // 15% chance of no-stimulus control trial
            if (Math.random() < 0.15) {
                stimulus.setStimulusType(TestStimulus.StimulusType.NONE);
                stimulus.setNoStimulusTrial(true);
                stimulus.setStimulusIntensity(0.0);
            } else {
                stimulus.setStimulusType(types[(int) (Math.random() * types.length)]);
                stimulus.setNoStimulusTrial(false);
                stimulus.setStimulusIntensity(0.3 + (Math.random() * 0.7)); // 0.3 to 1.0 intensity
            }
            
            stimulus.setStimulusLocation(locations[(int) (Math.random() * locations.length)]);
            stimulus.setStimulusDurationMs(1000 + (int) (Math.random() * 2000)); // 1-3 seconds
            
            testStimulusRepository.save(stimulus);
        }
    }
    
    private Map<String, Object> createStimulusResponse(TestStimulus stimulus, boolean physicianView) {
        Map<String, Object> response = new HashMap<>();
        response.put("id", stimulus.getId());
        response.put("sequence", stimulus.getStimulusSequence());
        response.put("noStimulusTrial", stimulus.getNoStimulusTrial());
        
        if (physicianView) {
            // Physician sees actual stimulus data
            response.put("actualStimulusType", stimulus.getStimulusType());
            response.put("actualIntensity", stimulus.getStimulusIntensity());
            response.put("actualLocation", stimulus.getStimulusLocation());
            response.put("duration", stimulus.getStimulusDurationMs());
        }
        
        // Both see patient responses
        response.put("patientFeltSensation", stimulus.getPatientFeltSensation());
        response.put("perceivedIntensity", stimulus.getPerceivedIntensity());
        response.put("perceivedType", stimulus.getPerceivedType());
        response.put("perceivedLocation", stimulus.getPerceivedLocation());
        response.put("responseConfidence", stimulus.getResponseConfidence());
        response.put("responseTime", stimulus.getResponseTime());
        
        if (physicianView) {
            response.put("correctDetection", stimulus.isCorrectDetection());
            response.put("typeAccuracy", stimulus.isTypeMatchCorrect());
            response.put("intensityAccuracy", stimulus.getIntensityAccuracy());
        }
        
        return response;
    }
    
    private Map<String, Object> createTestSummary(NeuropathyTest test) {
        Map<String, Object> summary = new HashMap<>();
        summary.put("id", test.getId());
        summary.put("status", test.getTestStatus());
        summary.put("footSide", test.getFootSide());
        summary.put("isBaseline", test.getBaselineTest());
        summary.put("startedAt", test.getStartedAt());
        summary.put("completedAt", test.getCompletedAt());
        summary.put("progress", test.getCompletionPercentage());
        summary.put("patientName", test.getPatient().getFullName());
        summary.put("deviceId", test.getDevice().getId());
        return summary;
    }
    
    private Map<String, Object> calculateTestAnalytics(List<TestStimulus> stimuli) {
        Map<String, Object> analytics = new HashMap<>();
        
        long totalResponses = stimuli.stream()
            .filter(s -> s.getResponseTime() != null)
            .count();
        
        long correctDetections = stimuli.stream()
            .filter(TestStimulus::isCorrectDetection)
            .count();
        
        double accuracy = totalResponses > 0 ? (double) correctDetections / totalResponses : 0.0;
        
        analytics.put("totalStimuli", stimuli.size());
        analytics.put("completedResponses", totalResponses);
        analytics.put("accuracy", accuracy);
        analytics.put("correctDetections", correctDetections);
        
        return analytics;
    }
    
    private String getTestInstructions() {
        return "Welcome to the neuropathy test. During this test, you may feel different sensations on your foot including vibration, temperature changes, or pinprick sensations. " +
               "Please indicate when you feel a sensation and describe what you felt. " +
               "IMPORTANT: Sometimes no stimulus will be present - it is completely normal to not feel anything during some trials. " +
               "Please respond honestly about what you feel or don't feel. The test will take approximately 15 minutes.";
    }
    
    private Map<String, Object> createMLAnalysisResponse(MLModelService.NeuropathyAnalysisResult analysis) {
        Map<String, Object> response = new HashMap<>();
        response.put("severity", analysis.getSeverity().toString());
        response.put("riskScore", analysis.getRiskScore());
        response.put("confidenceScore", analysis.getConfidenceScore());
        response.put("detailedAnalysis", analysis.getDetailedAnalysis());
        response.put("recommendations", analysis.getRecommendations());
        
        // Add severity interpretation
        response.put("severityInterpretation", getSeverityInterpretation(analysis.getSeverity()));
        response.put("riskLevel", getRiskLevel(analysis.getRiskScore()));
        
        return response;
    }
    
    private String getSeverityInterpretation(MLModelService.NeuropathySeverity severity) {
        switch (severity) {
            case SEVERE:
                return "Severe neuropathy detected - immediate medical attention recommended";
            case MODERATE:
                return "Moderate neuropathy - enhanced monitoring and treatment adjustments needed";
            case MILD:
                return "Mild neuropathy symptoms - continue current management with regular monitoring";
            case NORMAL:
                return "No significant neuropathy detected - maintain preventive care";
            default:
                return "Unable to determine severity";
        }
    }
    
    private String getRiskLevel(double riskScore) {
        if (riskScore >= 0.8) return "HIGH";
        if (riskScore >= 0.6) return "MODERATE";
        if (riskScore >= 0.3) return "LOW";
        return "MINIMAL";
    }
    
    private Map<String, Object> performPartialRiskAssessment(List<TestStimulus> completedStimuli) {
        Map<String, Object> assessment = new HashMap<>();
        
        // Calculate basic metrics from available data
        long totalResponses = completedStimuli.size();
        long correctDetections = completedStimuli.stream()
            .filter(TestStimulus::isCorrectDetection)
            .count();
        
        double partialAccuracy = totalResponses > 0 ? (double) correctDetections / totalResponses : 0.0;
        
        // Calculate false positive rate from control trials
        long controlTrials = completedStimuli.stream().filter(TestStimulus::getNoStimulusTrial).count();
        long falsePositives = completedStimuli.stream()
            .filter(s -> s.getNoStimulusTrial() && s.getPatientFeltSensation() != null && s.getPatientFeltSensation())
            .count();
        
        double falsePositiveRate = controlTrials > 0 ? (double) falsePositives / controlTrials : 0.0;
        
        // Preliminary risk calculation
        double preliminaryRisk = (1.0 - partialAccuracy) * 0.7 + falsePositiveRate * 0.3;
        
        assessment.put("preliminaryRiskScore", preliminaryRisk);
        assessment.put("partialAccuracy", partialAccuracy);
        assessment.put("falsePositiveRate", falsePositiveRate);
        assessment.put("samplesAnalyzed", totalResponses);
        assessment.put("riskLevel", getRiskLevel(preliminaryRisk));
        
        // Add early warning if risk is high
        if (preliminaryRisk > 0.7) {
            assessment.put("earlyWarning", "Elevated risk detected - consider clinical review");
        }
        
        return assessment;
    }
}