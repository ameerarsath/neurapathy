package com.smartshoe.api.service;

import com.smartshoe.api.entity.NeuropathyTest;
import com.smartshoe.api.repository.NeuropathyTestRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

/**
 * Service for handling neuropathy test operations
 */
@Service
public class NeuropathyTestService {
    
    @Autowired
    private NeuropathyTestRepository neuropathyTestRepository;
    
    public NeuropathyTest getTestById(Long testId) {
        Optional<NeuropathyTest> test = neuropathyTestRepository.findById(testId);
        if (!test.isPresent()) {
            throw new IllegalArgumentException("Test not found with ID: " + testId);
        }
        return test.get();
    }
    
    public NeuropathyTest saveTest(NeuropathyTest test) {
        return neuropathyTestRepository.save(test);
    }
}