package com.smartshoe.api.repository;

import com.smartshoe.api.entity.TestStimulus;
import com.smartshoe.api.entity.NeuropathyTest;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface TestStimulusRepository extends JpaRepository<TestStimulus, Long> {
    
    List<TestStimulus> findByNeuropathyTestOrderByStimulusSequenceAsc(NeuropathyTest neuropathyTest);
    
    List<TestStimulus> findByNeuropathyTestIdOrderByStimulusSequenceAsc(Long neuropathyTestId);
    
    @Query("SELECT ts FROM TestStimulus ts WHERE ts.neuropathyTest.id = :testId AND ts.responseTime IS NULL ORDER BY ts.stimulusSequence ASC")
    List<TestStimulus> findPendingResponses(@Param("testId") Long testId);
    
    @Query("SELECT ts FROM TestStimulus ts WHERE ts.neuropathyTest.id = :testId AND ts.stimulusSequence = :sequence")
    Optional<TestStimulus> findByTestIdAndSequence(@Param("testId") Long testId, @Param("sequence") Integer sequence);
    
    @Query("SELECT COUNT(ts) FROM TestStimulus ts WHERE ts.neuropathyTest.id = :testId AND ts.responseTime IS NOT NULL")
    Long countCompletedResponses(@Param("testId") Long testId);
    
    @Query("SELECT ts FROM TestStimulus ts WHERE ts.neuropathyTest.id = :testId AND ts.patientFeltSensation = true AND ts.noStimulusTrial = false")
    List<TestStimulus> findCorrectDetections(@Param("testId") Long testId);
    
    @Query("SELECT ts FROM TestStimulus ts WHERE ts.neuropathyTest.id = :testId AND ts.patientFeltSensation = false AND ts.noStimulusTrial = true")
    List<TestStimulus> findCorrectRejections(@Param("testId") Long testId);
    
    @Query("SELECT ts FROM TestStimulus ts WHERE ts.neuropathyTest.id = :testId AND ts.responseTime IS NOT NULL ORDER BY ts.stimulusSequence ASC")
    List<TestStimulus> findCompletedResponses(@Param("testId") Long testId);
}