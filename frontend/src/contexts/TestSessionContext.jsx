import { createContext, useContext, useState } from 'react';

const TestSessionContext = createContext();

export const TestSessionProvider = ({ children }) => {
  const [currentTest, setCurrentTest] = useState({
    sessionId: null,
    testType: null, // 'vibration', 'temperature', 'pinprick'
    isActive: false,
    startTime: null,
    stimulusApplied: false, // Only visible in physician portal
    patientResponse: {
      feltStimulus: false,
      location: null,
      intensity: null,
    },
  });

  const [testHistory, setTestHistory] = useState([]);

  const startNewSession = (testType) => {
    setCurrentTest({
      sessionId: Date.now(),
      testType,
      isActive: true,
      startTime: new Date(),
      stimulusApplied: false,
      patientResponse: {
        feltStimulus: false,
        location: null,
        intensity: null,
      },
    });
  };

  const recordStimulus = (applied) => {
    setCurrentTest(prev => ({
      ...prev,
      stimulusApplied: applied,
    }));
  };

  const recordResponse = (response) => {
    setCurrentTest(prev => ({
      ...prev,
      patientResponse: {
        ...prev.patientResponse,
        ...response,
      },
    }));
  };

  const endSession = () => {
    setTestHistory(prev => [...prev, { ...currentTest, endTime: new Date() }]);
    setCurrentTest({
      sessionId: null,
      testType: null,
      isActive: false,
      startTime: null,
      stimulusApplied: false,
      patientResponse: {
        feltStimulus: false,
        location: null,
        intensity: null,
      },
    });
  };

  return (
    <TestSessionContext.Provider
      value={{
        currentTest,
        testHistory,
        startNewSession,
        recordStimulus,
        recordResponse,
        endSession,
      }}
    >
      {children}
    </TestSessionContext.Provider>
  );
};

export const useTestSession = () => {
  const context = useContext(TestSessionContext);
  if (!context) {
    throw new Error('useTestSession must be used within a TestSessionProvider');
  }
  return context;
};
