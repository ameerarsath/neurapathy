import React, { useState, useEffect } from 'react';
import { useTestSession } from '../../contexts/TestSessionContext';
import FootDiagram from './FootDiagram';
import Card from './Card';
import Button from './Button';
import Modal from './Modal';

const PhysicianTestView = () => {
  const { currentTest, testHistory, recordStimulus } = useTestSession();
  const [selectedPoints, setSelectedPoints] = useState([]);
  const [stimulusIntensity, setStimulusIntensity] = useState(5);
  const [showStimulusControls, setShowStimulusControls] = useState(true);

  useEffect(() => {
    if (!currentTest.isActive) {
      setSelectedPoints([]);
      setStimulusIntensity(5);
      setShowStimulusControls(true);
    }
  }, [currentTest.isActive]);
    
    return (
      <Card key={test.sessionId} className="mb-4">
        <div className="p-4">
          <div className="flex justify-between mb-4">
            <h3 className="text-lg font-semibold">
              {test.testType} Test - {new Date(test.startTime).toLocaleTimeString()}
            </h3>
            <span className={`px-3 py-1 rounded ${
              accuracy >= 80 ? 'bg-green-100 text-green-800' :
              accuracy >= 50 ? 'bg-yellow-100 text-yellow-800' :
                            'bg-red-100 text-red-800'
            }`}>
              {accuracy}% Accuracy
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="font-medium mb-2">Stimulus Applied</h4>
              <div className="flex space-x-4">
                <div>
                  <p className="text-sm mb-1">Top View</p>
                  <FootDiagram
                    side="top"
                    isPhysicianView={true}
                    stimulusPoints={test.stimulusPoints || []}
                  />
                </div>
                <div>
                  <p className="text-sm mb-1">Bottom View</p>
                  <FootDiagram
                    side="bottom"
                    isPhysicianView={true}
                    stimulusPoints={test.stimulusPoints || []}
                  />
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-medium mb-2">Patient Response</h4>
              <div className="flex space-x-4">
                <div>
                  <p className="text-sm mb-1">Top View</p>
                  <FootDiagram
                    side="top"
                    selectedPoints={test.patientResponse.location || []}
                  />
                </div>
                <div>
                  <p className="text-sm mb-1">Bottom View</p>
                  <FootDiagram
                    side="bottom"
                    selectedPoints={test.patientResponse.location || []}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4">
            <h4 className="font-medium mb-2">Test Details</h4>
            <ul className="space-y-2">
              <li>
                <span className="font-medium">Stimulus Present:</span> {test.stimulusApplied ? 'Yes' : 'No'}
              </li>
              <li>
                <span className="font-medium">Patient Felt:</span> {test.patientResponse.feltStimulus ? 'Yes' : 'No'}
              </li>
              {test.patientResponse.feltStimulus && (
                <li>
                  <span className="font-medium">Reported Intensity:</span> {test.patientResponse.intensity}/10
                </li>
              )}
              <li>
                <span className="font-medium">Duration:</span> {
                  Math.round((new Date(test.endTime) - new Date(test.startTime)) / 1000)
                } seconds
              </li>
            </ul>
          </div>
        </div>
      </Card>
    );
  };

  const calculateAccuracy = (test) => {
    if (!test.stimulusApplied && !test.patientResponse.feltStimulus) return 100;
    if (!test.stimulusApplied && test.patientResponse.feltStimulus) return 0;
    if (test.stimulusApplied && !test.patientResponse.feltStimulus) return 0;
    
    // Compare locations
    const stimulusSet = new Set(test.stimulusPoints || []);
    const responseSet = new Set(test.patientResponse.location || []);
    
    let matchingPoints = 0;
    for (const point of responseSet) {
      if (stimulusSet.has(point)) matchingPoints++;
    }
    
    const locationAccuracy = (matchingPoints / Math.max(stimulusSet.size, responseSet.size)) * 100;
    const intensityWeight = test.patientResponse.intensity ? 
      (1 - Math.abs(test.stimulusIntensity - test.patientResponse.intensity) / 10) : 0;
    
    return Math.round((locationAccuracy * 0.7 + intensityWeight * 0.3));
  };

  const handlePointSelect = (point) => {
    setSelectedPoints(prev => {
      const newPoints = prev.includes(point) 
        ? prev.filter(p => p !== point)
        : [...prev, point];
      return newPoints;
    });
  };

  const handleApplyStimulus = () => {
    recordStimulus(true);
    setShowStimulusControls(false);
  };

  const handleNoStimulus = () => {
    recordStimulus(false);
    setShowStimulusControls(false);
    setSelectedPoints([]);
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <Card>
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold">Physician Control Panel</h2>
            <div className={`px-3 py-1 rounded-full ${
              currentTest.isActive 
                ? 'bg-green-100 text-green-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {currentTest.isActive ? 'Test in Progress' : 'Waiting for Test'}
            </div>
          </div>

          {currentTest.isActive && showStimulusControls && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium mb-4">Select Stimulus Points</h3>
                <div className="grid grid-cols-2 gap-8">
                  <div>
                    <p className="text-center mb-2">Top View</p>
                    <FootDiagram
                      side="top"
                      onLocationSelect={handlePointSelect}
                      selectedPoints={selectedPoints}
                      isPhysicianView={true}
                    />
                  </div>
                  <div>
                    <p className="text-center mb-2">Bottom View</p>
                    <FootDiagram
                      side="bottom"
                      onLocationSelect={handlePointSelect}
                      selectedPoints={selectedPoints}
                      isPhysicianView={true}
                    />
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium mb-4">Set Stimulus Intensity</h3>
                <div className="flex items-center space-x-4">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={stimulusIntensity}
                    onChange={(e) => setStimulusIntensity(Number(e.target.value))}
                    className="w-64"
                  />
                  <span className="font-medium">{stimulusIntensity}</span>
                </div>
              </div>

              <div className="flex justify-center space-x-4">
                <Button
                  onClick={handleApplyStimulus}
                  disabled={selectedPoints.length === 0}
                  className="bg-blue-600 text-white"
                >
                  Apply Stimulus
                </Button>
                <Button
                  onClick={handleNoStimulus}
                  className="bg-gray-200"
                >
                  No Stimulus
                </Button>
              </div>
            </div>
          )}

          {currentTest.isActive && !showStimulusControls && (
            <div className="text-center py-8">
              <h3 className="text-lg font-medium mb-2">Waiting for Patient Response</h3>
              <p className="text-gray-600">
                {currentTest.stimulusApplied 
                  ? `Stimulus applied to ${selectedPoints.length} location(s) with intensity ${stimulusIntensity}`
                  : 'No stimulus applied'}
              </p>
            </div>
          )}

          {!currentTest.isActive && currentTest.patientResponse && (
            <div className="space-y-4">
              <h3 className="text-lg font-medium">Last Test Result</h3>
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <div className="p-4">
                    <h4 className="font-medium mb-2">Stimulus</h4>
                    <p>Applied: {currentTest.stimulusApplied ? 'Yes' : 'No'}</p>
                    {currentTest.stimulusApplied && (
                      <>
                        <p>Intensity: {stimulusIntensity}/10</p>
                        <p>Locations: {selectedPoints.length}</p>
                      </>
                    )}
                  </div>
                </Card>
                <Card>
                  <div className="p-4">
                    <h4 className="font-medium mb-2">Response</h4>
                    <p>Felt: {currentTest.patientResponse.feltStimulus ? 'Yes' : 'No'}</p>
                    {currentTest.patientResponse.feltStimulus && (
                      <>
                        <p>Reported Intensity: {currentTest.patientResponse.intensity}/10</p>
                        <p>Locations Identified: {currentTest.patientResponse.location?.length || 0}</p>
                      </>
                    )}
                  </div>
                </Card>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default PhysicianTestView;
