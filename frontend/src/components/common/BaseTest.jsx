import React, { useState, useEffect } from 'react';
import { useTestSession } from '../../contexts/TestSessionContext';
import FootDiagram from './FootDiagram';
import Button from './Button';
import Card from './Card';
import Modal from './Modal';

const BaseTest = ({ testType }) => {
  const { currentTest, startNewSession, recordResponse, endSession } = useTestSession();
  const [testStep, setTestStep] = useState('initial'); // 'initial', 'feeling', 'location', 'intensity'
  const [elapsedTime, setElapsedTime] = useState(0);
  const [selectedLocations, setSelectedLocations] = useState([]);
  const [intensityRating, setIntensityRating] = useState(null);

  useEffect(() => {
    let timer;
    if (currentTest.isActive && testStep !== 'initial') {
      timer = setInterval(() => {
        setElapsedTime((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [currentTest.isActive, testStep]);  const handleStartTest = () => {
    startNewSession(testType);
    setTestStep('feeling');
  };

  const handleLocationSelect = (locationId) => {
    setSelectedLocations((prev) => {
      const newLocations = prev.includes(locationId)
        ? prev.filter((id) => id !== locationId)
        : [...prev, locationId];
      recordResponse({ location: newLocations });
      return newLocations;
    });
  };

  const handleFeltResponse = (felt) => {
    recordResponse({ feltStimulus: felt });
    if (felt) {
      setTestStep('location');
    } else {
      handleTestComplete();
    }
  };
  const handleLocationConfirm = () => {
    setTestStep('intensity');
  };

  const handleIntensitySelect = (rating) => {
    setIntensityRating(rating);
    recordResponse({
      intensity: rating,
      location: selectedLocations,
    });
    handleTestComplete();
  };

  const handleTestComplete = () => {
    endSession();
    setTestStep('initial');
    setShowInstructions(true);
    setSelectedLocations([]);
    setIntensityRating(null);
    setElapsedTime(0);
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return \`\${mins}:\${secs.toString().padStart(2, '0')}\`;
  };

  if (testStep === 'initial') {
    return (
      <Modal isOpen={true} onClose={() => {}}>
        <div className="p-6 max-w-2xl mx-auto">
          <h2 className="text-2xl font-bold mb-4">{testType} Sensory Test</h2>
          <div className="space-y-4 text-gray-600">
            <p className="font-medium">Welcome to your sensory test. Here's what to expect:</p>
            <ul className="list-disc pl-5 space-y-2">
              <li>You will be asked if you can feel any {testType.toLowerCase()} sensation in your foot</li>
              <li>If you feel something, you'll be asked to:</li>
              <ul className="list-circle pl-5 space-y-1">
                <li>Indicate where on your foot you felt it</li>
                <li>Rate how strong the sensation was (1-10)</li>
              </ul>
              <li className="font-semibold text-blue-600">Important: Sometimes, no stimulus will be present. It is completely okay to not feel anything.</li>
            </ul>
            <div className="bg-gray-50 p-4 rounded-lg mt-4">
              <p className="font-medium">This test will help track your sensory response over time.</p>
            </div>
          </div>
          <div className="mt-6 flex justify-end">
            <Button onClick={handleStartTest} className="bg-blue-600 text-white">
              Start Test
            </Button>
          </div>
        </div>
      </Modal>
    );
  }

  return (
    <div className="max-w-2xl mx-auto p-4">
      <Card>
        <div className="text-right mb-4">
          <span className="font-mono">Time: {formatTime(elapsedTime)}</span>
        </div>
          {testStep === 'feeling' && (
          <div className="space-y-4">
            <h2 className="text-xl font-bold text-center mb-6">
              Do you feel any {testType.toLowerCase()} sensation in your foot?
            </h2>
            <div className="flex justify-center space-x-4">
              <Button onClick={() => handleFeltResponse(true)}>Yes, I feel something</Button>
              <Button onClick={() => handleFeltResponse(false)}>No, I don't feel anything</Button>
            </div>
          </div>
        )}

        {testStep === 'location' && (
          <div className="space-y-6">
            <h3 className="text-lg font-semibold">Where did you feel it?</h3>
            <div className="flex justify-around">
              <div>
                <h4 className="text-center mb-2">Top View</h4>
                <FootDiagram
                  side="top"
                  onLocationSelect={handleLocationSelect}
                  selectedPoints={selectedLocations}
                />
              </div>
              <div>
                <h4 className="text-center mb-2">Bottom View</h4>
                <FootDiagram
                  side="bottom"
                  onLocationSelect={handleLocationSelect}
                  selectedPoints={selectedLocations}
                />
              </div>
            </div>

            <div className="mt-6">
              <h3 className="text-lg font-semibold mb-4">
                How intense was the sensation? (1-10)
              </h3>
              <div className="flex justify-center space-x-2">
                {[...Array(10)].map((_, i) => (
                  <Button
                    key={i + 1}
                    onClick={() => handleIntensitySelect(i + 1)}
                    className={\`w-10 h-10 \${
                      intensityRating === i + 1 ? 'bg-blue-500 text-white' : ''
                    }\`}
                  >
                    {i + 1}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};

export default BaseTest;
