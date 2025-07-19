import React, { useState, useMemo } from 'react';
import { useTestSession } from '../../contexts/TestSessionContext';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import TestChart from './components/TestChart';
import ResultSummary from './components/ResultSummary';
import LocationHeatmap from './components/LocationHeatmap';
import RetestRecommendation from './components/RetestRecommendation';

const TestResults = () => {
  const [selectedType, setSelectedType] = useState('all');
  const { testHistory } = useTestSession();

  const groupedTests = useMemo(() => {
    return testHistory.reduce((acc, test) => {
      const type = test.testType.toLowerCase();
      if (!acc[type]) {
        acc[type] = [];
      }
      acc[type].push(test);
      return acc;
    }, {});
  }, [testHistory]);

  const filteredTests = useMemo(() => {
    if (selectedType === 'all') {
      return testHistory;
    }
    return groupedTests[selectedType] || [];
  }, [selectedType, groupedTests, testHistory]);

  const testTypes = useMemo(() => {
    return ['all', ...Object.keys(groupedTests)];
  }, [groupedTests]);

  const handleExport = () => {
    const exportData = {
      exportDate: new Date().toISOString(),
      testResults: filteredTests.map(test => ({
        ...test,
        startTime: new Date(test.startTime).toISOString(),
        endTime: test.endTime ? new Date(test.endTime).toISOString() : null
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuropathy-test-results-${selectedType}-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div className="px-4 sm:px-0">
        <div className="sm:flex sm:items-center">
          <div className="sm:flex-auto">
            <h1 className="text-xl font-semibold text-gray-900">Test Results</h1>
            <p className="mt-2 text-sm text-gray-700">
              View and analyze your neuropathy test results over time.
            </p>
          </div>
          <div className="mt-4 sm:mt-0 sm:ml-16 sm:flex-none space-x-4">
            <Button onClick={handleExport}>
              Export Results
            </Button>
          </div>
        </div>

        <div className="mt-4 flex space-x-2">
          {testTypes.map(type => (
            <Button
              key={type}
              onClick={() => setSelectedType(type)}
              variant={selectedType === type ? 'primary' : 'secondary'}
            >
              {type.charAt(0).toUpperCase() + type.slice(1)}
            </Button>
          ))}
        </div>

        {filteredTests.length === 0 ? (
          <Card>
            <div className="p-6 text-center">
              <p className="text-gray-500">No test results available.</p>
            </div>
          </Card>
        ) : (
          <div className="space-y-8 mt-8">
            <RetestRecommendation tests={filteredTests} />

            <Card>
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Progress Overview</h2>
                <TestChart tests={filteredTests} />
              </div>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <Card>
                <ResultSummary tests={filteredTests} />
              </Card>
              <Card>
                <LocationHeatmap tests={filteredTests} />
              </Card>
            </div>

            <Card>
              <div className="p-6">
                <h2 className="text-lg font-medium text-gray-900 mb-4">Recent Tests</h2>
                <div className="divide-y divide-gray-200">
                  {filteredTests.slice(-5).reverse().map((test) => (
                    <div key={test.sessionId} className="py-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <p className="text-sm font-medium text-gray-900">{test.testType} Test</p>
                          <p className="text-sm text-gray-500">
                            {new Date(test.startTime).toLocaleDateString()} at{' '}
                            {new Date(test.startTime).toLocaleTimeString()}
                          </p>
                        </div>
                        <div className="text-right">
                          <p className={`text-sm font-medium ${
                            test.patientResponse.feltStimulus === test.stimulusApplied
                              ? 'text-green-600'
                              : 'text-red-600'
                          }`}>
                            {test.patientResponse.feltStimulus === test.stimulusApplied
                              ? 'Correct Response'
                              : 'Incorrect Response'}
                          </p>
                          {test.patientResponse.intensity && (
                            <p className="text-sm text-gray-500">
                              Intensity: {test.patientResponse.intensity}/10
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};

export default TestResults;