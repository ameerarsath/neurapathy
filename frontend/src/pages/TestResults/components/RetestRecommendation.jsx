import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import Card from '../../../components/common/Card';

const RetestRecommendation = ({ tests }) => {
  const recommendations = useMemo(() => {
    if (!tests.length) return [];

    const recentTests = tests.slice(-10);
    const recommendations = [];
    
    // Check for inconsistent responses
    const inconsistentLocations = findInconsistentLocations(recentTests);
    if (inconsistentLocations.length > 0) {
      recommendations.push({
        priority: 'high',
        message: `Inconsistent responses detected at locations: ${inconsistentLocations.join(', ')}`,
        reason: 'Response variability exceeds normal threshold'
      });
    }

    // Check for declining accuracy trend
    const accuracyTrend = calculateAccuracyTrend(recentTests);
    if (accuracyTrend < -15) { // 15% decline threshold
      recommendations.push({
        priority: 'medium',
        message: 'Notable decline in test accuracy',
        reason: `${Math.abs(accuracyTrend)}% decrease in accuracy over recent tests`
      });
    }

    // Check test frequency
    const lastTestDate = new Date(tests[tests.length - 1].startTime);
    const daysSinceLastTest = (new Date() - lastTestDate) / (1000 * 60 * 60 * 24);
    if (daysSinceLastTest > 14) { // 2 weeks threshold
      recommendations.push({
        priority: 'low',
        message: 'Regular testing reminder',
        reason: `Last test was ${Math.floor(daysSinceLastTest)} days ago`
      });
    }

    return recommendations;
  }, [tests]);

  const findInconsistentLocations = (tests) => {
    const locationResponses = {};
    
    tests.forEach(test => {
      const stimulusSet = new Set(test.stimulusPoints || []);
      const responseSet = new Set(test.patientResponse.location || []);
      
      [...new Set([...stimulusSet, ...responseSet])].forEach(location => {
        if (!locationResponses[location]) {
          locationResponses[location] = [];
        }
        locationResponses[location].push(
          stimulusSet.has(location) === responseSet.has(location)
        );
      });
    });

    return Object.entries(locationResponses)
      .filter(([_, responses]) => {
        if (responses.length < 3) return false;
        const inconsistencyRate = responses
          .filter(r => !r).length / responses.length;
        return inconsistencyRate > 0.4; // 40% inconsistency threshold
      })
      .map(([location]) => location);
  };

  const calculateAccuracyTrend = (tests) => {
    if (tests.length < 2) return 0;
    const firstTests = tests.slice(0, 3);
    const lastTests = tests.slice(-3);
    
    const avgFirst = firstTests.reduce((sum, test) => 
      sum + calculateAccuracy(test), 0) / firstTests.length;
    const avgLast = lastTests.reduce((sum, test) => 
      sum + calculateAccuracy(test), 0) / lastTests.length;
    
    return avgLast - avgFirst;
  };

  const calculateAccuracy = (test) => {
    if (!test.stimulusApplied && !test.patientResponse.feltStimulus) return 100;
    if (!test.stimulusApplied && test.patientResponse.feltStimulus) return 0;
    if (test.stimulusApplied && !test.patientResponse.feltStimulus) return 0;
    
    const stimulusSet = new Set(test.stimulusPoints || []);
    const responseSet = new Set(test.patientResponse.location || []);
    let matchingPoints = 0;
    
    for (const point of responseSet) {
      if (stimulusSet.has(point)) matchingPoints++;
    }
    
    return (matchingPoints / Math.max(stimulusSet.size, responseSet.size)) * 100;
  };

  if (!recommendations.length) {
    return null;
  }

  return (
    <Card>
      <div className="p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">
          Test Recommendations
        </h2>
        <div className="space-y-4">
          {recommendations.map((rec, index) => (
            <div
              key={index}
              className={`p-4 rounded-lg ${
                rec.priority === 'high'
                  ? 'bg-red-50 border border-red-200'
                  : rec.priority === 'medium'
                  ? 'bg-yellow-50 border border-yellow-200'
                  : 'bg-blue-50 border border-blue-200'
              }`}
            >
              <div className="flex items-start">
                <div className="flex-shrink-0">
                  <svg
                    className={`h-5 w-5 ${
                      rec.priority === 'high'
                        ? 'text-red-400'
                        : rec.priority === 'medium'
                        ? 'text-yellow-400'
                        : 'text-blue-400'
                    }`}
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className={`text-sm font-medium ${
                    rec.priority === 'high'
                      ? 'text-red-800'
                      : rec.priority === 'medium'
                      ? 'text-yellow-800'
                      : 'text-blue-800'
                  }`}>
                    {rec.message}
                  </h3>
                  <div className="mt-2 text-sm text-gray-600">
                    {rec.reason}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};

RetestRecommendation.propTypes = {
  tests: PropTypes.arrayOf(
    PropTypes.shape({
      startTime: PropTypes.string.isRequired,
      stimulusApplied: PropTypes.bool.isRequired,
      stimulusPoints: PropTypes.arrayOf(PropTypes.string),
      patientResponse: PropTypes.shape({
        feltStimulus: PropTypes.bool.isRequired,
        location: PropTypes.arrayOf(PropTypes.string)
      }).isRequired
    })
  ).isRequired
};

export default RetestRecommendation;
