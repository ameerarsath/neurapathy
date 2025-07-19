import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import Card from '../../../components/common/Card';

const ResultSummary = ({ tests }) => {
  const stats = useMemo(() => calculateStats(tests), [tests]);
  const trend = useMemo(() => calculateTrend(stats.trendsData), [stats.trendsData]);
  const trendSummary = useMemo(() => generateTrendSummary(trend), [trend]);

  return (
    <div className="p-6">
      <h2 className="text-lg font-medium text-gray-900 mb-4">Result Summary</h2>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Total Tests</h3>
          <p className="mt-1 text-2xl font-semibold text-gray-900">{stats.totalTests}</p>
          <div className="mt-1">
            <span className="text-xs text-gray-500">
              Last test: {tests.length > 0 ? new Date(tests[tests.length - 1].startTime).toLocaleDateString() : 'N/A'}
            </span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Average Accuracy</h3>
          <p className="mt-1 text-2xl font-semibold text-gray-900">{stats.averageAccuracy}%</p>
          <div className="mt-1">
            <span className={`text-xs ${trend.accuracy >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {trend.accuracy > 0 ? '↑' : trend.accuracy < 0 ? '↓' : '→'} {Math.abs(trend.accuracy)}% recent trend
            </span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Average Intensity</h3>
          <p className="mt-1 text-2xl font-semibold text-gray-900">{stats.averageIntensity}/10</p>
          <div className="mt-1">
            <span className={`text-xs ${trend.intensity >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {trend.intensity > 0 ? '↑' : trend.intensity < 0 ? '↓' : '→'} {Math.abs(trend.intensity)} recent trend
            </span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">False Positives</h3>
          <p className="mt-1 text-2xl font-semibold text-gray-900">{stats.falsePositives}</p>
          <div className="mt-1">
            <span className="text-xs text-gray-500">
              {((stats.falsePositives / stats.totalTests) * 100).toFixed(1)}% of tests
            </span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">False Negatives</h3>
          <p className="mt-1 text-2xl font-semibold text-gray-900">{stats.falseNegatives}</p>
          <div className="mt-1">
            <span className="text-xs text-gray-500">
              {((stats.falseNegatives / stats.totalTests) * 100).toFixed(1)}% of tests
            </span>
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500">Response Time</h3>
          <p className="mt-1 text-2xl font-semibold text-gray-900">
            {stats.averageResponseTime.toFixed(1)}s
          </p>
          <div className="mt-1">
            <span className="text-xs text-gray-500">
              Range: {stats.minResponseTime.toFixed(1)}s - {stats.maxResponseTime.toFixed(1)}s
            </span>
          </div>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-900 mb-2">Trend Analysis</h3>
        <p className="text-sm text-gray-600">{trendSummary}</p>
      </div>
    </div>
  );
};

const calculateStats = (tests) => {
  if (!tests.length) {
    return {
      totalTests: 0,
      averageAccuracy: 0,
      averageIntensity: 0,
      falsePositives: 0,
      falseNegatives: 0,
      averageResponseTime: 0,
      minResponseTime: 0,
      maxResponseTime: 0,
      trendsData: []
    };
  }

  const stats = tests.reduce((acc, test) => {
    // Calculate accuracy
    const accuracy = calculateAccuracy(test);
    acc.totalAccuracy += accuracy;
    
    // Track false positives and negatives
    if (!test.stimulusApplied && test.patientResponse.feltStimulus) {
      acc.falsePositives++;
    }
    if (test.stimulusApplied && !test.patientResponse.feltStimulus) {
      acc.falseNegatives++;
    }

    // Track intensity
    if (test.patientResponse.intensity) {
      acc.totalIntensity += test.patientResponse.intensity;
      acc.intensityCount++;
    }

    // Track response time
    const responseTime = (new Date(test.endTime) - new Date(test.startTime)) / 1000;
    acc.totalResponseTime += responseTime;
    acc.minResponseTime = Math.min(acc.minResponseTime || responseTime, responseTime);
    acc.maxResponseTime = Math.max(acc.maxResponseTime || responseTime, responseTime);

    // Track trends
    acc.trendsData.push({
      date: new Date(test.startTime),
      accuracy,
      intensity: test.patientResponse.intensity || 0,
      responseTime
    });

    return acc;
  }, {
    totalAccuracy: 0,
    totalIntensity: 0,
    intensityCount: 0,
    falsePositives: 0,
    falseNegatives: 0,
    totalResponseTime: 0,
    minResponseTime: null,
    maxResponseTime: null,
    trendsData: []
  });

  return {
    totalTests: tests.length,
    averageAccuracy: Math.round(stats.totalAccuracy / tests.length),
    averageIntensity: stats.intensityCount ? 
      Math.round((stats.totalIntensity / stats.intensityCount) * 10) / 10 : 0,
    falsePositives: stats.falsePositives,
    falseNegatives: stats.falseNegatives,
    averageResponseTime: stats.totalResponseTime / tests.length,
    minResponseTime: stats.minResponseTime,
    maxResponseTime: stats.maxResponseTime,
    trendsData: stats.trendsData.sort((a, b) => a.date - b.date)
  };
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

const calculateTrend = (trendsData) => {
  if (trendsData.length < 2) return { accuracy: 0, intensity: 0, responseTime: 0 };
  
  const recent = trendsData.slice(-5); // Look at last 5 tests
  if (recent.length < 2) return { accuracy: 0, intensity: 0, responseTime: 0 };

  const firstPoint = recent[0];
  const lastPoint = recent[recent.length - 1];

  return {
    accuracy: Math.round(lastPoint.accuracy - firstPoint.accuracy),
    intensity: Number((lastPoint.intensity - firstPoint.intensity).toFixed(1)),
    responseTime: Number((lastPoint.responseTime - firstPoint.responseTime).toFixed(1))
  };
};

const generateTrendSummary = (trend) => {
  const parts = [];
  
  if (trend.accuracy !== 0) {
    parts.push(`Accuracy has ${trend.accuracy > 0 ? 'improved' : 'declined'} by ${Math.abs(trend.accuracy)}%`);
  }
  
  if (trend.intensity !== 0) {
    parts.push(`Sensation intensity has ${trend.intensity > 0 ? 'increased' : 'decreased'} by ${Math.abs(trend.intensity)} points`);
  }

  if (trend.responseTime !== 0) {
    parts.push(`Response time has ${trend.responseTime > 0 ? 'increased' : 'decreased'} by ${Math.abs(trend.responseTime)} seconds`);
  }

  if (parts.length === 0) {
    return 'No significant changes in recent tests.';
  }

  return parts.join('. ') + ' in recent tests.';
};

ResultSummary.propTypes = {
  tests: PropTypes.arrayOf(
    PropTypes.shape({
      startTime: PropTypes.string.isRequired,
      endTime: PropTypes.string.isRequired,
      stimulusApplied: PropTypes.bool.isRequired,
      stimulusPoints: PropTypes.arrayOf(PropTypes.string),
      patientResponse: PropTypes.shape({
        feltStimulus: PropTypes.bool.isRequired,
        intensity: PropTypes.number,
        location: PropTypes.arrayOf(PropTypes.string)
      }).isRequired
    })
  ).isRequired
};

export default ResultSummary;