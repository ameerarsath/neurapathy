import React, { useMemo } from 'react';
import PropTypes from 'prop-types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area } from 'recharts';

const TestChart = ({ tests }) => {
  const chartData = useMemo(() => {
    return tests.map(test => {
      const accuracy = calculateTestAccuracy(test);
      const avgAccuracy = calculateMovingAverage(tests, test, 'accuracy');
      const confidenceInterval = calculateConfidenceInterval(tests, test);
      
      return {
        date: new Date(test.startTime).toLocaleDateString(),
        accuracy,
        avgAccuracy,
        confidenceLower: avgAccuracy - confidenceInterval,
        confidenceUpper: avgAccuracy + confidenceInterval,
        responseTime: test.responseTime || 0,
        intensity: test.patientResponse.intensity || 0,
        avgIntensity: calculateMovingAverage(tests, test, 'intensity')
      };
    });
  }, [tests]);

  const calculateTestAccuracy = (test) => {
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

  const calculateMovingAverage = (tests, currentTest, metric) => {
    const window = 3; // 3-test moving average
    const currentIndex = tests.findIndex(t => t === currentTest);
    const startIndex = Math.max(0, currentIndex - window + 1);
    const relevantTests = tests.slice(startIndex, currentIndex + 1);
    
    if (metric === 'accuracy') {
      return relevantTests.reduce((sum, t) => sum + calculateTestAccuracy(t), 0) / relevantTests.length;
    }
    return relevantTests.reduce((sum, t) => sum + (t.patientResponse[metric] || 0), 0) / relevantTests.length;
  };

  const calculateConfidenceInterval = (tests, currentTest) => {
    const window = 3;
    const currentIndex = tests.findIndex(t => t === currentTest);
    const startIndex = Math.max(0, currentIndex - window + 1);
    const relevantTests = tests.slice(startIndex, currentIndex + 1);
    
    if (relevantTests.length < 2) return 0;

    const accuracies = relevantTests.map(t => calculateTestAccuracy(t));
    const mean = accuracies.reduce((sum, acc) => sum + acc, 0) / accuracies.length;
    const squaredDiffs = accuracies.map(acc => Math.pow(acc - mean, 2));
    const variance = squaredDiffs.reduce((sum, diff) => sum + diff, 0) / (accuracies.length - 1);
    const standardError = Math.sqrt(variance / accuracies.length);
    
    return 1.96 * standardError; // 95% confidence interval
  };

  const getPercentileRank = (value, metric) => {
    const values = tests.map(test => metric === 'accuracy' ? calculateTestAccuracy(test) : test.patientResponse[metric] || 0);
    const sortedValues = [...values].sort((a, b) => a - b);
    const index = sortedValues.findIndex(v => v >= value);
    return Math.round((index / sortedValues.length) * 100);
  };

  return (
    <div className="space-y-6">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          
          {/* Confidence interval area */}
          <Area
            yAxisId="left"
            dataKey="confidenceUpper"
            stroke="none"
            fill="#8884d8"
            fillOpacity={0.1}
          />
          <Area
            yAxisId="left"
            dataKey="confidenceLower"
            stroke="none"
            fill="#8884d8"
            fillOpacity={0.1}
          />
          
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="accuracy"
            name="Accuracy"
            stroke="#8884d8"
            dot={true}
          />
          <Line
            yAxisId="left"
            type="monotone"
            dataKey="avgAccuracy"
            name="Avg Accuracy"
            stroke="#8884d8"
            strokeDasharray="5 5"
            dot={false}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="intensity"
            name="Intensity"
            stroke="#82ca9d"
            dot={true}
          />
          <Line
            yAxisId="right"
            type="monotone"
            dataKey="avgIntensity"
            name="Avg Intensity"
            stroke="#82ca9d"
            strokeDasharray="5 5"
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>

      {tests.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-900">Latest Performance</h4>
            <div className="mt-2 space-y-2">
              <p className="text-sm text-gray-600">
                Accuracy: {chartData[chartData.length - 1].accuracy.toFixed(1)}%
                <span className="ml-2 text-xs text-gray-500">
                  (Percentile: {getPercentileRank(chartData[chartData.length - 1].accuracy, 'accuracy')}%)
                </span>
              </p>
              <p className="text-sm text-gray-600">
                Intensity: {chartData[chartData.length - 1].intensity}
                <span className="ml-2 text-xs text-gray-500">
                  (Percentile: {getPercentileRank(chartData[chartData.length - 1].intensity, 'intensity')}%)
                </span>
              </p>
            </div>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-gray-900">Trend Analysis</h4>
            <div className="mt-2 space-y-2">
              {calculateTrend(chartData)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const calculateTrend = (data) => {
  if (data.length < 2) return null;

  const recentTests = data.slice(-5);
  const accuracyChange = recentTests[recentTests.length - 1].accuracy - recentTests[0].accuracy;
  const intensityChange = recentTests[recentTests.length - 1].intensity - recentTests[0].intensity;

  return (
    <>
      <p className={`text-sm ${accuracyChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
        Accuracy: {accuracyChange > 0 ? '+' : ''}{accuracyChange.toFixed(1)}% over last 5 tests
      </p>
      <p className={`text-sm ${intensityChange >= 0 ? 'text-green-600' : 'text-red-600'}`}>
        Intensity: {intensityChange > 0 ? '+' : ''}{intensityChange.toFixed(1)} over last 5 tests
      </p>
    </>
  );
};

TestChart.propTypes = {
  tests: PropTypes.arrayOf(
    PropTypes.shape({
      startTime: PropTypes.string.isRequired,
      stimulusApplied: PropTypes.bool.isRequired,
      stimulusPoints: PropTypes.arrayOf(PropTypes.string),
      responseTime: PropTypes.number,
      patientResponse: PropTypes.shape({
        feltStimulus: PropTypes.bool.isRequired,
        intensity: PropTypes.number,
        location: PropTypes.arrayOf(PropTypes.string)
      }).isRequired
    })
  ).isRequired
};

export default TestChart;