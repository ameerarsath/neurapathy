import React, { useMemo } from 'react';
import PropTypes from 'prop-types';

const LocationHeatmap = ({ tests }) => {
  const heatmapData = useMemo(() => {
    const locationStats = {};
    
    tests.forEach(test => {
      const stimulusSet = new Set(test.stimulusPoints || []);
      const responseSet = new Set(test.patientResponse.location || []);
      
      [...new Set([...stimulusSet, ...responseSet])].forEach(location => {
        if (!locationStats[location]) {
          locationStats[location] = { total: 0, correct: 0 };
        }
        locationStats[location].total++;
        if (stimulusSet.has(location) === responseSet.has(location)) {
          locationStats[location].correct++;
        }
      });
    });

    return Object.entries(locationStats).map(([location, stats]) => ({
      location,
      accuracy: (stats.correct / stats.total) * 100,
      total: stats.total
    }));
  }, [tests]);

  const getColorForAccuracy = (accuracy) => {
    if (accuracy >= 90) return 'bg-green-500';
    if (accuracy >= 70) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="p-4">
      <h3 className="text-lg font-medium text-gray-900 mb-4">Location Accuracy</h3>
      <div className="grid grid-cols-2 gap-4">
        {heatmapData.map(({ location, accuracy, total }) => (
          <div key={location} className="flex items-center space-x-2">
            <div 
              className={`w-4 h-4 rounded ${getColorForAccuracy(accuracy)}`} 
              title={`${accuracy.toFixed(1)}% accuracy`}
            />
            <span className="text-sm text-gray-600">
              {location}: {accuracy.toFixed(1)}% ({total} tests)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

LocationHeatmap.propTypes = {
  tests: PropTypes.arrayOf(
    PropTypes.shape({
      stimulusPoints: PropTypes.arrayOf(PropTypes.string),
      patientResponse: PropTypes.shape({
        location: PropTypes.arrayOf(PropTypes.string)
      })
    })
  ).isRequired
};

export default LocationHeatmap;
