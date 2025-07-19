import React, { useState } from 'react';
import PropTypes from 'prop-types';

const FootDiagram = ({ side = 'top', onLocationSelect, selectedPoints = [], isPhysicianView = false, stimulusPoints = [] }) => {
  const [hoveredZone, setHoveredZone] = useState(null);

  // Define foot zones - these correspond to the diagram
  const footZones = {
    top: [
      { id: 't1', path: 'M20,10 Q25,5 30,10 L35,30 L25,30 Z', label: 'Big Toe' },
      { id: 't2', path: 'M40,12 Q45,8 50,12 L52,25 L42,25 Z', label: '2nd Toe' },
      { id: 't3', path: 'M55,13 Q60,10 65,13 L67,23 L57,23 Z', label: '3rd Toe' },
      { id: 't4', path: 'M70,15 Q75,12 80,15 L82,22 L72,22 Z', label: '4th Toe' },
      { id: 't5', path: 'M85,18 Q90,15 95,18 L97,21 L87,21 Z', label: '5th Toe' },
      { id: 'f1', path: 'M20,30 Q50,35 80,30 L85,60 Q50,70 15,60 Z', label: 'Ball of Foot' },
      { id: 'h1', path: 'M15,60 Q50,70 85,60 L90,100 Q50,110 10,100 Z', label: 'Heel' },
    ],
    bottom: [
      { id: 'a1', path: 'M20,30 Q50,35 80,30 L85,45 Q50,50 15,45 Z', label: 'Arch Front' },
      { id: 'a2', path: 'M15,45 Q50,50 85,45 L88,75 Q50,80 12,75 Z', label: 'Arch Middle' },
      { id: 'h2', path: 'M12,75 Q50,80 88,75 L90,100 Q50,110 10,100 Z', label: 'Heel Bottom' },
    ],
  };

  const handleZoneClick = (zoneId) => {
    if (onLocationSelect) {
      onLocationSelect(zoneId);
    }
  };

  return (
    <div className="relative w-64 h-96">
      <svg
        viewBox="0 0 100 120"
        className="w-full h-full"
      >
        {footZones[side].map((zone) => (
          <g key={zone.id}>
            <path
              d={zone.path}
              fill={selectedPoints.includes(zone.id) ? '#90cdf4' : '#e2e8f0'}
              stroke="#2d3748"
              strokeWidth="1"
              onClick={() => handleZoneClick(zone.id)}
              onMouseEnter={() => setHoveredZone(zone.id)}
              onMouseLeave={() => setHoveredZone(null)}
              className="cursor-pointer transition-colors duration-200 hover:fill-blue-200"
            />
            {isPhysicianView && stimulusPoints.includes(zone.id) && (
              <circle
                cx={zone.id.startsWith('t') ? 30 + (parseInt(zone.id[1]) - 1) * 15 : 50}
                cy={zone.id.startsWith('t') ? 15 : zone.id.startsWith('f') ? 45 : 85}
                r="3"
                fill="red"
                className="animate-pulse"
              />
            )}
          </g>
        ))}
      </svg>
      {hoveredZone && (
        <div className="absolute bottom-0 left-0 right-0 text-center bg-gray-800 text-white py-1 text-sm rounded">
          {footZones[side].find(z => z.id === hoveredZone)?.label}
        </div>
      )}
    </div>
  );
};

FootDiagram.propTypes = {
  side: PropTypes.oneOf(['top', 'bottom']),
  onLocationSelect: PropTypes.func,
  selectedPoints: PropTypes.arrayOf(PropTypes.string),
  isPhysicianView: PropTypes.bool,
  stimulusPoints: PropTypes.arrayOf(PropTypes.string),
};

export default FootDiagram;
