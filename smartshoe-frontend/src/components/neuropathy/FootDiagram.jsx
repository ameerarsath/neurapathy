import React, { useState } from 'react'

const FootDiagram = ({ onLocationSelect, selectedLocation, viewType = 'top' }) => {
  const [hoveredRegion, setHoveredRegion] = useState(null)

  const footRegions = {
    top: [
      { id: 'big-toe', name: 'Big Toe', x: 45, y: 15, width: 15, height: 12 },
      { id: 'toes', name: 'Toes', x: 25, y: 18, width: 35, height: 10 },
      { id: 'ball', name: 'Ball of Foot', x: 25, y: 35, width: 50, height: 15 },
      { id: 'arch', name: 'Arch', x: 30, y: 55, width: 40, height: 20 },
      { id: 'heel', name: 'Heel', x: 35, y: 80, width: 30, height: 15 }
    ],
    bottom: [
      { id: 'toe-pads', name: 'Toe Pads', x: 25, y: 15, width: 50, height: 15 },
      { id: 'ball-bottom', name: 'Ball (Bottom)', x: 20, y: 35, width: 60, height: 20 },
      { id: 'arch-bottom', name: 'Arch (Bottom)', x: 25, y: 60, width: 50, height: 15 },
      { id: 'heel-bottom', name: 'Heel (Bottom)', x: 30, y: 80, width: 40, height: 15 }
    ]
  }

  const handleRegionClick = (region, event) => {
    const rect = event.currentTarget.getBoundingClientRect()
    const svgRect = event.currentTarget.closest('svg').getBoundingClientRect()
    
    const relativeX = ((event.clientX - svgRect.left) / svgRect.width) * 100
    const relativeY = ((event.clientY - svgRect.top) / svgRect.height) * 100
    
    const location = {
      region: region.id,
      name: region.name,
      x: Math.round(relativeX),
      y: Math.round(relativeY)
    }
    
    onLocationSelect(location)
  }

  const isSelected = (regionId) => {
    return selectedLocation && selectedLocation.region === regionId
  }

  const getRegionColor = (region) => {
    if (isSelected(region.id)) return '#ef4444' // red-500
    if (hoveredRegion === region.id) return '#f97316' // orange-500
    return '#e5e7eb' // gray-200
  }

  const FootShape = () => (
    <path
      d={viewType === 'top' 
        ? "M45 10 Q50 8 55 10 Q65 15 70 25 Q75 35 70 45 Q65 55 60 65 Q55 75 50 85 Q45 95 40 95 Q35 95 30 85 Q25 75 20 65 Q15 55 20 45 Q25 35 30 25 Q35 15 45 10 Z"
        : "M40 10 Q50 8 60 10 Q70 15 75 25 Q80 40 75 55 Q70 70 65 80 Q55 90 45 95 Q40 97 35 95 Q25 90 15 80 Q10 70 15 55 Q20 40 25 25 Q30 15 40 10 Z"
      }
      fill="#f3f4f6"
      stroke="#9ca3af"
      strokeWidth="1"
    />
  )

  return (
    <div className="flex flex-col items-center space-y-4">
      <div className="text-center">
        <h3 className="text-lg font-medium text-gray-900">
          {viewType === 'top' ? 'Top of Foot' : 'Bottom of Foot'}
        </h3>
        <p className="text-sm text-gray-600">Click where you felt the sensation</p>
      </div>
      
      <div className="relative">
        <svg
          viewBox="0 0 100 100"
          className="w-64 h-80 border border-gray-300 rounded-lg bg-white"
        >
          <FootShape />
          
          {footRegions[viewType].map((region) => (
            <g key={region.id}>
              <ellipse
                cx={region.x + region.width / 2}
                cy={region.y + region.height / 2}
                rx={region.width / 2}
                ry={region.height / 2}
                fill={getRegionColor(region)}
                stroke="#374151"
                strokeWidth="0.5"
                className="cursor-pointer transition-colors duration-200"
                onMouseEnter={() => setHoveredRegion(region.id)}
                onMouseLeave={() => setHoveredRegion(null)}
                onClick={(e) => handleRegionClick(region, e)}
              />
              
              <text
                x={region.x + region.width / 2}
                y={region.y + region.height / 2}
                textAnchor="middle"
                dominantBaseline="middle"
                className="text-xs font-medium fill-gray-700 pointer-events-none"
                fontSize="3"
              >
                {region.name.split(' ')[0]}
              </text>
            </g>
          ))}
          
          {selectedLocation && (
            <circle
              cx={selectedLocation.x}
              cy={selectedLocation.y}
              r="2"
              fill="#dc2626"
              stroke="#ffffff"
              strokeWidth="0.5"
            />
          )}
        </svg>
        
        {hoveredRegion && (
          <div className="absolute top-0 left-full ml-4 bg-gray-800 text-white px-2 py-1 rounded text-sm">
            {footRegions[viewType].find(r => r.id === hoveredRegion)?.name}
          </div>
        )}
      </div>
      
      {selectedLocation && (
        <div className="text-center p-3 bg-blue-50 rounded-lg">
          <p className="text-sm font-medium text-blue-900">
            Selected: {selectedLocation.name}
          </p>
          <p className="text-xs text-blue-700">
            Position: ({selectedLocation.x}, {selectedLocation.y})
          </p>
        </div>
      )}
    </div>
  )
}

export default FootDiagram